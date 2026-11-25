"""Train a WunderNN-inspired spectral fingerprint ranker or generate CASMI guesses."""

import argparse
import csv
import json
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from baseline import MAX_GUESSES, read_queries, search_library
from wunder_data import ADDUCT_INDEX, FP_BITS, META_SIZE, SPECTRUM_COLUMNS, encode_rows

MASS_BINS = 1024
TOKEN_COUNT = 16
TOKEN_WIDTH = MASS_BINS // TOKEN_COUNT
HIDDEN = 64


class SpectralExperts(nn.Module):
    """Blend ordered fragment and neutral-loss experts with a whole-spectrum MLP."""

    def __init__(self, hidden=HIDDEN, layers=1):
        super().__init__()
        HIDDEN = hidden
        view, pool = 2 * hidden, 3 * hidden
        self.adduct = nn.Embedding(len(ADDUCT_INDEX) + 1, 8)
        self.context = nn.Sequential(nn.Linear(META_SIZE + 8, 32), nn.SiLU())
        self.token = nn.Sequential(nn.Linear(TOKEN_WIDTH * 4, HIDDEN), nn.LayerNorm(HIDDEN), nn.SiLU())
        self.sequence = nn.GRU(HIDDEN, HIDDEN, num_layers=layers, batch_first=True)
        self.fragment = nn.Sequential(nn.Linear(HIDDEN * 2 + 32, view), nn.LayerNorm(view), nn.SiLU())
        self.loss = nn.Sequential(nn.Linear(HIDDEN * 2 + 32, view), nn.LayerNorm(view), nn.SiLU())
        self.pooled = nn.Sequential(nn.Linear(MASS_BINS * 2 + 32, pool), nn.LayerNorm(pool), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(view, FP_BITS), nn.Linear(view, FP_BITS), nn.Linear(pool, FP_BITS)])
        self.route = nn.Sequential(nn.Linear(32 + view * 2 + pool, 64), nn.SiLU(), nn.Linear(64, 3))
        self.dropout = nn.Dropout(0.1)

        steps = torch.arange(TOKEN_COUNT)
        lag = steps[:, None] - steps[None, :]
        rows = []
        for decay in (0.5, 0.875):
            weights = (1 - decay) * decay ** lag.clamp_min(0) * (lag >= 0)
            weights[:, 0] = decay ** steps
            rows.append(weights)
        self.register_buffer("ema", torch.stack(rows), persistent=False)
        self.register_buffer(
            "context_scale", torch.tensor([500.0, 500.0, 1.0, 50.0, 48.0, 1.0]),
            persistent=False,
        )

    def forward(self, peaks, meta, adducts):
        mzs, intensity = peaks.unbind(dim=-1)
        positions = torch.stack((mzs, meta[:, :1] - mzs), dim=1).round().long()
        valid = (positions > 0) & (positions < MASS_BINS)
        binned = peaks.new_zeros((len(peaks), 2, MASS_BINS))
        binned.scatter_add_(2, positions.clamp(0, MASS_BINS - 1), intensity[:, None, :] * valid)
        binned = torch.log1p(binned)

        context = self.context(torch.cat(((meta / self.context_scale).clamp(-4, 4), self.adduct(adducts)), dim=-1))
        tokens = binned.reshape(-1, TOKEN_COUNT, TOKEN_WIDTH)
        delta = torch.diff(tokens, dim=1, prepend=tokens[:, :1])
        smooth_fast = torch.matmul(self.ema[0], tokens)
        smooth_slow = torch.matmul(self.ema[1], tokens)
        sequence, _ = self.sequence(self.token(torch.cat((tokens, delta, tokens - smooth_fast, tokens - smooth_slow), -1)))
        summary = torch.cat((sequence[:, -1], sequence.amax(dim=1)), dim=-1).reshape(len(peaks), 2, -1)

        views = (
            self.fragment(torch.cat((summary[:, 0], context), dim=-1)),
            self.loss(torch.cat((summary[:, 1], context), dim=-1)),
            self.pooled(torch.cat((binned.flatten(1), context), dim=-1)),
        )
        gate = self.route(torch.cat((context, *views), dim=-1)).softmax(dim=-1)
        outputs = torch.stack([head(self.dropout(view)) for head, view in zip(self.heads, views)], dim=1)
        return (outputs * gate.unsqueeze(-1)).sum(dim=1), outputs


class Catalog:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.keys = data["keys"]
            self.smiles = data["smiles"]
            self.masses = data["masses"]
            self.fingerprints = data["fingerprints"]
            self.counts = data["counts"]
            self.natural = data["natural"]

    def candidates(self, neutral):
        masses = neutral[np.isfinite(neutral) & (neutral > 0)]
        if not len(masses):
            return np.empty(0, dtype=int), np.empty(0)
        tolerance = np.maximum(0.003, masses * 15e-6)
        starts = self.masses.searchsorted(masses - tolerance)
        stops = self.masses.searchsorted(masses + tolerance, side="right")
        indices = np.unique(np.concatenate([np.arange(start, stop) for start, stop in zip(starts, stops)]))
        relative_error = (self.masses[indices, None] - masses) / tolerance
        error = np.min(np.abs(relative_error), axis=1)
        return indices, error

    def nearest(self, neutral):
        masses = neutral[np.isfinite(neutral) & (neutral > 0)]
        mass = np.median(masses) if len(masses) else np.median(self.masses)
        middle = self.masses.searchsorted(mass)
        closest = np.arange(max(0, middle - MAX_GUESSES), min(len(self.keys), middle + MAX_GUESSES))
        return closest[np.argsort(np.abs(self.masses[closest] - mass), kind="stable")]

    def structure(self, index):
        return self.smiles[index].decode("utf-8")


def ranking_scores(fps, error, logits, prior, natural, mass_weight, natural_weight, network_weight=1.0):
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -12, 12)))
    present = fps.sum(axis=1)
    overlap = fps @ probabilities
    soft_tanimoto = overlap / np.maximum(present + probabilities.sum() - overlap, 1e-6)
    likelihood = fps @ (logits - prior) / np.sqrt(np.maximum(present, 1))
    evidence = soft_tanimoto + 0.025 * likelihood
    return network_weight * evidence - mass_weight * error + natural_weight * natural


def neutral_estimates(meta, shift_ppm):
    return meta[:, 1].astype(float) / (1 + meta[:, 5] * shift_ppm * 1e-6)


def calibrate_mass(catalog, keys, meta):
    known = np.isin(catalog.keys, np.unique(keys))
    true_masses = dict(zip(catalog.keys[known], catalog.masses[known]))
    errors = []
    for key, positions in group_indices(keys).items():
        truth = true_masses.get(key)
        if truth is None or truth <= 0:
            continue
        rows = meta[positions]
        errors_ppm = (rows[rows[:, 5] == 1, 1] / truth - 1) * 1e6
        inliers = errors_ppm[np.abs(errors_ppm) <= 30]
        if len(inliers):
            errors.append(np.median(inliers))
    return float(np.median(errors)) if errors else 0.0


def group_indices(keys):
    groups = defaultdict(list)
    for index, key in enumerate(keys):
        groups[key].append(index)
    return groups


def evaluation_groups(catalog, keys, meta, shift_ppm):
    panels = []
    masses = neutral_estimates(meta, shift_ppm)
    for truth, positions in group_indices(keys).items():
        rows, error = catalog.candidates(masses[positions])
        target = np.flatnonzero(catalog.keys[rows] == truth)
        fps = None
        target_index = -1
        if len(target):
            fps = np.unpackbits(catalog.fingerprints[rows], axis=1).astype(np.float32)
            target_index = int(target[0])
        panels.append((positions, fps, error, catalog.natural[rows], target_index))
    return panels


def evaluate(logits, prior, panels, mass_weight, natural_weight, network_weight=1.0):
    reciprocal = hit1 = mass_reciprocal = mass_natural_reciprocal = coverage = 0.0
    for positions, fps, error, natural, target in panels:
        if target < 0:
            continue
        coverage += 1
        scores = ranking_scores(
            fps, error, logits[positions].mean(axis=0), prior, natural,
            mass_weight, natural_weight, network_weight,
        )
        order = np.argsort(-scores, kind="stable")
        rank = int(np.flatnonzero(order == target)[0]) + 1
        mass_rank = int(np.flatnonzero(np.argsort(error, kind="stable") == target)[0]) + 1
        prior_order = np.argsort(mass_weight * error - natural_weight * natural, kind="stable")
        mass_natural_rank = int(np.flatnonzero(prior_order == target)[0]) + 1
        reciprocal += 1 / rank if rank <= MAX_GUESSES else 0
        mass_reciprocal += 1 / mass_rank if mass_rank <= MAX_GUESSES else 0
        mass_natural_reciprocal += 1 / mass_natural_rank if mass_natural_rank <= MAX_GUESSES else 0
        hit1 += rank == 1
    count = len(panels)
    return {
        "molecules": count, "mrr25": reciprocal / count, "mass_only_mrr25": mass_reciprocal / count,
        "mass_coconut_mrr25": mass_natural_reciprocal / count,
        "hit1": hit1 / count, "candidate_recall": coverage / count,
    }


def hard_candidate_rows(catalog, training_keys, labels, count, rng):
    lookup = {key: row for row, key in enumerate(catalog.keys)}
    unique, first = np.unique(labels, return_index=True)
    rows = np.zeros((int(unique.max()) + 1, count + 1), dtype=np.int32)
    valid = np.zeros_like(rows, dtype=bool)
    for label, key in zip(unique, training_keys[first]):
        target = lookup[key]
        options, _ = catalog.candidates(catalog.masses[target:target + 1])
        options = options[options != target]
        if len(options) > count:
            options = rng.choice(options, count, replace=False)
        rows[label, 0] = target
        rows[label, 1:len(options) + 1] = options
        valid[label, :len(options) + 1] = True
    return rows, valid


def retrieval_loss(logits, prior, fingerprints, valid):
    probabilities = logits.sigmoid()
    counts = fingerprints.sum(dim=-1)
    overlap = torch.bmm(fingerprints, probabilities.unsqueeze(-1)).squeeze(-1)
    tanimoto = overlap / (counts + probabilities.sum(dim=-1, keepdim=True) - overlap).clamp_min(1e-6)
    likelihood = torch.bmm(fingerprints, (logits - prior).unsqueeze(-1)).squeeze(-1)
    values = tanimoto + 0.025 * likelihood / counts.clamp_min(1).sqrt()
    # The positive is always column zero. Mask slots with no mass-matched negative.
    return -(values * 15).masked_fill(~valid, -1e4).log_softmax(dim=1)[:, 0].mean()


def device_for(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def inference(model, arrays, device, batch_size=256):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(arrays[0]), batch_size):
            inputs = [torch.from_numpy(values[start:start + batch_size]).to(device) for values in arrays]
            output, _ = model(*inputs)
            predictions.append(output.cpu().numpy())
    return np.concatenate(predictions)


def checkpoint_state(model):
    return {name: weight.detach().cpu().clone() for name, weight in model.state_dict().items()}


def averaged_states(states):
    # Mix aligned checkpoints from one training trajectory, retaining the newest member.
    return {
        name: 0.25 * states[-1][name] + (0.75 / len(states)) * sum(state[name] for state in states)
        for name in states[-1]
    }


def calibrate_blend(logits, prior, panels, mass_weight, natural_weight):
    policies = [
        {
            "network_weight": network_weight,
            **evaluate(logits, prior, panels, mass_weight, natural_weight, network_weight),
        }
        for network_weight in (0, 0.10, 0.25, 0.50, 0.75, 1.0, 1.50, 2.0)
    ]
    return max(policies, key=lambda policy: policy["mrr25"])


def train(args, device):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    with np.load(args.directory / "spectra.npz", allow_pickle=False) as data:
        features = tuple(data[name] for name in ("peaks", "meta", "adducts"))
        stages = data["split"]
        labels, keys, packed = data["labels"], data["keys"], data["fingerprints"]
    training = np.flatnonzero(stages == 0)
    development = np.flatnonzero(stages == 1)
    audit = np.flatnonzero(stages == 2)
    if np.intersect1d(keys[training], keys[stages != 0]).size:
        raise ValueError("Molecule overlap between training and evaluation")

    catalog = Catalog(args.catalog or args.directory / "catalog.npz")
    shift_ppm = calibrate_mass(catalog, keys[development], features[1][development])
    dev_panel = evaluation_groups(catalog, keys[development], features[1][development], shift_ppm)
    audit_panel = evaluation_groups(catalog, keys[audit], features[1][audit], shift_ppm)
    dev_features = tuple(values[development] for values in features)
    audit_features = tuple(values[audit] for values in features)
    positive = np.zeros(FP_BITS, dtype=np.float64)
    for start in range(0, len(training), 4096):
        batch = training[start:start + 4096]
        positive += np.unpackbits(packed[labels[batch]], axis=1).sum(axis=0)
    frequency = ((1 + positive) / (2 + len(training))).clip(0.001, 0.999)
    prior = np.log(frequency / (1 - frequency)).astype(np.float32)
    prior_device = torch.from_numpy(prior).to(device)
    if args.hard_negatives:
        negative_rows, negative_mask = hard_candidate_rows(
            catalog, keys[training], labels[training], args.hard_negatives, rng
        )
        print(f"Prepared mass-matched training alternatives with {args.hard_negatives} slots", flush=True)
    config = {"hidden": args.hidden, "layers": args.layers}
    model = SpectralExperts(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs * math.ceil(len(training) / args.batch_size), eta_min=0.0001,
    )
    recent = deque(maxlen=3)
    best_score = -1.0
    history = []
    print(f"Training {len(training):,} spectra on {device} with timsTOF shift {shift_ppm:+.2f} ppm", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(training)
        losses = torch.zeros((), device=device)
        for start in range(0, len(order), args.batch_size):
            rows = order[start:start + args.batch_size]
            inputs = [torch.from_numpy(values[rows]).to(device) for values in features]
            targets = torch.from_numpy(np.unpackbits(packed[labels[rows]], axis=1)).to(
                device=device, dtype=torch.float32
            )
            optimizer.zero_grad(set_to_none=True)
            scores, experts = model(*inputs)
            objective = F.binary_cross_entropy_with_logits(scores, targets)
            objective += 0.15 * F.binary_cross_entropy_with_logits(experts, targets[:, None].expand_as(experts))
            if args.hard_negatives:
                negatives = catalog.fingerprints[negative_rows[labels[rows]]]
                fingerprints = torch.from_numpy(np.unpackbits(negatives, axis=2)).to(
                    device=device, dtype=torch.float32,
                )
                valid = torch.from_numpy(negative_mask[labels[rows]]).to(device)
                objective += 0.12 * retrieval_loss(scores, prior_device, fingerprints, valid)
            objective.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            losses += objective.detach() * len(rows)
        current = checkpoint_state(model)
        recent.append(current)
        variants = [("single", current)]
        if len(recent) > 1:
            variants.append(("soup", averaged_states(recent)))
        for variant, state in variants:
            model.load_state_dict(state)
            logits = inference(model, dev_features, device)
            policies = [
                {
                    "mass_weight": mass_weight,
                    "natural_weight": natural_weight,
                    **evaluate(logits, prior, dev_panel, mass_weight, natural_weight),
                }
                for mass_weight in (0.025, 0.10, 0.25, 0.50)
                for natural_weight in (0.0, 0.03)
            ]
            dev_metrics = max(policies, key=lambda row: row["mrr25"])
            row = {"epoch": epoch, "variant": variant, "train_loss": float(losses.cpu()) / len(training), **dev_metrics}
            history.append(row)
            print(json.dumps(row), flush=True)
            if dev_metrics["mrr25"] > best_score:
                best_score = dev_metrics["mrr25"]
                best = state, row
        model.load_state_dict(current)

    state, selection = best
    model.load_state_dict(state)
    mass_weight, natural_weight = selection["mass_weight"], selection["natural_weight"]
    blend = calibrate_blend(inference(model, dev_features, device), prior, dev_panel, mass_weight, natural_weight)
    policy = {
        "mass_weight": mass_weight, "natural_weight": natural_weight, "network_weight": blend["network_weight"],
    }
    audit_metrics = evaluate(inference(model, audit_features, device), prior, audit_panel, **policy)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.checkpoint.with_suffix(".partial")
    torch.save(
        {
            "model": state, "prior": torch.from_numpy(prior), "shift_ppm": shift_ppm, "config": config, **policy,
        },
        temporary,
    )
    temporary.replace(args.checkpoint)
    report = {
        "seed": args.seed, "shift_ppm": shift_ppm, "selected": selection,
        "development_blend": blend, "audit": audit_metrics, "history": history,
    }
    args.checkpoint.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Audit {json.dumps(audit_metrics)}", flush=True)


def predict(data, catalog_path, checkpoint, output, device, use_library=True):
    catalog = Catalog(catalog_path)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = SpectralExperts(**saved.get("config", {})).to(device)
    model.load_state_dict(saved["model"])
    prior = saved["prior"].numpy()
    test_path = data / "test.parquet"
    test = pq.read_table(test_path, columns=["molecule_id", *SPECTRUM_COLUMNS]).to_pylist()
    peaks, meta, adducts = encode_rows(test)
    logits = inference(model, (peaks, meta, adducts), device)
    neutral = neutral_estimates(meta, saved.get("shift_ppm", 0.0))
    mass_weight = saved.get("mass_weight", 0.025)
    natural_weight = saved.get("natural_weight", 0.0)
    network_weight = saved.get("network_weight", 1.0)
    groups = group_indices([row["molecule_id"] for row in test])
    references = {}
    if use_library:
        library = pq.ParquetFile(data / "train.parquet")
        references = search_library(library, read_queries(test_path), min_shared_peaks=4)

    output.parent.mkdir(parents=True, exist_ok=True)
    promoted = 0
    temporary = output.with_suffix(".partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_id", "smiles"])
        for molecule_id in sorted(groups):
            positions = groups[molecule_id]
            masses = neutral[positions]
            rows, error = catalog.candidates(masses)
            if len(rows):
                fps = np.unpackbits(catalog.fingerprints[rows], axis=1).astype(np.float32)
                scores = ranking_scores(
                    fps, error, logits[positions].mean(axis=0), prior, catalog.natural[rows],
                    mass_weight, natural_weight, network_weight,
                )
                order = rows[np.lexsort((-catalog.counts[rows], -scores))]
            else:
                order = catalog.nearest(masses)
            allowed = set(catalog.keys[rows])
            best_refs = sorted(references.get(molecule_id, {}).items(), key=lambda item: item[1], reverse=True)
            winner = next(
                ((key, smiles) for key, ((similarity, _, _), smiles) in best_refs
                 if similarity >= 0.80 and key in allowed),
                None,
            )
            guesses = dict([winner]) if winner else {}
            promoted += bool(guesses)
            seen_smiles = set(guesses.values())
            for idx in order:
                key, smiles = catalog.keys[idx], catalog.structure(idx)
                if key not in guesses and smiles not in seen_smiles:
                    guesses[key] = smiles
                    seen_smiles.add(smiles)
                if len(guesses) >= MAX_GUESSES:
                    break
            writer.writerow([molecule_id, ";".join(list(guesses.values())[:MAX_GUESSES])])
    temporary.replace(output)
    print(f"Wrote {len(groups)} molecules with {promoted} confident spectral matches to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_subparsers(dest="action", required=True)
    fit = action.add_parser("train")
    fit.add_argument("--directory", type=Path, default=Path("data/wunder"))
    fit.add_argument("--catalog", type=Path)
    fit.add_argument("--checkpoint", type=Path, default=Path("outputs/wunder-hard.pt"))
    fit.add_argument("--epochs", type=int, default=8)
    fit.add_argument("--batch-size", type=int, default=256)
    fit.add_argument("--hard-negatives", type=int, default=8)
    fit.add_argument("--seed", type=int, default=19)
    fit.add_argument("--hidden", type=int, default=HIDDEN)
    fit.add_argument("--layers", type=int, default=1)
    fit.add_argument("--device", default="auto")
    inference_parser = action.add_parser("predict")
    inference_parser.add_argument("--data", type=Path, default=Path("data"))
    inference_parser.add_argument("--catalog", type=Path, default=Path("data/wunder/catalog-coconut.npz"))
    inference_parser.add_argument("--checkpoint", type=Path, default=Path("outputs/wunder-hard.pt"))
    inference_parser.add_argument("--output", type=Path, default=Path("outputs/wunder/submission.csv"))
    inference_parser.add_argument("--device", default="auto")
    inference_parser.add_argument("--no-library", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(min(6, torch.get_num_threads()))
    device = device_for(args.device)
    if args.action == "train":
        if args.epochs < 1 or args.batch_size < 1 or args.hard_negatives < 0:
            parser.error("Expected positive epoch and batch counts and a nonnegative hard-negative count")
        train(args, device)
    else:
        predict(args.data, args.catalog, args.checkpoint, args.output, device, not args.no_library)


if __name__ == "__main__":
    main()
