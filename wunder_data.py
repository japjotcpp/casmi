"""CASMI spectrum tensors, molecule splits, and a searchable structure catalog."""

import argparse
import csv
import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from baseline import PEAK_COLUMNS

PEAK_LIMIT = 48
FP_BITS = 1024
EXAMPLE_LIBRARY = "enveda-np-examples"
SPECTRUM_COLUMNS = [
    "precursor_mz", "adduct", "ionization_mode", "instrument_type", "collision_energy_ev",
    *PEAK_COLUMNS,
]
ION_MASSES = {
    "[M+H]+": (1, 1.007276), "[M-H]-": (1, -1.007276),
    "[M+Na]+": (1, 22.989218), "[M+NH4]+": (1, 18.033823),
    "[M+K]+": (1, 38.963158), "[M+Cl]-": (1, 34.969401),
    "[M+CH2O2-H]-": (1, 44.998203), "[M+C2H4O2-H]-": (1, 59.013853),
    "[M-H2O+H]+": (1, -17.003289), "[M-2H2O+H]+": (1, -35.013854),
    "[M-H2O-H]-": (1, -19.017841), "[M]+": (1, -0.000549),
    "[M]-": (1, 0.000549), "[2M+H]+": (2, 1.007276),
    "[2M+Na]+": (2, 22.989218), "[2M-H]-": (2, -1.007276),
    "[2M+Na-2H]-": (2, 20.974666), "[2M+CH2O2-H]-": (2, 44.998203),
}
ADDUCT_INDEX = {adduct: idx for idx, adduct in enumerate(ION_MASSES, start=1)}
# Precursor m/z, neutral mass, polarity, collision energy, peak count and instrument.
META_SIZE = 6


def neutral_mass(mz, adduct, negative):
    if adduct == "[M+2H]2+":
        return mz * 2 - 2.014552
    if adduct == "[M-2H]2-":
        return mz * 2 + 2.014552
    molecules, shift = ION_MASSES.get(adduct, (1, -1.007276 if negative else 1.007276))
    return (mz - shift) / molecules


def encode_rows(rows):
    peaks = np.zeros((len(rows), PEAK_LIMIT, 2), dtype=np.float32)
    meta = np.zeros((len(rows), META_SIZE), dtype=np.float32)
    adduct_ids = np.zeros(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        precursor = row["precursor_mz"]
        if precursor is None or not np.isfinite(precursor) or precursor <= 0:
            continue
        mzs = np.asarray(row["ms2_mzs"] or [], dtype=np.float32)
        weights = np.asarray(row["ms2_normalized_intensities"] or [], dtype=np.float32)
        keep = np.isfinite(mzs) & np.isfinite(weights) & (weights > 0) & (mzs > 0)
        keep &= np.abs(mzs - precursor) > 0.05
        mzs, weights = mzs[keep], weights[keep]
        if len(weights) > PEAK_LIMIT:
            top = np.argpartition(weights, -PEAK_LIMIT)[-PEAK_LIMIT:]
            mzs, weights = mzs[top], weights[top]
        if len(weights):
            peaks[index, :len(mzs), 0] = mzs
            peaks[index, :len(mzs), 1] = np.sqrt(weights / weights.max())

        energy = np.asarray(row["collision_energy_ev"] or [], dtype=np.float32)
        energy = energy[np.isfinite(energy)]
        negative = row["ionization_mode"] == "negative"
        instrument = (row["instrument_type"] or "").lower()
        adduct = row["adduct"]
        meta[index] = [
            precursor, neutral_mass(precursor, adduct, negative), negative,
            np.median(energy) if len(energy) else 0, len(mzs), "timstof" in instrument,
        ]
        adduct_ids[index] = ADDUCT_INDEX.get(adduct, 0)
    return peaks, meta, adduct_ids


def structure_chunk(pairs):
    RDLogger.DisableLog("rdApp.*")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    table = []
    for key, smiles in pairs:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or not molecule.GetNumAtoms():
            continue
        key = key or Chem.MolToInchiKey(molecule).split("-")[0]
        if not key:
            continue
        fingerprint = generator.GetFingerprintAsNumPy(molecule)
        table.append((key, smiles, Descriptors.ExactMolWt(molecule), np.packbits(fingerprint)))
    return table


def featurize_structures(pairs, workers):
    if workers == 1:
        return structure_chunk(pairs)
    chunks = [pairs[start:start + 4000] for start in range(0, len(pairs), 4000)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return [row for chunk in executor.map(structure_chunk, chunks) for row in chunk]


def catalog_arrays(records, counts, external_keys):
    records = sorted(records, key=lambda row: (row[2], row[0]))
    keys, smiles, masses, fingerprints = zip(*records)
    return dict(
        keys=np.asarray(keys),
        smiles=np.asarray(smiles, dtype="S"),
        masses=np.asarray(masses, dtype=np.float64),
        fingerprints=np.stack(fingerprints),
        counts=np.asarray([counts.get(key, 0) for key in keys], dtype=np.int32),
        natural=np.asarray([key in external_keys for key in keys], dtype=bool),
    )


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def read_coconut(path, known):
    """Read a COCONUT CSV or its download ZIP by column name."""
    if path.suffix.lower() == ".zip":
        with ZipFile(path) as archive:
            name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
            with archive.open(name) as stream:
                return coconut_rows(io.TextIOWrapper(stream, encoding="utf-8-sig"), known)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return coconut_rows(stream, known)


def coconut_rows(stream, known):
    reader = csv.DictReader(stream)
    columns = {name.strip().lower(): name for name in reader.fieldnames or []}
    smiles_name = next(
        (columns[name] for name in ("canonical_smiles", "smiles", "isomeric_smiles") if name in columns),
        None,
    )
    key_name = next(
        (columns[name] for name in ("standard_inchi_key", "inchi_key", "inchikey") if name in columns),
        None,
    )
    if smiles_name is None:
        raise ValueError("COCONUT CSV has no SMILES column")
    structures = {}
    natural = set()
    for row in reader:
        smiles = (row[smiles_name] or "").strip()
        key = (row[key_name] or "").split("-")[0].strip() if key_name else ""
        if not smiles or "." in smiles:
            continue
        if len(key) == 14:
            natural.add(key)
            if key not in known:
                structures.setdefault(key, smiles)
        else:
            structures.setdefault(smiles, smiles)
    return [(key if key in natural else "", smiles) for key, smiles in structures.items()], natural


def extend_catalog(existing, coconut, output, workers):
    with np.load(existing, allow_pickle=False) as base:
        known = set(base["keys"].tolist())
        pairs, natural = read_coconut(coconut, known)
        added = {}
        for record in featurize_structures(pairs, workers):
            key = record[0]
            natural.add(key)
            if key not in known:
                added.setdefault(key, record)
        index = {name: base[name] for name in base.files}
    if added:
        supplementary = catalog_arrays(list(added.values()), {}, natural)
        for name in index:
            index[name] = np.concatenate((index[name], supplementary[name]))
    index["natural"] = np.isin(index["keys"], sorted(natural))
    order = np.argsort(index["masses"], kind="stable")
    save_arrays(output, **{name: values[order] for name, values in index.items()})
    print(f"Catalog has {len(order):,} structures including {len(added):,} added from COCONUT", flush=True)


def sample_rows(compound_ids, train_mask, limit, per_compound, rng):
    eligible = np.flatnonzero(train_mask)
    if not len(eligible):
        raise ValueError("No eligible training spectra")
    order = np.lexsort((rng.random(len(eligible)), compound_ids[eligible]))
    ordered = eligible[order]
    keys = compound_ids[ordered]
    starts = np.r_[True, keys[1:] != keys[:-1]]
    position = np.arange(len(keys))
    within_compound = position - np.maximum.accumulate(np.where(starts, position, 0))
    selected = ordered[within_compound < per_compound]
    if limit and len(selected) > limit:
        selected = rng.choice(selected, limit, replace=False)
    return selected


def prepare(train, spectrum_path, catalog_path, limit, per_compound, workers):
    rng = np.random.default_rng(20261121)
    library = pq.ParquetFile(train)
    metadata = library.read(columns=["inchikey14", "normalized_smiles", "ingest_lib"])
    keys = np.asarray(metadata["inchikey14"].to_pylist(), dtype=str)
    smiles = np.asarray(metadata["normalized_smiles"].to_pylist(), dtype=object)
    sources = np.asarray(metadata["ingest_lib"].to_pylist(), dtype=str)
    valid = (np.char.str_len(keys) == 14) & (smiles != "") & (smiles != None)
    row_ids = np.flatnonzero(valid)
    distinct, first, inverse, counts = np.unique(
        keys[row_ids], return_index=True, return_inverse=True, return_counts=True
    )
    compound_ids = np.full(len(keys), -1, dtype=np.int32)
    compound_ids[row_ids] = inverse
    domain = valid & (sources == EXAMPLE_LIBRARY)
    domain_ids = np.unique(compound_ids[domain])
    if len(domain_ids) < 2:
        raise ValueError("At least two natural-product examples are needed for dev and audit")
    rng.shuffle(domain_ids)
    audit_ids, dev_ids = np.array_split(domain_ids, 2)
    held = np.isin(compound_ids, domain_ids)
    train_rows = sample_rows(compound_ids, valid & ~held, limit, per_compound, rng)
    dev_rows = np.flatnonzero(domain & np.isin(compound_ids, dev_ids))
    audit_rows = np.flatnonzero(domain & np.isin(compound_ids, audit_ids))
    selected = np.concatenate((train_rows, dev_rows, audit_rows))
    split = np.repeat(np.arange(3, dtype=np.uint8), [len(train_rows), len(dev_rows), len(audit_rows)])
    order = np.argsort(selected)
    selected, split = selected[order], split[order]
    print(f"Selected {len(train_rows):,} training spectra and {len(domain_ids)} held-out structures", flush=True)

    if catalog_path.exists():
        with np.load(catalog_path, allow_pickle=False) as file:
            catalog = {name: file[name] for name in file.files}
    else:
        first_rows = row_ids[first]
        pairs = list(zip(distinct.tolist(), smiles[first_rows].tolist()))
        records = featurize_structures(pairs, workers)
        catalog = catalog_arrays(records, dict(zip(distinct, counts)), set())
        save_arrays(catalog_path, **catalog)
    catalog_index = {key: index for index, key in enumerate(catalog["keys"])}
    labels = np.array([catalog_index.get(keys[row], -1) for row in selected], dtype=np.int32)
    available = labels >= 0
    selected, split, labels = selected[available], split[available], labels[available]

    peaks = np.zeros((len(selected), PEAK_LIMIT, 2), dtype=np.float32)
    meta = np.zeros((len(selected), META_SIZE), dtype=np.float32)
    adducts = np.zeros(len(selected), dtype=np.int64)
    offset = start = 0
    for batch in library.iter_batches(batch_size=16384, columns=SPECTRUM_COLUMNS):
        stop = np.searchsorted(selected, offset + len(batch))
        if stop > start:
            chosen = batch.take(selected[start:stop] - offset).to_pylist()
            peaks[start:stop], meta[start:stop], adducts[start:stop] = encode_rows(chosen)
        offset += len(batch)
        start = stop
    usable = (meta[:, 0] > 0) & (meta[:, 4] > 0)
    selected_keys = catalog["keys"][labels[usable]]
    for stage in (0, 1, 2):
        if not np.any(split[usable] == stage):
            raise ValueError(f"No usable spectra in split {stage}")
    save_arrays(
        spectrum_path, peaks=peaks[usable], meta=meta[usable], adducts=adducts[usable],
        labels=labels[usable], split=split[usable], keys=selected_keys,
        fingerprints=catalog["fingerprints"],
    )
    print(f"Wrote {len(selected_keys):,} encoded spectra to {spectrum_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--directory", type=Path, default=Path("data/wunder"))
    parser.add_argument("--limit", type=int, default=200_000, help="Training spectra, or 0 for all eligible spectra")
    parser.add_argument("--per-compound", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--coconut", type=Path, help="Build the extended catalog from this COCONUT CSV or ZIP")
    args = parser.parse_args()
    if args.limit < 0 or args.per_compound < 1 or args.workers < 1:
        parser.error("Expected a nonnegative limit and positive per-compound and worker counts")
    catalog = args.directory / "catalog.npz"
    if args.coconut is None:
        prepare(args.train, args.directory / "spectra.npz", catalog, args.limit, args.per_compound, args.workers)
    else:
        extend_catalog(catalog, args.coconut, args.directory / "catalog-coconut.npz", args.workers)


if __name__ == "__main__":
    main()
