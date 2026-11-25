"""Retrieve candidate structures from the CASMI training spectral library."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
MAX_GUESSES = 25
MASS_TOLERANCE_PPM = 20
FRAGMENT_BIN_DA = 0.05
PEAK_COLUMNS = ["ms2_mzs", "ms2_normalized_intensities"]
LIBRARY_COLUMNS = ["precursor_mz", "ionization_mode", "adduct", "inchikey14", "normalized_smiles"]


def fragment_vector(mzs, intensities, precursor):
    """Return sorted bins and unit-normalized, square-root-weighted intensities."""
    if mzs is None or intensities is None:
        return np.empty(0, dtype=np.int32), np.empty(0)
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    mask = np.isfinite(mzs) & np.isfinite(intensities) & (intensities > 0)
    mask &= np.abs(mzs - precursor) > FRAGMENT_BIN_DA
    bins, inverse = np.unique(
        np.rint(mzs[mask] / FRAGMENT_BIN_DA).astype(np.int32), return_inverse=True
    )
    weights = np.sqrt(np.bincount(inverse, weights=intensities[mask]))
    if len(weights):
        weights /= np.linalg.norm(weights)
    return bins, weights


def cosine(query, reference):
    query_bins, query_weights = query
    ref_bins, ref_weights = reference
    _, query_indices, ref_indices = np.intersect1d(
        query_bins, ref_bins, assume_unique=True, return_indices=True
    )
    return float(query_weights[query_indices] @ ref_weights[ref_indices])


def read_queries(path):
    columns = ["molecule_id", "precursor_mz", "ionization_mode", "adduct", *PEAK_COLUMNS]
    molecules = defaultdict(list)
    for row in pq.read_table(path, columns=columns).to_pylist():
        molecule_id = row["molecule_id"]
        mass = row["precursor_mz"]
        if not molecule_id or mass is None or not np.isfinite(mass) or mass <= 0:
            raise ValueError("Test spectra require a molecule_id and a finite positive precursor m/z")
        vector = fragment_vector(*(row[column] for column in PEAK_COLUMNS), mass)
        molecules[molecule_id].append((mass, row["ionization_mode"], row["adduct"], vector))
    if not molecules:
        raise ValueError("Test data contains no molecules")
    return molecules


def search_library(library, molecules, min_shared_peaks=0):
    # Sort all query spectra by precursor m/z so each batch can skip unrelated rows.
    queries = sorted(
        [(molecule_id, *spectrum) for molecule_id, spectra in molecules.items() for spectrum in spectra],
        key=lambda query: query[1],
    )
    masses = np.array([query[1] for query in queries])
    fraction = MASS_TOLERANCE_PPM / 1e6
    ranks = defaultdict(dict)
    scanned = 0
    for batch in library.iter_batches(batch_size=8192, columns=LIBRARY_COLUMNS + PEAK_COLUMNS):
        precursor = batch.column("precursor_mz").to_numpy(zero_copy_only=False)
        starts = np.searchsorted(masses, precursor / (1 + fraction), side="left")
        stops = np.searchsorted(masses, precursor / (1 - fraction), side="right")
        selected = np.flatnonzero(starts < stops)
        for index, row in zip(selected, batch.take(selected).to_pylist()):
            smiles = row["normalized_smiles"]
            key = row["inchikey14"]
            if not smiles or not key:
                continue
            mass = precursor[index]
            mode = row["ionization_mode"]
            matching_queries = [
                query for query in queries[starts[index]:stops[index]]
                if mode == query[2]
            ]
            if not matching_queries:
                continue
            reference = fragment_vector(*(row[column] for column in PEAK_COLUMNS), mass)
            for molecule_id, query_mass, _, query_adduct, query_vector in matching_queries:
                if min_shared_peaks and np.intersect1d(
                    query_vector[0], reference[0], assume_unique=True
                ).size < min_shared_peaks:
                    continue
                score = (
                    cosine(query_vector, reference),
                    bool(query_adduct) and query_adduct == row["adduct"],
                    -abs(query_mass - mass) / query_mass,
                )
                previous = ranks[molecule_id].get(key)
                if previous is None or score > previous[0]:
                    ranks[molecule_id][key] = (score, smiles)
        scanned += len(batch)
        if scanned % (8192 * 128) == 0:
            print(f"Scanned {scanned:,}/{library.metadata.num_rows:,} reference spectra", flush=True)
    return ranks


def nearest_mass_fallback(library, molecules):
    """Find plausible guesses for queries with no reference in the narrow mass window."""
    best = defaultdict(dict)
    query_modes = {spectrum[1] for spectra in molecules.values() for spectrum in spectra}
    for batch in library.iter_batches(batch_size=65536, columns=LIBRARY_COLUMNS):
        masses = batch.column("precursor_mz").to_numpy(zero_copy_only=False)
        modes = np.asarray(batch.column("ionization_mode"))
        valid = np.flatnonzero(np.isfinite(masses) & (masses > 0))
        if not len(valid):
            continue
        ordered = valid[np.argsort(masses[valid], kind="stable")]
        by_mode = {}
        for mode in query_modes:
            same_mode = ordered[modes[ordered] == mode]
            rows = same_mode if len(same_mode) else ordered
            by_mode[mode] = rows, masses[rows]
        keys = batch.column("inchikey14")
        structures = batch.column("normalized_smiles")
        for molecule_id, spectra in molecules.items():
            candidates = best[molecule_id]
            for mass, mode, _, _ in spectra:
                rows, sorted_masses = by_mode[mode]
                middle = np.searchsorted(sorted_masses, mass)
                nearby = rows[max(0, middle - MAX_GUESSES):middle + MAX_GUESSES]
                for index in nearby:
                    key = keys[int(index)].as_py()
                    smiles = structures[int(index)].as_py()
                    if not key or not smiles:
                        continue
                    score = (modes[index] == mode, -abs(masses[index] - mass) / mass)
                    previous = candidates.get(key)
                    if previous is None or score > previous[0]:
                        candidates[key] = (score, smiles)
            if len(candidates) > MAX_GUESSES:
                best[molecule_id] = dict(
                    sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:MAX_GUESSES]
                )
    return best


def predict(data_dir, output):
    molecules = read_queries(data_dir / "test.parquet")
    library = pq.ParquetFile(data_dir / "train.parquet")
    print(f"Searching {library.metadata.num_rows:,} reference spectra for {len(molecules)} molecules", flush=True)
    ranks = search_library(library, molecules)
    missing = {key: spectra for key, spectra in molecules.items() if not ranks[key]}
    if missing:
        print(f"Using nearest precursor m/z for {len(missing)} molecules", flush=True)
        ranks.update(nearest_mass_fallback(library, missing))

    rows = []
    for molecule_id in molecules:
        candidates = ranks[molecule_id]
        ordered = sorted(candidates.values(), reverse=True)
        smiles = list(dict.fromkeys(smiles for _, smiles in ordered))[:MAX_GUESSES]
        if not smiles:
            raise ValueError(f"No usable reference SMILES for {molecule_id}")
        rows.append((molecule_id, ";".join(smiles)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_id", "smiles"])
        writer.writerows(sorted(rows))
    print(f"Wrote {len(rows)} molecules to {output}", flush=True)


def kaggle_data_dir():
    for test in Path("/kaggle/input").rglob("test.parquet"):
        if test.parent.name.casefold() == COMPETITION.casefold():
            return test.parent
    raise FileNotFoundError(f"Attach the {COMPETITION} competition data to this notebook")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    args = parser.parse_args()
    predict(args.data, args.output)


if __name__ == "__main__":
    main()
