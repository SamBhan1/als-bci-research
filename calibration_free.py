"""Calibration-free spelling for people with ALS: does seeding from other patients fix UMM's failure?

Compares, on the eight ALS participants of BNCI 2014-008 (source files with row/column codes):

  supervised      shrinkage LDA trained on the participant's own first 15 labelled letters (needs calibration)
  transfer        shrinkage LDA trained on the other seven patients, never adapted (no calibration)
  umm             UMM, current letter only (Sosulski & Tangermann 2023; no calibration, no other data)
  umm_memory      UMM with pooled covariance and confidence-weighted means from earlier letters
  seeded_umm      umm_memory whose mean memory starts from the other seven patients (this repository)

All calibration-free methods run as an online simulation: letter i is decoded from letters 1..i only, and no
label of the test participant is ever used.

    python calibration_free.py --source-dir data\\nm000169\\sourcedata
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.covariance import ledoit_wolf

FS = 256
BIN = int(0.05 * FS)          # 50 ms mean bins (fixed in advance)
N_BINS = 16                   # 0-800 ms after the flash
BASELINE = int(0.1 * FS)
FLASHES, CODES, LETTERS = 120, 12, 35
CALIBRATION_LETTERS = 15      # supervised reference: first 15 letters for training, last 20 for testing
SEED_WEIGHT = 5.0             # the other patients count as much as five confident letters (fixed in advance)
SEQUENCES = (1, 2, 3, 5, 7, 10)
SYMBOLS = [(row, col) for row in range(1, 7) for col in range(7, 13)]
METHODS = ("supervised", "transfer", "umm", "umm_memory", "seeded_umm")


@dataclass
class Participant:
    name: str
    x: np.ndarray        # letters x flashes x features
    codes: np.ndarray    # letters x flashes, 1-6 rows, 7-12 columns
    target: np.ndarray   # letters x 2: (row code, column code)


def load_participant(path: Path) -> Participant:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
    signal = sosfiltfilt(butter(4, [0.5, 16.0], btype="band", fs=FS, output="sos"), np.asarray(data.X, dtype=float), axis=0)
    labels, stimulus = np.asarray(data.y).astype(int), np.asarray(data.y_stim).astype(int)
    onsets = np.flatnonzero((stimulus[1:] != 0) & (stimulus[:-1] == 0)) + 1
    if len(onsets) != LETTERS * FLASHES:
        raise ValueError(f"{path.name}: expected {LETTERS * FLASHES} flashes, found {len(onsets)}")
    features = np.empty((len(onsets), N_BINS * signal.shape[1]))
    for i, onset in enumerate(onsets):
        epoch = signal[onset: onset + N_BINS * BIN] - signal[onset - BASELINE: onset].mean(axis=0)
        features[i] = epoch.reshape(N_BINS, BIN, -1).mean(axis=1).ravel()
    codes = stimulus[onsets].reshape(LETTERS, FLASHES)
    is_target = (labels[onsets] == 2).reshape(LETTERS, FLASHES)
    target = np.empty((LETTERS, 2), dtype=int)
    for letter in range(LETTERS):
        hit = np.unique(codes[letter][is_target[letter]])
        if len(hit) != 2 or not (hit[0] <= 6 < hit[1]):
            raise ValueError(f"{path.name} letter {letter}: targets {hit} are not one row and one column")
        target[letter] = hit
    return Participant(path.stem, features.reshape(LETTERS, FLASHES, -1), codes, target)


def shrunk_inverse(samples: np.ndarray) -> np.ndarray:
    """Inverse Ledoit-Wolf covariance of unlabelled epochs (the 'global' covariance UMM uses)."""
    return np.linalg.pinv(ledoit_wolf(samples)[0])


def class_means(x: np.ndarray, codes: np.ndarray, symbol: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    hit = (codes == symbol[0]) | (codes == symbol[1])
    return x[hit].mean(axis=0), x[~hit].mean(axis=0)


def lda_weights(xs: list[np.ndarray], codes: list[np.ndarray], targets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shrinkage LDA from labelled letters. Returns (weights, target mean, non-target mean)."""
    x = np.concatenate(xs)
    hit = np.concatenate([(c == t[0]) | (c == t[1]) for c, t in zip(codes, targets)])
    mu_t, mu_n = x[hit].mean(axis=0), x[~hit].mean(axis=0)
    centred = np.concatenate([x[hit] - mu_t, x[~hit] - mu_n])
    return shrunk_inverse(centred) @ (mu_t - mu_n), mu_t, mu_n


def decode_linear(x: np.ndarray, codes: np.ndarray, weights: np.ndarray) -> tuple[int, int]:
    scores = x @ weights
    by_code = np.array([scores[codes == code].mean() for code in range(1, CODES + 1)])
    return int(np.argmax(by_code[:6]) + 1), int(np.argmax(by_code[6:]) + 7)


def confidence(distances: np.ndarray) -> float:
    """UMM Eq. 5: gap between best and runner-up, in standard deviations of the non-selected symbols."""
    order = np.argsort(distances)
    rest = np.delete(distances, order[-1])
    spread = rest.std()
    return float((distances[order[-1]] - distances[order[-2]]) / spread) if spread > 0 else 0.0


def run_umm(p: Participant, n_flashes: int, *, memory: bool, seed: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Online UMM over one participant. Returns a boolean array: was each letter decoded correctly."""
    correct = np.zeros(LETTERS, dtype=bool)
    weight = SEED_WEIGHT if seed is not None else 0.0
    sum_t = seed[0] * weight if seed is not None else 0.0
    sum_n = seed[1] * weight if seed is not None else 0.0
    for letter in range(LETTERS):
        x, codes = p.x[letter, :n_flashes], p.codes[letter, :n_flashes]
        pool = p.x[: letter + 1, :n_flashes].reshape(-1, x.shape[1]) if memory else x
        inverse = shrunk_inverse(pool)
        distances = np.empty(len(SYMBOLS))
        hypotheses = []
        for k, symbol in enumerate(SYMBOLS):
            mu_t, mu_n = class_means(x, codes, symbol)
            hypotheses.append((mu_t, mu_n))
            if memory and weight > 0:
                # Eq. 7 with the current letter at weight 1: its confidence is unknown until it is decoded.
                mu_t, mu_n = (sum_t + mu_t) / (weight + 1), (sum_n + mu_n) / (weight + 1)
            delta = mu_t - mu_n
            distances[k] = delta @ inverse @ delta
        best = int(np.argmax(distances))
        correct[letter] = SYMBOLS[best] == tuple(p.target[letter])
        if memory:
            c = min(confidence(distances), 1.0)
            sum_t, sum_n, weight = sum_t + c * hypotheses[best][0], sum_n + c * hypotheses[best][1], weight + c
    return correct


def evaluate(participants: list[Participant], n_flashes: int) -> dict[str, np.ndarray]:
    """Per-method array of shape participants x letters (supervised: NaN for its calibration letters)."""
    out = {m: np.full((len(participants), LETTERS), np.nan) for m in METHODS}
    for i, p in enumerate(participants):
        others = [q for q in participants if q.name != p.name]
        w_transfer, seed_t, seed_n = _transfer(others, n_flashes)
        w_own, _, _ = lda_weights([p.x[l, :n_flashes] for l in range(CALIBRATION_LETTERS)],
                                  [p.codes[l, :n_flashes] for l in range(CALIBRATION_LETTERS)],
                                  [p.target[l] for l in range(CALIBRATION_LETTERS)])
        for letter in range(LETTERS):
            x, codes, truth = p.x[letter, :n_flashes], p.codes[letter, :n_flashes], tuple(p.target[letter])
            out["transfer"][i, letter] = decode_linear(x, codes, w_transfer) == truth
            if letter >= CALIBRATION_LETTERS:
                out["supervised"][i, letter] = decode_linear(x, codes, w_own) == truth
        out["umm"][i] = run_umm(p, n_flashes, memory=False)
        out["umm_memory"][i] = run_umm(p, n_flashes, memory=True)
        out["seeded_umm"][i] = run_umm(p, n_flashes, memory=True, seed=(seed_t, seed_n))
    return out


def _transfer(others: list[Participant], n_flashes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = [q.x[l, :n_flashes] for q in others for l in range(LETTERS)]
    codes = [q.codes[l, :n_flashes] for q in others for l in range(LETTERS)]
    targets = [q.target[l] for q in others for l in range(LETTERS)]
    return lda_weights(xs, codes, targets)


def bootstrap_mean(values: np.ndarray, seed: int = 0, resamples: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarise(correct: np.ndarray, letters: slice) -> dict:
    per_participant = np.nanmean(correct[:, letters], axis=1)
    low, high = bootstrap_mean(per_participant)
    return {"mean": round(float(per_participant.mean()), 4), "ci95": [round(low, 4), round(high, 4)],
            "min": round(float(per_participant.min()), 4), "per_participant": [round(float(v), 4) for v in per_participant]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=Path("data/nm000169/sourcedata"))
    parser.add_argument("--output", type=Path, default=Path("results/calibration_free_results.json"))
    args = parser.parse_args()
    participants = [load_participant(path) for path in sorted(args.source_dir.glob("A*.mat"))]
    if len(participants) != 8:
        raise SystemExit(f"expected 8 participants in {args.source_dir}, found {len(participants)}")

    blocks = {"all_35_letters": slice(0, LETTERS), "first_15_letters": slice(0, CALIBRATION_LETTERS),
              "last_20_letters": slice(CALIBRATION_LETTERS, LETTERS)}
    results: dict = {"participants": [p.name for p in participants], "by_sequences": {}}
    for sequences in SEQUENCES:
        correct = evaluate(participants, sequences * CODES)
        entry = {}
        for method in METHODS:
            entry[method] = {name: summarise(correct[method], block) for name, block in blocks.items()
                             if not (method == "supervised" and name != "last_20_letters")}
        if sequences == 10:
            late = blocks["last_20_letters"]
            pairs = {"seeded_umm_vs_umm_memory_all35": ("seeded_umm", "umm_memory", blocks["all_35_letters"]),
                     "seeded_umm_vs_transfer_all35": ("seeded_umm", "transfer", blocks["all_35_letters"]),
                     "seeded_umm_vs_supervised_last20": ("seeded_umm", "supervised", late)}
            entry["paired_wilcoxon"] = {}
            for name, (a, b, block) in pairs.items():
                da, db = np.nanmean(correct[a][:, block], axis=1), np.nanmean(correct[b][:, block], axis=1)
                p = float(wilcoxon(da, db).pvalue) if np.any(da != db) else 1.0
                entry["paired_wilcoxon"][name] = {"mean_difference": round(float((da - db).mean()), 4), "p": round(p, 4)}
        results["by_sequences"][str(sequences)] = entry
        row = " | ".join(f"{m} {entry[m].get('all_35_letters', entry[m]['last_20_letters'])['mean']:.3f}" for m in METHODS)
        print(f"{sequences:>2} sequences: {row}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
