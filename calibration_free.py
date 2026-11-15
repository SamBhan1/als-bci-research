"""Calibration-free P300 spelling for people with ALS: does seeding UMM from other people fix its failure?

Three datasets, all 6x6 spellers at 256 Hz, restricted to the same eight channels:

  rome_als      BNCI 2014-008      8 people with ALS    row/column     35 letters each
  rome_healthy  BNCI 2014-009      10 healthy people    row/column     18 letters each
  etsu_als      bigP3BCI Study L   11 people with ALS   checkerboard   30 letters each (different site/hardware)

Methods (see PROTOCOL.md, both addenda):

  supervised            shrinkage LDA trained on the participant's own first labelled letters (needs calibration)
  transfer              shrinkage LDA trained on the other participants of the same dataset, never adapted
  umm                   UMM, current letter only (Sosulski & Tangermann 2023; reimplemented from the paper)
  umm_memory            UMM with pooled covariance and confidence-weighted means from earlier letters
  seeded_umm            umm_memory whose mean memory starts from the other participants of the same dataset
  seeded_umm[<other>]   the same, seeded from every participant of a different dataset

Calibration-free methods run as an online simulation: letter i is decoded from letters 1..i only and no label
of the test participant is ever used.

    python calibration_free.py
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.covariance import ledoit_wolf

FS = 256
CHANNELS = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]
N_SYMBOLS = 36
RESULT_SCHEMA_VERSION = 1
SEED_WEIGHT = 5.0                       # other people count as much as five confident letters (fixed in advance)
CALIBRATION = {"rome_als": 15, "rome_healthy": 6, "etsu_als": 12}
SEQUENCES = (1, 2, 3, 5, 7, 10)         # row/column sweep on rome_als (12 flashes per sequence)
PIPELINES = ("binned", "authors")
ALS_DATASETS = ("rome_als", "etsu_als")


@dataclass
class Participant:
    name: str
    x: list[np.ndarray]        # per letter: flashes x features
    lit: list[np.ndarray]      # per letter: flashes x 36 booleans, True where that matrix cell flashed
    target: np.ndarray         # per letter: index 0-35 of the cell the person was attending to

    def first(self, n_flashes: int | None) -> "Participant":
        if n_flashes is None:
            return self
        if n_flashes <= 0:
            raise ValueError("n_flashes must be positive or None")
        return Participant(self.name, [x[:n_flashes] for x in self.x], [m[:n_flashes] for m in self.lit], self.target)


def validate_participant(p: Participant) -> None:
    """Fail early on malformed participant data instead of producing plausible-looking scores."""
    if not p.name:
        raise ValueError("participant name is empty")
    if not (len(p.x) == len(p.lit) == len(p.target)) or not p.x:
        raise ValueError(f"{p.name}: x, lit and target must have the same non-zero number of letters")
    for letter, (x, lit, target) in enumerate(zip(p.x, p.lit, p.target)):
        if x.ndim != 2 or lit.ndim != 2:
            raise ValueError(f"{p.name} letter {letter}: x and lit must be matrices")
        if x.shape[0] != lit.shape[0] or lit.shape[1] != N_SYMBOLS or x.shape[0] == 0:
            raise ValueError(f"{p.name} letter {letter}: flashes and symbol dimensions do not agree")
        if not np.isfinite(x).all():
            raise ValueError(f"{p.name} letter {letter}: non-finite EEG feature")
        if not 0 <= int(target) < N_SYMBOLS or not lit[:, int(target)].any():
            raise ValueError(f"{p.name} letter {letter}: invalid or unobserved target symbol")


# ---------------------------------------------------------------- features

def featurise(signal: np.ndarray, onsets: np.ndarray, pipeline: str) -> np.ndarray:
    """signal: samples x 8 channels at 256 Hz. Returns flashes x features."""
    signal = np.asarray(signal, dtype=float)
    onsets = np.asarray(onsets, dtype=int).reshape(-1)
    if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
        raise ValueError("signal must be a non-empty samples x channels matrix")
    if pipeline == "binned":      # fixed in the first protocol addendum
        clean = sosfiltfilt(butter(4, [0.5, 16.0], btype="band", fs=FS, output="sos"), signal, axis=0)
        width, bins, base = int(0.05 * FS), 16, int(0.1 * FS)
        if np.any(onsets < base) or np.any(onsets + bins * width > len(signal)):
            raise ValueError("binned epoch would extend outside the recording")
        out = np.empty((len(onsets), bins * signal.shape[1]))
        for i, onset in enumerate(onsets):
            epoch = clean[onset: onset + bins * width] - clean[onset - base: onset].mean(axis=0)
            out[i] = epoch.reshape(bins, width, -1).mean(axis=1).ravel()
        return out
    if pipeline == "authors":     # settings of the UMM authors' public demo: 0.5-8 Hz, 40 Hz, 0.05-0.7 s, no baseline
        clean = sosfiltfilt(butter(4, [0.5, 8.0], btype="band", fs=FS, output="sos"), signal, axis=0)
        slow = resample_poly(clean, 5, 32, axis=0)                     # 256 Hz -> 40 Hz
        starts = np.round(onsets * 40 / FS).astype(int) + 2            # 0.05 s
        if np.any(starts < 0) or np.any(starts + 27 > len(slow)):
            raise ValueError("authors' epoch would extend outside the resampled recording")
        return np.stack([slow[s: s + 27].ravel() for s in starts])    # up to 0.70 s inclusive
    raise ValueError(pipeline)


def rowcol_lit(codes: np.ndarray) -> np.ndarray:
    """Row/column codes (1-6 rows, 7-12 columns) -> flashes x 36 cell membership."""
    codes = np.asarray(codes, dtype=int).reshape(-1)
    if np.any((codes < 1) | (codes > 12)):
        raise ValueError("row/column stimulus codes must be in 1..12")
    lit = np.zeros((len(codes), N_SYMBOLS), dtype=bool)
    for i, code in enumerate(codes):
        if code <= 6:
            lit[i, (code - 1) * 6: code * 6] = True
        else:
            lit[i, code - 7:: 6] = True
    return lit


def _usable(onsets: np.ndarray, n_samples: int) -> np.ndarray:
    return (onsets >= int(0.1 * FS)) & (onsets + FS <= n_samples)


# ---------------------------------------------------------------- loaders

def _load_rome_session(session, pipeline: str) -> tuple[list, list, list]:
    names = [str(c) for c in session.channels]
    signal = np.asarray(session.X, dtype=float)[:, [names.index(c) for c in CHANNELS]]
    labels, stimulus = np.asarray(session.y).astype(int), np.asarray(session.y_stim).astype(int)
    onsets = np.flatnonzero((stimulus[1:] != 0) & (stimulus[:-1] == 0)) + 1
    starts = np.asarray(session.trial).astype(int) - 1                 # MATLAB indices are 1-based
    letter_of = np.searchsorted(starts, onsets, side="right") - 1
    if np.any(letter_of < 0):
        raise ValueError("stimulus onset precedes the first trial boundary")
    features = featurise(signal, onsets, pipeline)
    xs, lits, targets = [], [], []
    for letter in range(len(starts)):
        pick = letter_of == letter
        if not pick.any():
            raise ValueError(f"letter {letter}: no stimulus flashes found")
        lit = rowcol_lit(stimulus[onsets[pick]])
        is_target = labels[onsets[pick]] == 2
        cell = np.flatnonzero(lit[is_target].all(axis=0))             # the one cell in every target flash
        if len(cell) != 1 or not (lit[:, cell[0]] == is_target).all():
            raise ValueError(f"letter {letter}: target flashes do not identify one cell")
        xs.append(features[pick]); lits.append(lit); targets.append(int(cell[0]))
    return xs, lits, targets


def load_rome(path: Path, pipeline: str) -> Participant:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
    xs, lits, targets = [], [], []
    for session in np.atleast_1d(data):
        x, lit, target = _load_rome_session(session, pipeline)
        xs += x; lits += lit; targets += target
    participant = Participant(path.stem, xs, lits, np.array(targets))
    validate_participant(participant)
    return participant


def load_etsu(paths: list[Path], pipeline: str) -> Participant:
    import mne
    mne.set_log_level("ERROR")
    xs, lits, targets = [], [], []
    for path in sorted(paths):
        raw = mne.io.read_raw_edf(path, preload=True)
        names = raw.ch_names
        signal = raw.get_data(picks=[f"EEG_{c}" for c in CHANNELS]).T * 1e6
        first_cell = names.index("A_1_1")
        cells = raw.get_data(picks=names[first_cell: first_cell + N_SYMBOLS]) > 0
        begin, kind, current = raw.get_data(picks=["StimulusBegin", "StimulusType", "CurrentTarget"])
        onsets = np.flatnonzero((begin[1:] > 0) & (begin[:-1] == 0)) + 1
        onsets = onsets[_usable(onsets, len(signal))]
        features = featurise(signal, onsets, pipeline)
        breaks = np.flatnonzero(np.diff(onsets) > 200) + 1            # long gap = next letter
        for a, b in zip(np.r_[0, breaks], np.r_[breaks, len(onsets)]):
            o = onsets[a:b]
            target = int(current[o[0]]) - 1
            lit = cells[:, o].T
            if target < 0 or not (lit[:, target] == (kind[o] > 0)).all():
                raise ValueError(f"{path.name}: CurrentTarget disagrees with StimulusType")
            xs.append(features[a:b]); lits.append(lit); targets.append(target)
    participant = Participant(paths[0].name[:4], xs, lits, np.array(targets))
    validate_participant(participant)
    return participant


def load_datasets(data_dir: Path, pipeline: str) -> dict[str, list[Participant]]:
    etsu: dict[str, list[Path]] = {}
    for path in sorted((data_dir / "raw").glob("L_*_SE001_CB_Train*.edf")):
        etsu.setdefault(path.name[:4], []).append(path)
    out = {
        "rome_als": [load_rome(p, pipeline) for p in sorted((data_dir / "nm000169" / "sourcedata").glob("A*.mat"))],
        "rome_healthy": [load_rome(p, pipeline) for p in sorted((data_dir / "bnci009").glob("A*S.mat"))],
        "etsu_als": [load_etsu(paths, pipeline) for paths in etsu.values()],
    }
    expected = {"rome_als": 8, "rome_healthy": 10, "etsu_als": 11}
    for name, people in out.items():
        if len(people) != expected[name]:
            raise SystemExit(f"{name}: expected {expected[name]} participants under {data_dir}, found {len(people)}. See README.")
        if len({p.name for p in people}) != len(people):
            raise ValueError(f"{name}: participant identifiers are not unique")
        for participant in people:
            validate_participant(participant)
    return out


# ---------------------------------------------------------------- decoders

def shrunk_inverse(samples: np.ndarray) -> np.ndarray:
    """Inverse Ledoit-Wolf covariance of unlabelled epochs (the 'global' covariance UMM uses)."""
    return np.linalg.pinv(ledoit_wolf(samples)[0])


def hypothesis_means(x: np.ndarray, lit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each of the 36 symbols: mean of the flashes that lit it, and mean of the rest. Shapes 36 x features."""
    weights = lit.T.astype(float)
    n_lit = weights.sum(axis=1, keepdims=True)
    lit_sum = weights @ x
    return lit_sum / np.maximum(n_lit, 1), (x.sum(axis=0) - lit_sum) / np.maximum(len(x) - n_lit, 1)


def labelled_means(people: list[Participant]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target mean, non-target mean, and class-centred epochs pooled over labelled letters of `people`."""
    target, other = [], []
    for p in people:
        for x, lit, cell in zip(p.x, p.lit, p.target):
            target.append(x[lit[:, cell]]); other.append(x[~lit[:, cell]])
    target, other = np.concatenate(target), np.concatenate(other)
    mu_t, mu_n = target.mean(axis=0), other.mean(axis=0)
    return mu_t, mu_n, np.concatenate([target - mu_t, other - mu_n])


def lda_weights(people: list[Participant]) -> np.ndarray:
    mu_t, mu_n, centred = labelled_means(people)
    return shrunk_inverse(centred) @ (mu_t - mu_n)


def decode_linear(x: np.ndarray, lit: np.ndarray, weights: np.ndarray) -> int:
    mu_t, mu_n = hypothesis_means((x @ weights)[:, None], lit)
    return int(np.argmax(mu_t - mu_n))


def confidence(distances: np.ndarray) -> float:
    """UMM Eq. 5: gap between best and runner-up, in standard deviations of the non-selected symbols."""
    order = np.argsort(distances)
    spread = np.delete(distances, order[-1]).std()
    return float((distances[order[-1]] - distances[order[-2]]) / spread) if spread > 0 else 0.0


def seed_from(people: list[Participant]) -> tuple[np.ndarray, np.ndarray]:
    """Average target and non-target response of other people, each person first divided by their own overall
    standard deviation. Recordings come in different units (the healthy Rome files are about seven times
    larger than the ALS ones), so the seed is stored unit-free and rescaled to the new user in `run_umm`."""
    target, other = [], []
    for p in people:
        scale = np.concatenate(p.x).std()
        for x, lit, cell in zip(p.x, p.lit, p.target):
            target.append(x[lit[:, cell]] / scale); other.append(x[~lit[:, cell]] / scale)
    return np.concatenate(target).mean(axis=0), np.concatenate(other).mean(axis=0)


def run_umm(p: Participant, *, memory: bool, seed: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Online UMM over one participant. Returns booleans: was each letter decoded correctly.

    `seed` comes from `seed_from`. It is rescaled with the standard deviation of the participant's own epochs
    seen so far, which needs no labels and no future data.
    """
    correct = np.zeros(len(p.x), dtype=bool)
    seed_weight = SEED_WEIGHT if seed is not None else 0.0
    sum_t, sum_n, weight = 0.0, 0.0, 0.0
    seen_chunks = []
    for letter, (x, lit) in enumerate(zip(p.x, p.lit)):
        seen_chunks.append(x)
        seen = np.concatenate(seen_chunks) if memory else x
        inverse = shrunk_inverse(seen if memory else x)
        mu_t, mu_n = hypothesis_means(x, lit)
        if memory and weight + seed_weight > 0:
            # Eq. 7 with the current letter at weight 1: its confidence is unknown until it is decoded.
            prior_t = seed_weight * seen.std() * seed[0] if seed is not None else 0.0
            prior_n = seed_weight * seen.std() * seed[1] if seed is not None else 0.0
            total = seed_weight + weight + 1
            delta = (prior_t + sum_t + mu_t) / total - (prior_n + sum_n + mu_n) / total
        else:
            delta = mu_t - mu_n
        distances = np.einsum("sf,fg,sg->s", delta, inverse, delta)
        best = int(np.argmax(distances))
        correct[letter] = best == p.target[letter]
        if memory:
            c = min(confidence(distances), 1.0)
            sum_t, sum_n, weight = sum_t + c * mu_t[best], sum_n + c * mu_n[best], weight + c
    return correct


def evaluate(people: list[Participant], calibration: int, external: dict[str, list[Participant]] | None = None,
             n_flashes: int | None = None) -> dict[str, list[np.ndarray]]:
    """Per method: one array of per-letter correctness per participant (supervised: NaN on its calibration letters)."""
    people = [p.first(n_flashes) for p in people]
    external_seeds = {}
    for name, others in (external or {}).items():
        external_seeds[f"seeded_umm[{name}]"] = seed_from([q.first(n_flashes) for q in others])
    out: dict[str, list[np.ndarray]] = {m: [] for m in ["supervised", "transfer", "umm", "umm_memory", "seeded_umm", *external_seeds]}
    for p in people:
        others = [q for q in people if q.name != p.name]
        mu_t, mu_n, centred = labelled_means(others)
        w_transfer = shrunk_inverse(centred) @ (mu_t - mu_n)
        own = Participant(p.name, p.x[:calibration], p.lit[:calibration], p.target[:calibration])
        w_own = lda_weights([own])
        letters = range(len(p.x))
        out["transfer"].append(np.array([decode_linear(p.x[l], p.lit[l], w_transfer) == p.target[l] for l in letters], dtype=float))
        out["supervised"].append(np.array([decode_linear(p.x[l], p.lit[l], w_own) == p.target[l] if l >= calibration else np.nan
                                           for l in letters], dtype=float))
        out["umm"].append(run_umm(p, memory=False).astype(float))
        out["umm_memory"].append(run_umm(p, memory=True).astype(float))
        out["seeded_umm"].append(run_umm(p, memory=True, seed=seed_from(others)).astype(float))
        for name, seed in external_seeds.items():
            out[name].append(run_umm(p, memory=True, seed=seed).astype(float))
    return out


# ---------------------------------------------------------------- statistics

def per_participant(correct: list[np.ndarray], block: str, calibration: int) -> np.ndarray:
    cut = {"all": slice(None), "early": slice(0, calibration), "late": slice(calibration, None)}[block]
    return np.array([np.nanmean(c[cut]) if np.isfinite(c[cut]).any() else np.nan for c in correct])


def bootstrap_mean(values: np.ndarray, seed: int = 0, resamples: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a non-empty finite vector")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sign_flip_p(differences: np.ndarray) -> float:
    """Exact two-sided sign-flip permutation test on paired participant differences (all 2^n assignments)."""
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not np.isfinite(differences).all():
        raise ValueError("sign-flip differences must be a finite vector")
    d = differences[differences != 0]
    if len(d) == 0:
        return 1.0
    observed = abs(d.sum())
    bits = (np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1
    return float((np.abs((2.0 * bits - 1.0) @ d) >= observed - 1e-12).mean())


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("summary values must be a non-empty finite vector")
    low, high = bootstrap_mean(values)
    return {"mean": round(float(values.mean()), 4), "ci95": [round(low, 4), round(high, 4)],
            "min": round(float(values.min()), 4), "n": int(len(values)),
            "per_participant": [round(float(v), 4) for v in values]}


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired inputs must be same-shaped finite vectors")
    d = a - b
    low, high = bootstrap_mean(d)
    return {"mean_difference": round(float(d.mean()), 4), "ci95": [round(low, 4), round(high, 4)], "n": int(len(d)),
            "better_same_worse": [int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())],
            "p_sign_flip": round(sign_flip_p(d), 5),
            "p_wilcoxon": round(float(wilcoxon(a, b).pvalue), 5) if np.any(d != 0) else 1.0}


def external_for(name: str, datasets: dict[str, list[Participant]]) -> dict[str, list[Participant]]:
    return {other: people for other, people in datasets.items() if other != name}


def analyse(datasets: dict[str, list[Participant]], *, sweep: bool) -> dict:
    result: dict = {"schema_version": RESULT_SCHEMA_VERSION, "chance_accuracy": 1 / N_SYMBOLS, "datasets": {}}
    scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for name, people in datasets.items():
        calibration = CALIBRATION[name]
        correct = evaluate(people, calibration, external_for(name, datasets))
        scores[name] = {m: {b: per_participant(c, b, calibration) for b in ("all", "early", "late")} for m, c in correct.items()}
        entry = {"participants": [p.name for p in people], "calibration_letters": calibration,
                 "letters": [len(p.x) for p in people], "methods": {}}
        for method, blocks in scores[name].items():
            entry["methods"][method] = {b: summary(v) for b, v in blocks.items() if not (method == "supervised" and b != "late")}
        s = scores[name]
        entry["comparisons"] = {
            "seeded_umm - umm_memory (all letters)": paired(s["seeded_umm"]["all"], s["umm_memory"]["all"]),
            "seeded_umm - transfer (all letters)": paired(s["seeded_umm"]["all"], s["transfer"]["all"]),
            "seeded_umm - supervised (after calibration letters)": paired(s["seeded_umm"]["late"], s["supervised"]["late"]),
        }
        for method in s:
            if method.startswith("seeded_umm["):
                entry["comparisons"][f"{method} - umm_memory (all letters)"] = paired(s[method]["all"], s["umm_memory"]["all"])
                entry["comparisons"][f"{method} - seeded_umm (all letters)"] = paired(s[method]["all"], s["seeded_umm"]["all"])
        if sweep and name == "rome_als":
            entry["by_sequences"] = {}
            for sequences in SEQUENCES:
                part = evaluate(people, calibration, None, n_flashes=sequences * 12)
                entry["by_sequences"][str(sequences)] = {
                    m: round(float(per_participant(c, "late", calibration).mean()), 4) for m, c in part.items()}
        result["datasets"][name] = entry
        print(f"  {name}: " + " | ".join(f"{m} {scores[name][m]['all' if m != 'supervised' else 'late'].mean():.3f}" for m in scores[name]))
    pool = lambda method: np.concatenate([scores[d][method]["all"] for d in ALS_DATASETS])
    late = lambda method: np.concatenate([scores[d][method]["late"] for d in ALS_DATASETS])
    result["all_als_participants"] = {
        "seeded_umm": summary(pool("seeded_umm")), "umm_memory": summary(pool("umm_memory")), "transfer": summary(pool("transfer")),
        "seeded_umm - umm_memory (all letters)": paired(pool("seeded_umm"), pool("umm_memory")),
        "seeded_umm - transfer (all letters)": paired(pool("seeded_umm"), pool("transfer")),
        "seeded_umm - supervised (after calibration letters)": paired(late("seeded_umm"), late("supervised")),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "calibration_free_results.json")
    args = parser.parse_args()
    results = {"schema_version": RESULT_SCHEMA_VERSION, "seed_weight": SEED_WEIGHT, "channels": CHANNELS,
               "n_symbols": N_SYMBOLS, "chance_accuracy": 1 / N_SYMBOLS, "calibration_letters": CALIBRATION,
               "pipelines": {}}
    for pipeline in PIPELINES:
        print(f"pipeline: {pipeline}")
        results["pipelines"][pipeline] = analyse(load_datasets(args.data_dir, pipeline), sweep=pipeline == "binned")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
