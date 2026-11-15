"""Small, dependency-light baseline for subject-independent EEG evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re

import numpy as np


ROOT = Path(__file__).resolve().parent


@dataclass
class Dataset:
    x: np.ndarray  # [trials, channels, time]
    y: np.ndarray  # [trials]
    subject: np.ndarray  # [trials]


def make_demo_data(seed: int = 7, subjects: int = 8, trials_per_subject: int = 80) -> Dataset:
    """Create EEG-like epochs with a shared class signal and subject-specific leakage."""
    rng = np.random.default_rng(seed)
    channels, time = 8, 128
    t = np.linspace(0, 1, time, endpoint=False)
    shared = np.sin(2 * np.pi * 12 * t) * np.hanning(time)
    xs, ys, ss = [], [], []

    for person in range(subjects):
        label = rng.integers(0, 2, size=trials_per_subject)
        subject_signature = rng.normal(0, 1, size=(channels, 1))
        for target in label:
            epoch = rng.normal(0, 0.8, size=(channels, time))
            epoch += subject_signature * 0.35  # tempting but non-generalizable shortcut
            if target == 1:
                epoch[0] += shared * 1.3
                epoch[1] += np.cos(2 * np.pi * 10 * t) * np.hanning(time) * 0.7
            xs.append(epoch)
            ys.append(target)
            ss.append(person)
    return Dataset(np.asarray(xs), np.asarray(ys), np.asarray(ss))


def load_edf_data(data_dir: Path, max_files: int | None = None, channels: list[str] | None = None) -> Dataset:
    """Load P300 target/non-target epochs from PhysioNet EDF+ recordings."""
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("Install MNE first: .venv\\Scripts\\python.exe -m pip install mne") from exc

    files = sorted(path for path in data_dir.rglob("*.edf") if path.stat().st_size > 0)
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No EDF files found under {data_dir}")

    epochs, labels, subjects = [], [], []
    for path in files:
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        eeg_names = channels or [name for name in raw.ch_names if name.startswith("EEG_")]
        eeg_names = [name for name in eeg_names if name in raw.ch_names]
        if not eeg_names or "StimulusCode" not in raw.ch_names or "StimulusType" not in raw.ch_names:
            continue
        raw.filter(0.5, 30.0, picks=eeg_names, verbose="ERROR")
        eeg, _ = raw[eeg_names, :]
        stimulus, _ = raw[["StimulusCode", "StimulusType"], :]
        code, kind = stimulus
        starts = np.flatnonzero((code[1:] != 0) & (code[:-1] == 0)) + 1
        window_start = int(-0.2 * raw.info["sfreq"])
        window_end = int(0.8 * raw.info["sfreq"])
        subject_match = re.search(r"(L_\d+)", path.name)
        subject = subject_match.group(1) if subject_match else path.stem.split("_")[0]
        for start in starts:
            left, right = start + window_start, start + window_end
            if left >= 0 and right <= eeg.shape[1]:
                epoch = eeg[:, left:right].copy()
                baseline_end = int(0.2 * raw.info["sfreq"])
                epoch -= epoch[:, :baseline_end].mean(axis=1, keepdims=True)
                epochs.append(epoch.astype(np.float32, copy=False))
                labels.append(int(kind[start] > 0.5))
                subjects.append(subject)
    if not epochs:
        raise RuntimeError("EDF files were found, but no valid stimulus epochs were extracted")
    return Dataset(np.asarray(epochs), np.asarray(labels), np.asarray(subjects))


def features(x: np.ndarray) -> np.ndarray:
    """Time-resolved P300 features after filtering and baseline correction."""
    bins = np.array_split(x, 10, axis=2)
    means = np.concatenate([part.mean(axis=2) for part in bins], axis=1)
    stds = np.concatenate([part.std(axis=2) for part in bins], axis=1)
    peaks = np.concatenate([part.max(axis=2) for part in bins], axis=1)
    return np.concatenate([means, stds, peaks], axis=1)


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 800, lr: float = 0.08):
    mean, scale = x.mean(0), x.std(0) + 1e-6
    z = (x - mean) / scale
    weights = np.zeros(z.shape[1])
    bias = 0.0
    class_weight = np.where(y == 1, len(y) / (2 * max(np.sum(y == 1), 1)), len(y) / (2 * max(np.sum(y == 0), 1)))
    for _ in range(steps):
        logits = z @ weights + bias
        probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        error = (probs - y) * class_weight
        weights -= lr * (z.T @ error) / len(y)
        bias -= lr * float(np.mean(error))
    return mean, scale, weights, bias


def predict(model, x: np.ndarray) -> np.ndarray:
    mean, scale, weights, bias = model
    logits = ((x - mean) / scale) @ weights + bias
    return (logits >= 0).astype(int)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    scores = []
    for cls in (0, 1):
        tp = np.sum((y == cls) & (pred == cls))
        actual = np.sum(y == cls)
        scores.append(float(tp / actual) if actual else 0.0)
    accuracy = float(np.mean(y == pred))
    balanced = float(np.mean(scores))
    f1s = []
    for cls in (0, 1):
        tp = np.sum((y == cls) & (pred == cls))
        fp = np.sum((y != cls) & (pred == cls))
        fn = np.sum((y == cls) & (pred != cls))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "balanced_accuracy": balanced, "macro_f1": float(np.mean(f1s))}


def random_trial_split(data: Dataset, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data.y))
    cut = int(len(idx) * 0.8)
    return idx[:cut], idx[cut:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, help="Directory containing PhysioNet EDF files")
    parser.add_argument("--max-files", type=int, default=None, help="Limit EDF files for a quick run")
    args = parser.parse_args()
    data = load_edf_data(args.data_dir, args.max_files) if args.data_dir else make_demo_data()
    x = features(data.x)

    train, test = random_trial_split(data)
    random_result = metrics(data.y[test], predict(fit_logistic(x[train], data.y[train]), x[test]))

    held_out = []
    for person in np.unique(data.subject):
        train = data.subject != person
        test = data.subject == person
        pred = predict(fit_logistic(x[train], data.y[train]), x[test])
        held_out.append(metrics(data.y[test], pred))

    print("ALS BCI baseline: real EDF EEG data" if args.data_dir else "ALS BCI baseline: synthetic EEG-like data")
    print(f"Trials: {len(data.y)} | Participants: {len(np.unique(data.subject))}")
    print("\nRandom trial split (potential subject leakage)")
    print("  " + ", ".join(f"{k}={v:.3f}" for k, v in random_result.items()))
    print("\nLeave-one-subject-out")
    for key in held_out[0]:
        values = [row[key] for row in held_out]
        print(f"  {key}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    print("\nInterpretation: use the participant-held-out result as the honest baseline.")


if __name__ == "__main__":
    main()

