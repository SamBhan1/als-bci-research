"""External validation: train on PhysioNet Study L, test on NEMAR ALS patients."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from advanced_experiment import aggregate, balanced_train_indices, bootstrap_ci, p300_window
from experiment import Dataset, load_edf_data


CHANNELS = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]


def alignment_transforms(x: np.ndarray, subjects: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-participant Euclidean-alignment whitening transforms."""
    transforms = {}
    for subject in np.unique(subjects):
        samples = x[subjects == subject]
        covariance = np.einsum("nct,ndt->ncd", samples, samples).mean(axis=0) / samples.shape[-1]
        covariance += np.eye(covariance.shape[0]) * (np.trace(covariance) / covariance.shape[0]) * 1e-5
        values, vectors = np.linalg.eigh(covariance)
        transforms[subject] = (vectors * (1.0 / np.sqrt(np.maximum(values, 1e-12)))) @ vectors.T
    return transforms


def apply_alignment(x: np.ndarray, subjects: np.ndarray, transforms: dict[str, np.ndarray]) -> np.ndarray:
    aligned = np.empty_like(x)
    for subject in np.unique(subjects):
        aligned[subjects == subject] = np.einsum("cd,ndt->nct", transforms[subject], x[subjects == subject])
    return aligned


def load_nemar(data_dir: Path) -> Dataset:
    import mne

    epochs, labels, subjects = [], [], []
    for events_path in sorted(data_dir.rglob("*_events.tsv")):
        bdf_path = events_path.with_name(events_path.name.replace("_events.tsv", "_eeg.bdf"))
        raw = mne.io.read_raw_bdf(bdf_path, preload=True, verbose="ERROR")
        raw.filter(0.5, 30.0, picks=CHANNELS, verbose="ERROR")
        eeg, _ = raw[CHANNELS, :]
        with events_path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            for row in rows:
                sample = int(row["sample"])
                left, right = sample - int(0.2 * raw.info["sfreq"]), sample + int(0.8 * raw.info["sfreq"])
                if left < 0 or right > eeg.shape[1]:
                    continue
                epoch = eeg[:, left:right].copy()
                epoch -= epoch[:, : int(0.2 * raw.info["sfreq"])].mean(axis=1, keepdims=True)
                epochs.append(epoch.astype(np.float32, copy=False))
                labels.append(int(row["trial_type"] == "Target"))
                subjects.append(events_path.parts[-4])
    return Dataset(np.asarray(epochs), np.asarray(labels), np.asarray(subjects))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--model", choices=("logistic", "riemann", "aligned_riemann", "ensemble"), default="riemann")
    parser.add_argument("--calibration-fraction", type=float, default=0.0, help="Labeled fraction of each test participant used for adaptation")
    args = parser.parse_args()

    train = load_edf_data(args.train_dir, channels=[f"EEG_{name}" for name in CHANNELS])
    test = load_nemar(args.test_dir)
    if not 0.0 <= args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must be in [0, 1)")
    if args.calibration_fraction:
        calibration_indices, evaluation_indices = [], []
        for subject in np.unique(test.subject):
            indices = np.flatnonzero(test.subject == subject)
            cut = max(args.repeats, int(len(indices) * args.calibration_fraction))
            calibration_indices.extend(indices[:cut])
            evaluation_indices.extend(indices[cut:])
        calibration = Dataset(test.x[calibration_indices], test.y[calibration_indices], test.subject[calibration_indices])
        test = Dataset(test.x[evaluation_indices], test.y[evaluation_indices], test.subject[evaluation_indices])
        train = aggregate(train, args.repeats)
        calibration = aggregate(calibration, args.repeats)
        train = Dataset(
            np.concatenate([train.x, calibration.x]),
            np.concatenate([train.y, calibration.y]),
            np.concatenate([train.subject, calibration.subject]),
        )
    else:
        train = aggregate(train, args.repeats)
    test = aggregate(test, args.repeats)
    keep = balanced_train_indices(train.y)
    if args.model == "aligned_riemann":
        transforms = alignment_transforms(train.x, train.subject)
        train_signal = apply_alignment(train.x, train.subject, transforms)
        test_signal = apply_alignment(test.x, test.subject, transforms)
    else:
        train_signal, test_signal = train.x, test.x
    x_train = p300_window(train_signal[keep])
    x_test = p300_window(test_signal)
    model = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs")
    if args.model in ("riemann", "aligned_riemann", "ensemble"):
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace

        covariances = XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas")
        train_cov = covariances.fit_transform(x_train, train.y[keep])
        test_cov = covariances.transform(x_test)
        tangent = TangentSpace(metric="riemann")
        x_train = tangent.fit_transform(train_cov)
        x_test = tangent.transform(test_cov)
    else:
        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)
    model.fit(x_train, train.y[keep])
    pred = model.predict(x_test)
    proba = model.predict_proba(x_test)[:, 1]
    if args.model == "ensemble":
        from mne.decoding import XdawnTransformer
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        xdawn = XdawnTransformer(n_components=4, reg="oas")
        lda_train = xdawn.fit_transform(p300_window(train_signal[keep]), train.y[keep]).reshape(len(keep), -1)
        lda_test = xdawn.transform(p300_window(test_signal)).reshape(len(test_signal), -1)
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(lda_train, train.y[keep])
        lda_probability = lda.predict_proba(lda_test)[:, 1]
        proba = 0.5 * (proba + lda_probability)
    pred = (proba >= 0.5).astype(int)
    print(f"Model: {args.model} | Calibration: {args.calibration_fraction:.0%} | Train epochs: {len(train.y)} | NEMAR test epochs: {len(test.y)} | participants: {len(np.unique(test.subject))}")
    print(f"External balanced accuracy: {balanced_accuracy_score(test.y, pred):.3f}")
    print(f"External macro F1: {f1_score(test.y, pred, average='macro'):.3f}")
    print(f"External ROC-AUC: {roc_auc_score(test.y, proba):.3f}")
    rows = []
    for subject in np.unique(test.subject):
        mask = test.subject == subject
        rows.append({
            "balanced_accuracy": balanced_accuracy_score(test.y[mask], pred[mask]),
            "macro_f1": f1_score(test.y[mask], pred[mask], average="macro"),
            "roc_auc": roc_auc_score(test.y[mask], proba[mask]),
        })
    print("Participant-level external statistics")
    for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
        values = np.asarray([row[metric] for row in rows])
        low, high = bootstrap_ci(values, seed=103)
        print(f"  {metric}: mean={values.mean():.3f}, std={values.std(ddof=1):.3f}, 95% CI=[{low:.3f}, {high:.3f}]")


if __name__ == "__main__":
    main()
