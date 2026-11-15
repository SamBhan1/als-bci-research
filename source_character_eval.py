"""Evaluate exact row/column character decisions using NEMAR source metadata."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from advanced_experiment import aggregate, balanced_train_indices, bootstrap_ci, p300_window
from experiment import Dataset, load_edf_data


CHANNELS = ["Fz", "Cz", "Pz", "Oz", "P3", "P4", "PO7", "PO8"]


@dataclass
class SourceData:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    code: np.ndarray
    trial: np.ndarray


def load_source(data_dir: Path) -> SourceData:
    xs, ys, subjects, codes, trials = [], [], [], [], []
    for path in sorted(data_dir.glob("A*.mat")):
        source = loadmat(path, squeeze_me=True, struct_as_record=False)["data"]
        signal = np.asarray(source.X, dtype=np.float32)
        labels = np.asarray(source.y).astype(int)
        stimulus = np.asarray(source.y_stim).astype(int)
        starts = np.flatnonzero((stimulus[1:] != 0) & (stimulus[:-1] == 0)) + 1
        for start in starts:
            left, right = start - int(0.2 * 256), start + int(0.8 * 256)
            if left < 0 or right > len(signal):
                continue
            epoch = signal[left:right].T.copy()
            epoch -= epoch[:, : int(0.2 * 256)].mean(axis=1, keepdims=True)
            xs.append(epoch)
            ys.append(int(labels[start] == 2))
            subjects.append(path.stem)
            codes.append(int(stimulus[start]))
            trials.append(int(np.searchsorted(np.asarray(source.trial) - 1, start, side="right") - 1))
    return SourceData(np.asarray(xs), np.asarray(ys), np.asarray(subjects), np.asarray(codes), np.asarray(trials))


def average_by_code(source: SourceData, mask: np.ndarray) -> SourceData:
    """Average the repeated flashes for each participant/character/row-or-column code."""
    xs, ys, subjects, codes, trials = [], [], [], [], []
    selected = np.flatnonzero(mask)
    keys = {(source.subject[i], int(source.trial[i]), int(source.code[i])) for i in selected}
    for subject, trial, code in sorted(keys):
        indices = selected[
            (source.subject[selected] == subject)
            & (source.trial[selected] == trial)
            & (source.code[selected] == code)
        ]
        xs.append(source.x[indices].mean(axis=0))
        ys.append(int(source.y[indices].mean() > 0.5))
        subjects.append(subject)
        codes.append(code)
        trials.append(trial)
    return SourceData(np.asarray(xs), np.asarray(ys), np.asarray(subjects), np.asarray(codes), np.asarray(trials))


def fit_model(train: Dataset, model_name: str):
    keep = balanced_train_indices(train.y)
    if model_name == "riemann":
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace

        spatial = XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas")
        train_cov = spatial.fit_transform(p300_window(train.x[keep]), train.y[keep])
        tangent = TangentSpace(metric="riemann")
        train_features = tangent.fit_transform(train_cov)
        model = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs")
    elif model_name == "xdawn_lda":
        from mne.decoding import XdawnTransformer

        spatial = XdawnTransformer(n_components=4, reg="oas")
        train_features = spatial.fit_transform(p300_window(train.x[keep]), train.y[keep]).reshape(len(keep), -1)
        tangent = None
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        spatial = None
        tangent = None
        train_features = p300_window(train.x[keep]).reshape(len(keep), -1)
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model.fit(train_features, train.y[keep])
    return spatial, tangent, model, keep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=False)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.6)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--training-source", choices=("physionet", "nemar", "calibration"), default="physionet")
    parser.add_argument("--model", choices=("riemann", "xdawn_lda", "lda"), default="riemann")
    args = parser.parse_args()

    if args.training_source == "physionet" and args.train_dir is None:
        raise ValueError("--train-dir is required when --training-source=physionet")
    train = None if args.training_source in ("nemar", "calibration") else load_edf_data(args.train_dir, channels=[f"EEG_{name}" for name in CHANNELS])
    source = load_source(args.source_dir)
    all_scores, participant_scores = [], []
    for subject in np.unique(source.subject):
        subject_mask = source.subject == subject
        trials = np.unique(source.trial[subject_mask])
        calibration_trials = trials[: max(1, int(len(trials) * args.calibration_fraction))]
        evaluation_trials = trials[len(calibration_trials) :]
        calibration_mask = subject_mask & np.isin(source.trial, calibration_trials)
        evaluation_mask = subject_mask & np.isin(source.trial, evaluation_trials)
        calibration_source = average_by_code(source, calibration_mask)
        calibration = Dataset(calibration_source.x, calibration_source.y, calibration_source.subject)
        if args.training_source == "nemar":
            source_train_mask = source.subject != subject
            train_source = average_by_code(source, source_train_mask)
            train = Dataset(train_source.x, train_source.y, train_source.subject)
        if args.training_source == "calibration":
            train_augmented = calibration
        elif args.training_source == "nemar":
            train_augmented = Dataset(
                np.concatenate([train.x, calibration.x]),
                np.concatenate([train.y, calibration.y]),
                np.concatenate([train.subject, calibration.subject]),
            )
        else:
            train_agg = aggregate(train, args.repeats)
            train_augmented = Dataset(
                np.concatenate([train_agg.x, calibration.x]),
                np.concatenate([train_agg.y, calibration.y]),
                np.concatenate([train_agg.subject, calibration.subject]),
            )
        spatial, tangent, model, _ = fit_model(train_augmented, args.model)
        evaluation_source = average_by_code(source, evaluation_mask)
        test_window = p300_window(evaluation_source.x)
        if args.model == "riemann":
            test_features = tangent.transform(spatial.transform(test_window))
        elif args.model == "xdawn_lda":
            test_features = spatial.transform(test_window).reshape(len(test_window), -1)
        else:
            test_features = test_window.reshape(len(test_window), -1)
        probabilities = model.predict_proba(test_features)[:, 1]
        labels = evaluation_source.y
        codes = evaluation_source.code
        trial_ids = evaluation_source.trial
        subject_scores = []
        for trial in np.unique(trial_ids):
            mask = trial_ids == trial
            unique_codes = np.unique(codes[mask])
            target_rate = np.asarray([labels[mask & (codes == code)].mean() for code in unique_codes])
            true_targets = set(unique_codes[target_rate > 0.5])
            if len(true_targets) != 2:
                continue
            score_by_code = [probabilities[mask & (codes == code)].mean() for code in unique_codes]
            score_by_code = np.asarray(score_by_code)
            rows = unique_codes <= 6
            columns = unique_codes >= 7
            if not rows.any() or not columns.any():
                continue
            predicted_targets = {int(unique_codes[rows][np.argmax(score_by_code[rows])]), int(unique_codes[columns][np.argmax(score_by_code[columns])])}
            subject_scores.append(float(predicted_targets == true_targets))
        all_scores.extend(subject_scores)
        participant_scores.append(float(np.mean(subject_scores)))

    all_scores = np.asarray(all_scores)
    participant_scores = np.asarray(participant_scores)
    low, high = bootstrap_ci(all_scores, seed=503)
    participant_low, participant_high = bootstrap_ci(participant_scores, seed=509)
    print(f"Calibration: {args.calibration_fraction:.0%} | Characters: {len(all_scores)} | participants: {len(participant_scores)}")
    print(f"Exact row/column-pair accuracy: {all_scores.mean():.3f} | 95% CI over characters: [{low:.3f}, {high:.3f}]")
    print(f"Participant-mean accuracy: {participant_scores.mean():.3f} | 95% CI over participants: [{participant_low:.3f}, {participant_high:.3f}]")


if __name__ == "__main__":
    main()
