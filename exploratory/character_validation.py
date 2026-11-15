"""Exploratory character diagnostic; NEMAR lacks stimulus identity metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from advanced_experiment import aggregate, balanced_train_indices, bootstrap_ci, p300_window
from external_validation import CHANNELS, load_nemar
from experiment import Dataset, load_edf_data


def main() -> None:
    raise RuntimeError(
        "NEMAR events.tsv has target labels but no stimulus identity/order. "
        "Character accuracy cannot be validly reconstructed from this derivative."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.6)
    parser.add_argument("--stimuli-per-character", type=int, default=12)
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()

    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must be between 0 and 1")
    train = load_edf_data(args.train_dir, channels=[f"EEG_{name}" for name in CHANNELS])
    external = load_nemar(args.test_dir)
    calibration_indices, evaluation_indices = [], []
    events_per_character = args.stimuli_per_character * args.repetitions
    for subject in np.unique(external.subject):
        indices = np.flatnonzero(external.subject == subject)
        n_characters = len(indices) // events_per_character
        calibration_characters = int(n_characters * args.calibration_fraction)
        boundary = calibration_characters * events_per_character
        calibration_indices.extend(indices[:boundary])
        evaluation_indices.extend(indices[boundary : n_characters * events_per_character])
    calibration = Dataset(external.x[calibration_indices], external.y[calibration_indices], external.subject[calibration_indices])
    evaluation = Dataset(external.x[evaluation_indices], external.y[evaluation_indices], external.subject[evaluation_indices])
    train = aggregate(train, args.repetitions)
    calibration = aggregate(calibration, args.repetitions)
    train = Dataset(
        np.concatenate([train.x, calibration.x]),
        np.concatenate([train.y, calibration.y]),
        np.concatenate([train.subject, calibration.subject]),
    )
    keep = balanced_train_indices(train.y)

    from pyriemann.estimation import XdawnCovariances
    from pyriemann.tangentspace import TangentSpace

    x_train = p300_window(train.x[keep])
    spatial = XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas")
    train_cov = spatial.fit_transform(x_train, train.y[keep])
    tangent = TangentSpace(metric="riemann")
    train_features = tangent.fit_transform(train_cov)
    model = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs")
    model.fit(train_features, train.y[keep])

    test_cov = spatial.transform(p300_window(evaluation.x))
    probabilities = model.predict_proba(tangent.transform(test_cov))[:, 1]
    character_scores, character_subjects = [], []
    for subject in np.unique(evaluation.subject):
        mask = evaluation.subject == subject
        y = evaluation.y[mask]
        p = probabilities[mask]
        n_characters = len(y) // events_per_character
        y = y[: n_characters * events_per_character].reshape(n_characters, args.repetitions, args.stimuli_per_character)
        p = p[: n_characters * events_per_character].reshape(n_characters, args.repetitions, args.stimuli_per_character)
        true_targets = y.mean(axis=1).argsort(axis=1)[:, -2:]
        predicted_targets = p.mean(axis=1).argsort(axis=1)[:, -2:]
        character_scores.extend(np.all(np.sort(true_targets) == np.sort(predicted_targets), axis=1))
        character_subjects.extend([subject] * n_characters)

    scores = np.asarray(character_scores, dtype=float)
    print(f"Calibration: {args.calibration_fraction:.0%} | Evaluation characters: {len(scores)} | participants: {len(np.unique(character_subjects))}")
    print(f"Exact row/column-pair character accuracy: {scores.mean():.3f}")
    low, high = bootstrap_ci(scores, seed=211)
    print(f"95% bootstrap CI over characters: [{low:.3f}, {high:.3f}]")
    participant_scores = np.asarray([scores[np.asarray(character_subjects) == subject].mean() for subject in np.unique(character_subjects)])
    low, high = bootstrap_ci(participant_scores, seed=223)
    print(f"Participant-mean character accuracy: {participant_scores.mean():.3f}")
    print(f"95% bootstrap CI over participants: [{low:.3f}, {high:.3f}]")


if __name__ == "__main__":
    main()
