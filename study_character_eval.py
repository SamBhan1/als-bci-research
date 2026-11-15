"""Valid sequence-level P300 decoding using PhysioNet StimulusCode metadata."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from advanced_experiment import aggregate, balanced_train_indices, bootstrap_ci, p300_window
from experiment import Dataset


@dataclass
class SequenceData:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    code: np.ndarray
    sequence: np.ndarray


def load_sequences(data_dir: Path) -> SequenceData:
    import mne

    xs, ys, subjects, codes, sequences = [], [], [], [], []
    for path in sorted(path for path in data_dir.rglob("*.edf") if path.stat().st_size > 0):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        eeg_names = [name for name in raw.ch_names if name.startswith("EEG_")]
        if not eeg_names or not {"StimulusCode", "StimulusType", "PhaseInSequence"}.issubset(raw.ch_names):
            continue
        raw.filter(0.5, 30.0, picks=eeg_names, verbose="ERROR")
        eeg, _ = raw[eeg_names, :]
        marker, _ = raw[["StimulusCode", "StimulusType", "PhaseInSequence"], :]
        stim_code, stim_type, phase = marker
        starts = np.flatnonzero((stim_code[1:] != 0) & (stim_code[:-1] == 0)) + 1
        sequence_starts = np.flatnonzero((phase[1:] == 2) & (phase[:-1] != 2)) + 1
        sequence_ends = list(sequence_starts[1:]) + [len(phase)]
        subject_match = re.search(r"(L_\d+)", path.name)
        subject = subject_match.group(1) if subject_match else path.stem
        for sequence_number, (left_bound, right_bound) in enumerate(zip(sequence_starts, sequence_ends)):
            events = starts[(starts >= left_bound) & (starts < right_bound)]
            sequence_id = f"{subject}:{path.stem}:{sequence_number}"
            for start in events:
                left, right = start - int(0.2 * raw.info["sfreq"]), start + int(0.8 * raw.info["sfreq"])
                if left < 0 or right > eeg.shape[1]:
                    continue
                epoch = eeg[:, left:right].copy()
                epoch -= epoch[:, : int(0.2 * raw.info["sfreq"])].mean(axis=1, keepdims=True)
                xs.append(epoch.astype(np.float32, copy=False))
                ys.append(int(stim_type[start] > 0.5))
                subjects.append(subject)
                codes.append(int(stim_code[start]))
                sequences.append(sequence_id)
    return SequenceData(np.asarray(xs), np.asarray(ys), np.asarray(subjects), np.asarray(codes), np.asarray(sequences))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    data = load_sequences(args.data_dir)
    scores, subjects = [], []
    for subject in np.unique(data.subject):
        train_mask = data.subject != subject
        test_mask = ~train_mask
        train = Dataset(data.x[train_mask], data.y[train_mask], data.subject[train_mask])
        train = aggregate(train, args.repeats)
        keep = balanced_train_indices(train.y)
        from pyriemann.estimation import XdawnCovariances
        from pyriemann.tangentspace import TangentSpace

        spatial = XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas")
        train_cov = spatial.fit_transform(p300_window(train.x[keep]), train.y[keep])
        tangent = TangentSpace(metric="riemann")
        train_features = tangent.fit_transform(train_cov)
        model = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs")
        model.fit(train_features, train.y[keep])
        test_probabilities = model.predict_proba(tangent.transform(spatial.transform(p300_window(data.x[test_mask]))))[:, 1]
        test_codes = data.code[test_mask]
        test_labels = data.y[test_mask]
        test_sequences = data.sequence[test_mask]
        for sequence in np.unique(test_sequences):
            mask = test_sequences == sequence
            unique_codes = np.unique(test_codes[mask])
            true_scores = np.asarray([test_labels[mask & (test_codes == code)].mean() for code in unique_codes])
            true_targets = set(unique_codes[true_scores > 0.5])
            if len(true_targets) != 2:
                continue
            predicted_order = np.argsort([test_probabilities[mask & (test_codes == code)].mean() for code in unique_codes])[-2:]
            predicted_targets = set(unique_codes[predicted_order])
            scores.append(float(predicted_targets == true_targets))
            subjects.append(subject)
    scores = np.asarray(scores)
    print(f"Sequences: {len(scores)} | participants: {len(np.unique(subjects))} | epoch averaging: {args.repeats}")
    print(f"Exact target-pair sequence accuracy: {scores.mean():.3f}")
    low, high = bootstrap_ci(scores, seed=301)
    print(f"95% bootstrap CI over sequences: [{low:.3f}, {high:.3f}]")
    per_subject = np.asarray([scores[np.asarray(subjects) == subject].mean() for subject in np.unique(subjects)])
    low, high = bootstrap_ci(per_subject, seed=307)
    print(f"Participant-mean sequence accuracy: {per_subject.mean():.3f}")
    print(f"95% bootstrap CI over participants: [{low:.3f}, {high:.3f}]")


if __name__ == "__main__":
    main()
