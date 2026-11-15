"""Exploratory extensions for cross-site adaptation and confidence-based stopping.

This module deliberately sits beside ``calibration_free.py``.  It does not change the
primary result.  It asks two next-step questions using only unlabeled data from the
new participant:

1. Can a seed recorded on another EEG setup be aligned to the new setup?
2. Can the speller stop after a confident sequence instead of always collecting the
   maximum number of flashes?

The alignment is a CORAL-style covariance transport.  Source participants are
feature-standardised using their own recordings; the seed is then transported to the
new participant's covariance geometry using only flashes observed so far.  The
stopping routine is an online replay: it never sees a future flash or the target label
while deciding when to stop.  Target labels are used only after the decision, to score
the replay.

Run the exploratory study after downloading the datasets:

    python adaptive_study.py --data-dir data --output results/adaptive_study_results.json

The threshold sweep is exploratory.  It must not be presented as a tuned clinical
operating point without a training-side threshold-selection procedure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import ledoit_wolf

import calibration_free as cf


SEQUENCE_LENGTHS = {"rome_als": 12, "rome_healthy": 12, "etsu_als": 18}
DEFAULT_THRESHOLDS = (1.0,)
EPSILON = 1e-8


@dataclass(frozen=True)
class AlignedSeed:
    """A source seed and the covariance estimated from source participants."""

    target: np.ndarray
    non_target: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class StoppingResult:
    """Letter-level online replay output."""

    correct: np.ndarray
    sequences: np.ndarray
    forced_at_max: np.ndarray


def _feature_scale(x: np.ndarray) -> np.ndarray:
    scale = np.asarray(x, dtype=float).std(axis=0)
    return np.where(scale > EPSILON, scale, 1.0)


def aligned_seed_from(people: list[cf.Participant]) -> AlignedSeed:
    """Build a feature-standardised seed and source covariance from labelled source data.

    Labels identify source target/non-target flashes, but no labels from the eventual
    test participant are used.  Each source participant is normalised separately so a
    healthy recording with a larger voltage scale cannot dominate the ALS seed.
    """
    target, other, all_rows = [], [], []
    for participant in people:
        rows = np.concatenate(participant.x)
        centre, scale = rows.mean(axis=0), _feature_scale(rows)
        for x, lit, cell in zip(participant.x, participant.lit, participant.target):
            normalised = (x - centre) / scale
            all_rows.append(normalised)
            target.append(normalised[lit[:, cell]])
            other.append(normalised[~lit[:, cell]])
    if not target or not other:
        raise ValueError("aligned seed needs labelled source flashes")
    rows = np.concatenate(all_rows)
    return AlignedSeed(
        np.concatenate(target).mean(axis=0),
        np.concatenate(other).mean(axis=0),
        ledoit_wolf(rows)[0],
    )


def _psd_root(covariance: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Stable square root or inverse square root of a covariance matrix."""
    symmetric = (covariance + covariance.T) / 2.0
    values, vectors = eigh(symmetric)
    floor = max(float(values.max()) * 1e-6, EPSILON)
    values = np.maximum(values, floor)
    weights = 1.0 / np.sqrt(values) if inverse else np.sqrt(values)
    return (vectors * weights) @ vectors.T


def adapt_seed(seed: AlignedSeed, seen: np.ndarray, method: str = "coral") -> tuple[np.ndarray, np.ndarray]:
    """Transport a source seed into the current participant's feature units.

    ``seen`` contains only the current and previously observed flashes.  ``featurewise``
    fixes per-feature scale; ``coral`` additionally transports covariance geometry.
    Both methods are label-free on the test participant.
    """
    if method not in {"featurewise", "coral"}:
        raise ValueError("method must be 'featurewise' or 'coral'")
    centre, scale = seen.mean(axis=0), _feature_scale(seen)
    if method == "featurewise":
        transform = np.eye(seen.shape[1])
    else:
        target_covariance = ledoit_wolf((seen - centre) / scale)[0]
        transform = _psd_root(target_covariance) @ _psd_root(seed.covariance, inverse=True)
    target = centre + scale * (seed.target @ transform.T)
    non_target = centre + scale * (seed.non_target @ transform.T)
    return target, non_target


def _decision(
    current_x: np.ndarray,
    current_lit: np.ndarray,
    seen: np.ndarray,
    sum_t: np.ndarray,
    sum_n: np.ndarray,
    memory_weight: float,
    seed: tuple[np.ndarray, np.ndarray] | AlignedSeed | None,
    alignment: str,
) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Decode a prefix and return best symbol, confidence, and current class means."""
    inverse = cf.shrunk_inverse(seen)
    mu_t, mu_n = cf.hypothesis_means(current_x, current_lit)
    seed_weight = cf.SEED_WEIGHT if seed is not None else 0.0
    if isinstance(seed, AlignedSeed):
        prior_t, prior_n = adapt_seed(seed, seen, alignment)
    elif seed is not None:
        prior_t = seen.std() * seed[0]
        prior_n = seen.std() * seed[1]
    else:
        prior_t = prior_n = 0.0
    if memory_weight + seed_weight > 0:
        total = memory_weight + seed_weight + 1.0
        delta = (seed_weight * prior_t + sum_t + mu_t) / total - (seed_weight * prior_n + sum_n + mu_n) / total
    else:
        delta = mu_t - mu_n
    distances = np.einsum("sf,fg,sg->s", delta, inverse, delta)
    best = int(np.argmax(distances))
    return best, cf.confidence(distances), mu_t, mu_n


def run_umm_aligned(
    participant: cf.Participant,
    seed: AlignedSeed,
    *,
    alignment: str = "coral",
) -> np.ndarray:
    """Run seeded UMM with featurewise or CORAL alignment."""
    correct = np.zeros(len(participant.x), dtype=bool)
    sum_t = sum_n = 0.0
    memory_weight = 0.0
    seen_chunks: list[np.ndarray] = []
    for letter, (x, lit) in enumerate(zip(participant.x, participant.lit)):
        seen_chunks.append(x)
        seen = np.concatenate(seen_chunks)
        best, score, mu_t, mu_n = _decision(x, lit, seen, sum_t, sum_n, memory_weight, seed, alignment)
        correct[letter] = best == participant.target[letter]
        c = min(score, 1.0)
        sum_t, sum_n = sum_t + c * mu_t[best], sum_n + c * mu_n[best]
        memory_weight += c
    return correct


def run_umm_stopping(
    participant: cf.Participant,
    *,
    sequence_length: int,
    threshold: float,
    min_sequences: int = 2,
    seed: tuple[np.ndarray, np.ndarray] | AlignedSeed | None = None,
    alignment: str = "coral",
) -> StoppingResult:
    """Run a confidence-based online stopping replay.

    A letter is selected at the first sequence whose UMM confidence reaches
    ``threshold`` after ``min_sequences``.  If it never reaches the threshold, the
    maximum available sequence is used and ``forced_at_max`` records that fallback.
    """
    if sequence_length <= 0 or threshold < 0 or min_sequences <= 0:
        raise ValueError("sequence_length and min_sequences must be positive; threshold must be non-negative")
    correct, used, forced = [], [], []
    sum_t = sum_n = 0.0
    memory_weight = 0.0
    previous_chunks: list[np.ndarray] = []
    for letter, (x, lit) in enumerate(zip(participant.x, participant.lit)):
        max_sequences = int(np.ceil(len(x) / sequence_length))
        chosen = None
        for sequence in range(1, max_sequences + 1):
            end = min(sequence * sequence_length, len(x))
            prefix_x, prefix_lit = x[:end], lit[:end]
            seen = np.concatenate(previous_chunks + [prefix_x])
            best, score, mu_t, mu_n = _decision(
                prefix_x, prefix_lit, seen, sum_t, sum_n, memory_weight, seed, alignment
            )
            if sequence >= min_sequences and (score >= threshold or sequence == max_sequences):
                chosen = (sequence, score, best, mu_t, mu_n)
                break
        assert chosen is not None
        sequence, score, best, mu_t, mu_n = chosen
        correct.append(best == participant.target[letter])
        used.append(sequence)
        forced.append(score < threshold)
        c = min(score, 1.0)
        sum_t, sum_n = sum_t + c * mu_t[best], sum_n + c * mu_n[best]
        memory_weight += c
        previous_chunks.append(x[: sequence * sequence_length])
    return StoppingResult(np.array(correct, dtype=bool), np.array(used, dtype=int), np.array(forced, dtype=bool))


def _summary(correct: list[np.ndarray], sequences: list[np.ndarray], forced: list[np.ndarray]) -> dict:
    participant_accuracy = np.array([c.mean() for c in correct], dtype=float)
    participant_sequences = np.array([s.mean() for s in sequences], dtype=float)
    participant_forced = np.array([f.mean() for f in forced], dtype=float)
    return {
        "accuracy": cf.summary(participant_accuracy),
        "mean_sequences": cf.summary(participant_sequences),
        "forced_at_max_fraction": cf.summary(participant_forced),
    }


def evaluate_alignment(datasets: dict[str, list[cf.Participant]]) -> dict:
    """Compare local, featurewise, and CORAL seeds for every target dataset.

    Each target participant is evaluated with a leave-one-participant-out local seed
    plus seeds made from every other dataset.  This makes the cross-site question
    explicit instead of hiding it inside a pooled score.
    """
    result = {"pipeline": "binned", "datasets": {}}
    for target_name, people in datasets.items():
        external = {
            source_name: (cf.seed_from(source_people), aligned_seed_from(source_people))
            for source_name, source_people in datasets.items()
            if source_name != target_name
        }
        methods: dict[str, list[np.ndarray]] = {
            "seeded_umm": [], "featurewise_seeded_umm": [], "coral_seeded_umm": []
        }
        for source_name in external:
            methods[f"seeded_umm[{source_name}]"] = []
            methods[f"featurewise_seeded_umm[{source_name}]"] = []
            methods[f"coral_seeded_umm[{source_name}]"] = []
        for person in people:
            others = [q for q in people if q.name != person.name]
            raw_seed = cf.seed_from(others)
            aligned_seed = aligned_seed_from(others)
            methods["seeded_umm"].append(cf.run_umm(person, memory=True, seed=raw_seed))
            methods["featurewise_seeded_umm"].append(
                run_umm_aligned(person, aligned_seed, alignment="featurewise")
            )
            methods["coral_seeded_umm"].append(run_umm_aligned(person, aligned_seed, alignment="coral"))
            for source_name, (external_raw, external_aligned) in external.items():
                methods[f"seeded_umm[{source_name}]"].append(
                    cf.run_umm(person, memory=True, seed=external_raw)
                )
                methods[f"featurewise_seeded_umm[{source_name}]"] .append(
                    run_umm_aligned(person, external_aligned, alignment="featurewise")
                )
                methods[f"coral_seeded_umm[{source_name}]"] .append(
                    run_umm_aligned(person, external_aligned, alignment="coral")
                )
        entry = {"participants": [p.name for p in people], "methods": {}}
        for method, scores in methods.items():
            entry["methods"][method] = cf.summary(np.array([s.mean() for s in scores]))
        entry["comparisons"] = {
            "coral - local": cf.paired(
                np.array([s.mean() for s in methods["coral_seeded_umm"]]),
                np.array([s.mean() for s in methods["seeded_umm"]]),
            ),
            "featurewise - local": cf.paired(
                np.array([s.mean() for s in methods["featurewise_seeded_umm"]]),
                np.array([s.mean() for s in methods["seeded_umm"]]),
            ),
        }
        for source_name in external:
            raw_name = f"seeded_umm[{source_name}]"
            coral_name = f"coral_seeded_umm[{source_name}]"
            entry["comparisons"][f"coral[{source_name}] - raw[{source_name}]"] = cf.paired(
                np.array([s.mean() for s in methods[coral_name]]),
                np.array([s.mean() for s in methods[raw_name]]),
            )
        result["datasets"][target_name] = entry
    return result


def evaluate_stopping(
    people: list[cf.Participant],
    *,
    dataset_name: str,
    thresholds: tuple[float, ...],
    min_sequences: int,
) -> dict:
    """Evaluate stopping on one dataset with a same-dataset seed."""
    sequence_length = SEQUENCE_LENGTHS[dataset_name]
    output = {"dataset": dataset_name, "sequence_length": sequence_length, "min_sequences": min_sequences, "thresholds": {}}
    for threshold in thresholds:
        methods: dict[str, list[np.ndarray]] = {"seeded_umm": [], "coral_seeded_umm": []}
        sequence_counts: dict[str, list[np.ndarray]] = {name: [] for name in methods}
        forced_counts: dict[str, list[np.ndarray]] = {name: [] for name in methods}
        for person in people:
            others = [q for q in people if q.name != person.name]
            raw = run_umm_stopping(
                person, sequence_length=sequence_length, threshold=threshold,
                min_sequences=min_sequences, seed=cf.seed_from(others)
            )
            aligned = run_umm_stopping(
                person, sequence_length=sequence_length, threshold=threshold,
                min_sequences=min_sequences, seed=aligned_seed_from(others), alignment="coral"
            )
            methods["seeded_umm"].append(raw.correct)
            methods["coral_seeded_umm"].append(aligned.correct)
            sequence_counts["seeded_umm"].append(raw.sequences)
            sequence_counts["coral_seeded_umm"].append(aligned.sequences)
            forced_counts["seeded_umm"].append(raw.forced_at_max)
            forced_counts["coral_seeded_umm"].append(aligned.forced_at_max)
        output["thresholds"][str(threshold)] = {
            method: _summary(methods[method], sequence_counts[method], forced_counts[method])
            for method in methods
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "adaptive_study_results.json")
    parser.add_argument("--dataset", choices=tuple(SEQUENCE_LENGTHS), default="rome_als")
    parser.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    parser.add_argument("--min-sequences", type=int, default=2)
    args = parser.parse_args()
    thresholds = tuple(float(value) for value in args.thresholds.split(",") if value.strip())
    datasets = cf.load_datasets(args.data_dir, "binned")
    result = {
        "schema_version": 1,
        "alignment": evaluate_alignment(datasets),
        "stopping": evaluate_stopping(
            datasets[args.dataset], dataset_name=args.dataset, thresholds=thresholds, min_sequences=args.min_sequences
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for dataset, entry in result["alignment"]["datasets"].items():
        print(dataset + ": " + ", ".join(f"{name}={value['mean']:.3f}" for name, value in entry["methods"].items()))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
