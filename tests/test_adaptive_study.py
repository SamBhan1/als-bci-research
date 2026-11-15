"""Tests for the exploratory adaptive extensions using synthetic spellers only."""

import numpy as np

import adaptive_study as adaptive
import calibration_free as cf


def synthetic(name: str, seed: int, signal: float = 1.5, scale: float = 1.0) -> cf.Participant:
    rng = np.random.default_rng(seed)
    pattern = np.zeros(16)
    pattern[:4] = signal
    xs, lits, targets = [], [], []
    for _ in range(8):
        lit = cf.rowcol_lit(np.concatenate([rng.permutation(12) + 1 for _ in range(10)]))
        target = int(rng.integers(cf.N_SYMBOLS))
        x = rng.normal(size=(len(lit), 16))
        x[lit[:, target]] += pattern
        xs.append(x * scale)
        lits.append(lit)
        targets.append(target)
    return cf.Participant(name, xs, lits, np.array(targets))


def test_aligned_seed_is_invariant_to_source_voltage_units():
    source = [synthetic(f"source{i}", 10 + i) for i in range(3)]
    loud = [synthetic(f"source{i}", 10 + i, scale=50.0) for i in range(3)]
    normal = adaptive.aligned_seed_from(source)
    scaled = adaptive.aligned_seed_from(loud)
    assert np.allclose(normal.target, scaled.target)
    assert np.allclose(normal.non_target, scaled.non_target)
    assert np.allclose(normal.covariance, scaled.covariance)


def test_coral_decoder_and_stopping_replay_return_letter_level_outputs():
    people = [synthetic(f"p{i}", 20 + i, signal=2.0) for i in range(3)]
    seed = adaptive.aligned_seed_from(people[1:])
    aligned = adaptive.run_umm_aligned(people[0], seed, alignment="coral")
    assert aligned.shape == (8,)
    stopped = adaptive.run_umm_stopping(
        people[0], sequence_length=12, threshold=0.5, min_sequences=2,
        seed=seed, alignment="coral"
    )
    assert len(stopped.correct) == len(stopped.sequences) == len(stopped.forced_at_max) == 8
    assert np.all(stopped.sequences >= 2) and np.all(stopped.sequences <= 10)


def test_stopping_does_not_use_flashes_after_the_decision():
    people = [synthetic(f"p{i}", 30 + i, signal=2.0) for i in range(3)]
    seed = cf.seed_from(people[1:])
    original = adaptive.run_umm_stopping(
        people[0], sequence_length=12, threshold=0.5, min_sequences=2, seed=seed
    )
    wiped = cf.Participant(
        people[0].name,
        [np.r_[x[:n * 12], np.zeros_like(x[n * 12:])] for x, n in zip(people[0].x, original.sequences)],
        people[0].lit,
        people[0].target,
    )
    replay = adaptive.run_umm_stopping(
        wiped, sequence_length=12, threshold=0.5, min_sequences=2, seed=seed
    )
    assert np.array_equal(original.correct, replay.correct)
    assert np.array_equal(original.sequences, replay.sequences)
