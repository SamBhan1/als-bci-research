"""Tests for the timing-correction study on synthetic spellers (16 time bins, one channel)."""

import numpy as np

import calibration_free as cf
import latency_study as ls

LETTERS = 20


def synthetic(name: str, seed: int, first_bin: int, signal: float = 0.6) -> cf.Participant:
    """Row/column speller whose target response occupies four time bins starting at `first_bin`."""
    rng = np.random.default_rng(seed)
    pattern = np.zeros(ls.BINS)
    pattern[first_bin: first_bin + 4] = signal
    xs, lits, targets = [], [], []
    for _ in range(LETTERS):
        lit = cf.rowcol_lit(np.concatenate([rng.permutation(12) + 1 for _ in range(10)]))
        target = int(rng.integers(cf.N_SYMBOLS))
        x = rng.normal(size=(len(lit), ls.BINS))
        x[lit[:, target]] += pattern
        xs.append(x); lits.append(lit); targets.append(target)
    return cf.Participant(name, xs, lits, np.array(targets))


def test_shift_seed_delays_and_advances_with_zero_padding():
    wave = np.arange(1.0, 17.0)
    assert ls.shift_seed(wave, 0).tolist() == wave.tolist()
    assert ls.shift_seed(wave, 2).tolist() == [0, 0, *range(1, 15)]
    assert ls.shift_seed(wave, -3).tolist() == [*range(4, 17), 0, 0, 0]
    two_channels = np.arange(32.0)
    assert ls.shift_seed(two_channels, 1).reshape(16, 2)[1].tolist() == [0.0, 1.0]   # whole time bins move, channels stay


def test_response_similarity_finds_the_time_offset():
    early = [synthetic(f"e{i}", i, first_bin=3, signal=2.0) for i in range(3)]
    late = [synthetic(f"l{i}", 10 + i, first_bin=7, signal=2.0) for i in range(3)]
    pair = ls.response_similarity({"early": early, "late": late})["pairs"]["early vs late"]
    assert pair["best_shift_ms_applied_to_second"] == -4 * ls.BIN_MS      # the late response must move 200 ms earlier
    assert pair["at_best_shift"] > 0.9 > pair["at_zero_shift"]


def test_latency_search_recovers_a_mistimed_seed_without_labels():
    user = synthetic("u", 1, first_bin=8)
    donors = [synthetic(f"d{i}", 20 + i, first_bin=4) for i in range(3)]   # same shape, 200 ms too early
    seed = cf.seed_from(donors)
    raw = cf.run_umm(user, memory=True, seed=seed).mean()
    correct, chosen = ls.run_latency_seeded(user, seed)
    assert correct.mean() > raw + 0.3 and correct.mean() > 0.7
    assert abs(chosen[-1] - 4) <= 1                                      # about +200 ms; the square test pattern is flat
    # Decisions do not depend on the labels.
    blind = cf.Participant(user.name, user.x, user.lit, np.full(LETTERS, -1))
    assert (ls.run_latency_seeded(blind, seed)[1] == chosen).all()


def test_latency_search_keeps_zero_shift_for_a_well_timed_seed():
    user = synthetic("u", 2, first_bin=6)
    donors = [synthetic(f"d{i}", 30 + i, first_bin=6) for i in range(3)]
    correct, chosen = ls.run_latency_seeded(user, cf.seed_from(donors))
    assert abs(chosen[-1]) <= 1 and correct.mean() > 0.7


def test_umm_variants_run_and_decode_a_clean_speller():
    user = synthetic("u", 3, first_bin=6, signal=2.0)
    for covariance in ("curr", "cumu"):
        for mean in ("curr", "opti", "conf"):
            for centre in (False, True):
                assert ls.run_umm_variant(user, covariance=covariance, mean=mean, centre=centre).mean() > 0.9
    # The (cumu, conf, no centring) variant is exactly the main study's umm_memory.
    assert (ls.run_umm_variant(user, covariance="cumu", mean="conf", centre=False) == cf.run_umm(user, memory=True)).all()
