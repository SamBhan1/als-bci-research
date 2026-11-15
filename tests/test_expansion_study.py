"""Tests for the expansion study on synthetic spellers."""

import numpy as np

import calibration_free as cf
import expansion_study as es


def synthetic(name: str, seed: int, pattern_start: int = 0, signal: float = 1.0, letters: int = 14) -> cf.Participant:
    rng = np.random.default_rng(seed)
    pattern = np.zeros(16)
    pattern[pattern_start: pattern_start + 4] = signal
    xs, lits, targets = [], [], []
    for _ in range(letters):
        lit = cf.rowcol_lit(np.concatenate([rng.permutation(12) + 1 for _ in range(10)]))
        target = int(rng.integers(cf.N_SYMBOLS))
        x = rng.normal(size=(len(lit), 16))
        x[lit[:, target]] += pattern
        xs.append(x); lits.append(lit); targets.append(target)
    return cf.Participant(name, xs, lits, np.array(targets))


def test_monte_carlo_sign_flip_agrees_with_exact_and_handles_large_samples():
    small = np.array([0.3, 0.1, -0.05, 0.2, 0.4, 0.0, 0.15])
    assert es.mc_sign_flip_p(small) == cf.sign_flip_p(small)             # small samples use the exact test
    rng = np.random.default_rng(1)
    null = rng.normal(size=27)
    shifted = null * 0.2 + 1.0
    assert es.mc_sign_flip_p(shifted, draws=100_000) < 0.001
    assert es.mc_sign_flip_p(null - null.mean(), draws=100_000) > 0.2
    assert es.mc_sign_flip_p(np.zeros(27)) == 1.0


def test_paired_reports_large_samples_with_monte_carlo():
    rng = np.random.default_rng(2)
    a = rng.uniform(0.6, 1.0, size=27)
    result = es.paired(a, a - 0.2)
    assert result["n"] == 27 and np.isclose(result["mean_difference"], 0.2) and result["p_sign_flip"] < 0.001
    assert result["better_same_worse"] == [27, 0, 0]


def test_similarity_is_high_for_matching_responses_and_low_for_mismatched():
    same_a = [synthetic(f"a{i}", i, pattern_start=4, signal=2.0) for i in range(3)]
    same_b = [synthetic(f"b{i}", 10 + i, pattern_start=4, signal=2.0) for i in range(3)]
    moved = [synthetic(f"c{i}", 20 + i, pattern_start=10, signal=2.0) for i in range(3)]
    plain, adjusted = es.similarity(same_a, same_b)
    assert plain > 0.9 and adjusted > 0.9
    plain, adjusted = es.similarity(moved, same_b)
    assert abs(plain) < 0.3 and abs(adjusted) < 0.3


def test_seed_size_curve_reports_each_size_and_a_threshold():
    people = [synthetic(f"p{i}", 30 + i, signal=0.6) for i in range(5)]
    base = np.array([cf.run_umm(p, memory=True).mean() for p in people])
    curve = es.seed_size_curve(people, base)
    assert list(curve["by_number_of_donors"]) == ["1", "2", "3"]          # sizes above 4 donors are skipped
    assert curve["smallest_enough"] in (None, 1, 2, 3)
    assert all(0.0 <= v["mean"] <= 1.0 and v["n"] == 5 for v in curve["by_number_of_donors"].values())


def test_borrowed_seed_never_uses_the_same_named_person_and_flags_same_people(monkeypatch):
    wet = [synthetic(f"N_{i:02d}", 40 + i) for i in range(3)]
    dry = [synthetic(f"N_{i:02d}", 50 + i) for i in range(3)]
    other = [synthetic(f"B_{i:02d}", 60 + i) for i in range(3)]
    datasets = {"etsu_n_wet": wet, "etsu_n_dry": dry, "etsu_b": other}
    seen = []
    original = cf.seed_from
    monkeypatch.setattr(cf, "seed_from", lambda people: (seen.append([q.name for q in people]), original(people))[1])
    base = {n: np.array([cf.run_umm(p, memory=True).mean() for p in people]) for n, people in datasets.items()}
    local = {n: base[n] for n in datasets}
    result = es.borrowed(datasets, base, local)
    assert result["pairs"]["etsu_n_wet <- etsu_n_dry"]["same_people"] and not result["pairs"]["etsu_b <- etsu_n_dry"]["same_people"]
    assert ["N_01", "N_02"] in seen and ["N_00", "N_02"] in seen           # the test person's other session is left out
    assert result["what_makes_a_seed_travel"]["ordered_pairs"] == 4        # six ordered pairs minus the two wet/dry ones
