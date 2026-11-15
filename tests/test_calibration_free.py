"""Tests on synthetic spellers. No EEG download needed."""

import numpy as np

import calibration_free as cf

LETTERS = 20


def synthetic(name: str, seed: int, signal: float = 1.5, scale: float = 1.0, checkerboard: bool = False) -> cf.Participant:
    """Target flashes get a fixed pattern added to white noise. Row/column by default, random groups otherwise."""
    rng = np.random.default_rng(seed)
    pattern = np.zeros(16)
    pattern[:4] = signal
    xs, lits, targets = [], [], []
    for _ in range(LETTERS):
        if checkerboard:
            lit = np.zeros((126, cf.N_SYMBOLS), dtype=bool)
            for flash in range(126):
                lit[flash, rng.choice(cf.N_SYMBOLS, size=4, replace=False)] = True
        else:
            lit = cf.rowcol_lit(np.concatenate([rng.permutation(12) + 1 for _ in range(10)]))
        target = int(rng.integers(cf.N_SYMBOLS))
        x = rng.normal(size=(len(lit), 16))
        x[lit[:, target]] += pattern
        xs.append(x * scale); lits.append(lit); targets.append(target)
    return cf.Participant(name, xs, lits, np.array(targets))


def test_rowcol_lit_marks_one_row_or_one_column():
    lit = cf.rowcol_lit(np.array([1, 6, 7, 12]))
    assert lit.sum(axis=1).tolist() == [6, 6, 6, 6]
    assert lit[0, :6].all() and lit[1, 30:].all()            # rows 1 and 6
    assert lit[2, ::6].all() and lit[3, 5::6].all()          # columns 1 and 6
    assert (lit[0] & lit[2]).sum() == 1                      # a row and a column share exactly one cell


def test_hypothesis_means_match_a_direct_loop():
    rng = np.random.default_rng(0)
    x, lit = rng.normal(size=(120, 5)), cf.rowcol_lit(np.tile(np.arange(1, 13), 10))
    mu_t, mu_n = cf.hypothesis_means(x, lit)
    for s in (0, 17, 35):
        assert np.allclose(mu_t[s], x[lit[:, s]].mean(axis=0)) and np.allclose(mu_n[s], x[~lit[:, s]].mean(axis=0))


def test_confidence_is_gap_over_spread_and_never_negative():
    d = np.array([1.0, 2.0, 3.0, 10.0])
    assert np.isclose(cf.confidence(d), (10 - 3) / np.std([1.0, 2.0, 3.0]))
    assert cf.confidence(np.ones(5)) == 0.0


def test_supervised_lda_decodes_a_clean_speller():
    p = synthetic("a", 1)
    w = cf.lda_weights([cf.Participant("a", p.x[:8], p.lit[:8], p.target[:8])])
    hits = [cf.decode_linear(p.x[l], p.lit[l], w) == p.target[l] for l in range(8, LETTERS)]
    assert np.mean(hits) > 0.9


def test_umm_decodes_rowcol_and_checkerboard_and_ignores_labels():
    for checkerboard in (False, True):
        p = synthetic("a", 2, signal=2.0, checkerboard=checkerboard)
        assert cf.run_umm(p, memory=True).mean() > 0.9
        # With impossible labels nothing can be "correct": labels are only used for scoring.
        blind = cf.Participant(p.name, p.x, p.lit, np.full(LETTERS, -1))
        assert not cf.run_umm(blind, memory=True).any()


def test_seed_helps_when_signal_is_weak_and_is_unit_free():
    people = [synthetic(f"p{i}", 10 + i, signal=0.45) for i in range(4)]
    plain = cf.run_umm(people[0], memory=True).mean()
    seeded = cf.run_umm(people[0], memory=True, seed=cf.seed_from(people[1:])).mean()
    assert seeded >= plain and seeded > 0.5
    # The same seed people recorded in units 50x larger must give exactly the same decisions.
    loud = [synthetic(f"p{i}", 10 + i, signal=0.45, scale=50.0) for i in range(1, 4)]
    assert np.allclose(cf.seed_from(loud)[0], cf.seed_from(people[1:])[0])
    assert (cf.run_umm(people[0], memory=True, seed=cf.seed_from(loud)) ==
            cf.run_umm(people[0], memory=True, seed=cf.seed_from(people[1:]))).all()


def test_evaluate_never_uses_the_test_participant_for_transfer_or_seed(monkeypatch):
    people = [synthetic(f"p{i}", 20 + i) for i in range(3)]
    transfer_from, seed_from_people = [], []
    original_means, original_seed = cf.labelled_means, cf.seed_from
    monkeypatch.setattr(cf, "labelled_means", lambda ps: (transfer_from.append([q.name for q in ps]), original_means(ps))[1])
    monkeypatch.setattr(cf, "seed_from", lambda ps: (seed_from_people.append([q.name for q in ps]), original_seed(ps))[1])
    out = cf.evaluate(people, calibration=8, external={"elsewhere": [synthetic("x", 99)]})
    for i, p in enumerate(people):
        others = [q.name for q in people if q.name != p.name]
        assert others in transfer_from and others in seed_from_people
    assert all(p.name not in names for names in seed_from_people for p in people if len(names) == 2 and p.name not in names)
    assert ["x"] in seed_from_people
    assert all(np.isnan(c[:8]).all() and not np.isnan(c[8:]).any() for c in out["supervised"])
    assert set(out) == {"supervised", "transfer", "umm", "umm_memory", "seeded_umm", "seeded_umm[elsewhere]"}
    assert all(len(c) == LETTERS for cs in out.values() for c in cs)


def test_first_n_flashes_only_uses_those_flashes():
    p = synthetic("a", 3)
    wiped = cf.Participant(p.name, [np.r_[x[:12], np.zeros_like(x[12:])] for x in p.x], p.lit, p.target)
    assert (cf.run_umm(p.first(12), memory=False) == cf.run_umm(wiped.first(12), memory=False)).all()


def test_sign_flip_permutation_is_exact():
    assert cf.sign_flip_p(np.ones(8)) == 2 / 256              # only the all-plus and all-minus assignments
    assert cf.sign_flip_p(np.array([1.0, -1.0])) == 1.0
    assert cf.sign_flip_p(np.zeros(5)) == 1.0


def test_paired_and_summary_report_participant_level_numbers():
    a, b = np.array([0.9, 0.8, 0.7, 0.6]), np.array([0.5, 0.8, 0.2, 0.7])
    result = cf.paired(a, b)
    assert np.isclose(result["mean_difference"], 0.2) and result["better_same_worse"] == [2, 1, 1] and result["n"] == 4
    s = cf.summary(a)
    assert s["mean"] == 0.75 and s["min"] == 0.6 and s["ci95"][0] <= 0.75 <= s["ci95"][1]
