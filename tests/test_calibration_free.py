"""Tests on synthetic spellers. No EEG download needed."""

import numpy as np

import calibration_free as cf


def synthetic(name: str, seed: int, signal: float = 1.5, flip: bool = False) -> cf.Participant:
    """35 letters, 10 sequences; target flashes get a fixed pattern added to white noise."""
    rng = np.random.default_rng(seed)
    pattern = np.zeros(16)
    pattern[:4] = -signal if flip else signal
    x = rng.normal(size=(cf.LETTERS, cf.FLASHES, 16))
    codes = np.stack([np.concatenate([rng.permutation(12) + 1 for _ in range(10)]) for _ in range(cf.LETTERS)])
    target = np.column_stack([rng.integers(1, 7, cf.LETTERS), rng.integers(7, 13, cf.LETTERS)])
    for letter in range(cf.LETTERS):
        hit = (codes[letter] == target[letter, 0]) | (codes[letter] == target[letter, 1])
        x[letter, hit] += pattern
    return cf.Participant(name, x, codes, target)


def test_symbols_cover_the_matrix():
    assert len(cf.SYMBOLS) == 36 and len(set(cf.SYMBOLS)) == 36
    assert all(1 <= r <= 6 and 7 <= c <= 12 for r, c in cf.SYMBOLS)


def test_class_means_split_by_row_and_column():
    x = np.arange(12, dtype=float).reshape(12, 1)
    codes = np.arange(1, 13)
    mu_t, mu_n = cf.class_means(x, codes, (1, 7))
    assert mu_t[0] == (0 + 6) / 2 and np.isclose(mu_n[0], (66 - 6) / 10)


def test_confidence_is_gap_over_spread_and_never_negative():
    d = np.array([1.0, 2.0, 3.0, 10.0])
    assert np.isclose(cf.confidence(d), (10 - 3) / np.std([1.0, 2.0, 3.0]))
    assert cf.confidence(np.ones(5)) == 0.0


def test_supervised_lda_decodes_a_clean_speller():
    p = synthetic("a", 1)
    w, _, _ = cf.lda_weights([p.x[l] for l in range(15)], [p.codes[l] for l in range(15)], [p.target[l] for l in range(15)])
    hits = [cf.decode_linear(p.x[l], p.codes[l], w) == tuple(p.target[l]) for l in range(15, 35)]
    assert np.mean(hits) > 0.9


def test_umm_decodes_a_clean_speller_and_ignores_labels():
    p = synthetic("a", 2, signal=2.0)
    assert cf.run_umm(p, 120, memory=True).mean() > 0.9
    # With the labels replaced by an impossible symbol nothing can be "correct": labels are only used for scoring.
    blind = cf.Participant(p.name, p.x, p.codes, np.zeros_like(p.target))
    assert not cf.run_umm(blind, 120, memory=True).any()


def test_seed_from_other_people_is_used_and_helps_when_signal_is_weak():
    people = [synthetic(f"p{i}", 10 + i, signal=0.45) for i in range(4)]
    _, seed_t, seed_n = cf._transfer(people[1:], 120)
    plain = cf.run_umm(people[0], 120, memory=True).mean()
    seeded = cf.run_umm(people[0], 120, memory=True, seed=(seed_t, seed_n)).mean()
    assert seeded >= plain and seeded > 0.5


def test_evaluate_never_trains_transfer_on_the_test_participant(monkeypatch):
    people = [synthetic(f"p{i}", 20 + i) for i in range(3)]
    seen = []
    original = cf._transfer
    monkeypatch.setattr(cf, "_transfer", lambda others, n: (seen.append([q.name for q in others]), original(others, n))[1])
    out = cf.evaluate(people, 120)
    assert seen == [["p1", "p2"], ["p0", "p2"], ["p0", "p1"]]
    assert np.isnan(out["supervised"][:, : cf.CALIBRATION_LETTERS]).all()
    assert not np.isnan(out["supervised"][:, cf.CALIBRATION_LETTERS:]).any()
    assert all(out[m].shape == (3, cf.LETTERS) for m in cf.METHODS)


def test_fewer_sequences_use_only_the_first_flashes():
    p = synthetic("a", 3)
    changed = cf.Participant(p.name, p.x.copy(), p.codes, p.target)
    changed.x[:, 12:] = 0.0  # wipe everything after the first sequence
    assert (cf.run_umm(p, 12, memory=False) == cf.run_umm(changed, 12, memory=False)).all()
