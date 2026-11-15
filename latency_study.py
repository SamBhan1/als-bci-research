"""Why seeds fail across recording setups, a label-free timing correction, and a UMM fidelity check.

See PROTOCOL.md, addendum 3. Uses the `binned` pipeline (16 bins of 50 ms x 8 channels) for the seed work
and the `authors` pipeline for the fidelity grid.

    python latency_study.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import calibration_free as cf

BINS, BIN_MS = 16, 50
SHIFTS = tuple(range(-6, 7))            # -300 ms .. +300 ms, fixed in advance
ALS = ("rome_als", "etsu_als")


# ---------------------------------------------------------------- diagnostic

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


def shift_seed(vector: np.ndarray, bins: int) -> np.ndarray:
    """Delay a seed mean by `bins` time bins (negative = earlier), padding with zeros."""
    wave = vector.reshape(BINS, -1)
    out = np.zeros_like(wave)
    if bins >= 0:
        out[bins:] = wave[: BINS - bins]
    else:
        out[:bins] = wave[-bins:]
    return out.ravel()


def response_similarity(datasets: dict[str, list[cf.Participant]]) -> dict:
    """Cosine similarity of average target-minus-non-target responses between datasets, per time shift."""
    waves = {name: np.subtract(*cf.seed_from(people)) for name, people in datasets.items()}
    out: dict = {"peak_ms_at_Pz": {}, "pairs": {}}
    pz = cf.CHANNELS.index("Pz")
    for name, wave in waves.items():
        grid = wave.reshape(BINS, -1)
        column = pz if grid.shape[1] > pz else 0          # synthetic test data has a single channel
        out["peak_ms_at_Pz"][name] = int(np.argmax(grid[:, column]) * BIN_MS + BIN_MS // 2)
    names = list(waves)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            by_shift = {str(k * BIN_MS): round(cosine(waves[a], shift_seed(waves[b], k)), 3) for k in SHIFTS}
            best = max(by_shift, key=by_shift.get)
            out["pairs"][f"{a} vs {b}"] = {"at_zero_shift": by_shift["0"], "best_shift_ms_applied_to_second": int(best),
                                           "at_best_shift": by_shift[best], "by_shift_ms": by_shift}
    return out


# ---------------------------------------------------------------- latency-searched seeded UMM

def run_latency_seeded(p: cf.Participant, seed: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Seeded UMM that tries the seed at every shift and follows the one with the highest running confidence.

    Returns (per-letter correctness, per-letter chosen shift in bins). Label-free and causal: the shift used
    for letter i depends only on flashes of letters 1..i.
    """
    seeds = [(shift_seed(seed[0], k), shift_seed(seed[1], k)) for k in SHIFTS]
    running = np.zeros(len(SHIFTS))
    correct, chosen = np.zeros(len(p.x), dtype=bool), np.zeros(len(p.x), dtype=int)
    sum_t, sum_n, weight = 0.0, 0.0, 0.0
    seen_chunks = []
    for letter, (x, lit) in enumerate(zip(p.x, p.lit)):
        seen_chunks.append(x)
        seen = np.concatenate(seen_chunks)
        inverse, scale = cf.shrunk_inverse(seen), seen.std()
        mu_t, mu_n = cf.hypothesis_means(x, lit)
        total = cf.SEED_WEIGHT + weight + 1
        all_distances = []
        for seed_t, seed_n in seeds:
            delta = ((cf.SEED_WEIGHT * scale * seed_t + sum_t + mu_t) - (cf.SEED_WEIGHT * scale * seed_n + sum_n + mu_n)) / total
            all_distances.append(np.einsum("sf,fg,sg->s", delta, inverse, delta))
        confidences = np.array([min(cf.confidence(d), 1.0) for d in all_distances])
        running += confidences
        pick = int(np.argmax(running))
        best = int(np.argmax(all_distances[pick]))
        correct[letter], chosen[letter] = best == p.target[letter], SHIFTS[pick]
        c = confidences[pick]
        sum_t, sum_n, weight = sum_t + c * mu_t[best], sum_n + c * mu_n[best], weight + c
    return correct, chosen


def seed_study(datasets: dict[str, list[cf.Participant]]) -> dict:
    out: dict = {"datasets": {}}
    scores: dict[str, dict[str, np.ndarray]] = {}
    for name, people in datasets.items():
        methods: dict[str, list[float]] = {"umm_memory": [], "seeded_umm": [], "latency_seeded": []}
        final_shift: dict[str, list[int]] = {"latency_seeded": []}
        external = {other: cf.seed_from(q) for other, q in datasets.items() if other != name}
        for other in external:
            methods[f"seeded_umm[{other}]"], methods[f"latency_seeded[{other}]"] = [], []
            final_shift[f"latency_seeded[{other}]"] = []
        for p in people:
            local = cf.seed_from([q for q in people if q.name != p.name])
            methods["umm_memory"].append(cf.run_umm(p, memory=True).mean())
            methods["seeded_umm"].append(cf.run_umm(p, memory=True, seed=local).mean())
            correct, chosen = run_latency_seeded(p, local)
            methods["latency_seeded"].append(correct.mean()); final_shift["latency_seeded"].append(int(chosen[-1]) * BIN_MS)
            for other, seed in external.items():
                methods[f"seeded_umm[{other}]"].append(cf.run_umm(p, memory=True, seed=seed).mean())
                correct, chosen = run_latency_seeded(p, seed)
                methods[f"latency_seeded[{other}]"].append(correct.mean())
                final_shift[f"latency_seeded[{other}]"].append(int(chosen[-1]) * BIN_MS)
        s = scores[name] = {m: np.array(v) for m, v in methods.items()}
        entry = {"participants": [p.name for p in people], "methods": {m: cf.summary(v) for m, v in s.items()},
                 "final_shift_ms": final_shift,
                 "comparisons": {"latency_seeded - seeded_umm (local seed)": cf.paired(s["latency_seeded"], s["seeded_umm"])}}
        for other in external:
            entry["comparisons"][f"latency_seeded[{other}] - seeded_umm[{other}]"] = cf.paired(s[f"latency_seeded[{other}]"], s[f"seeded_umm[{other}]"])
            entry["comparisons"][f"latency_seeded[{other}] - umm_memory"] = cf.paired(s[f"latency_seeded[{other}]"], s["umm_memory"])
            entry["comparisons"][f"latency_seeded[{other}] - seeded_umm (local seed)"] = cf.paired(s[f"latency_seeded[{other}]"], s["seeded_umm"])
        out["datasets"][name] = entry
        print(f"  {name}: " + " | ".join(f"{m} {v.mean():.3f}" for m, v in s.items()))
    # New-clinic scenario: every person with ALS seeded only from the other site's ALS participants.
    other = {"rome_als": "etsu_als", "etsu_als": "rome_als"}
    pool = lambda pattern: np.concatenate([scores[d][pattern.format(other=other[d])] for d in ALS])
    cross, raw, memory, local = pool("latency_seeded[{other}]"), pool("seeded_umm[{other}]"), pool("umm_memory"), pool("seeded_umm")
    out["new_clinic_all_als"] = {
        "latency_seeded[other site]": cf.summary(cross), "seeded_umm[other site]": cf.summary(raw),
        "umm_memory": cf.summary(memory), "seeded_umm (local seed)": cf.summary(local),
        "latency_seeded[other site] - umm_memory": cf.paired(cross, memory),
        "latency_seeded[other site] - seeded_umm[other site]": cf.paired(cross, raw),
        "latency_seeded[other site] - seeded_umm (local seed)": cf.paired(cross, local),
    }
    return out


def site_shift(local_people: list[cf.Participant], borrowed: tuple[np.ndarray, np.ndarray]) -> int:
    """Shift (in bins) that best matches a borrowed average response to labelled people at the test site."""
    local = np.subtract(*cf.seed_from(local_people))
    wave = borrowed[0] - borrowed[1]
    return max(SHIFTS, key=lambda k: cosine(local, shift_seed(wave, k)))


def fixed_shift_check(datasets: dict[str, list[cf.Participant]]) -> dict:
    """Post-hoc test of the timing explanation (PROTOCOL.md): borrowed seed shifted by a site-level offset."""
    out = {}
    for name, people in datasets.items():
        for other, donors in datasets.items():
            if other == name:
                continue
            seed = cf.seed_from(donors)
            scores, shifts = [], []
            for p in people:
                k = site_shift([q for q in people if q.name != p.name], seed)
                shifted = (shift_seed(seed[0], k), shift_seed(seed[1], k))
                scores.append(cf.run_umm(p, memory=True, seed=shifted).mean()); shifts.append(k * BIN_MS)
            out[f"{name} seeded from {other}"] = {**cf.summary(np.array(scores)), "shift_ms": shifts}
            print(f"  {name} <- {other}: {np.mean(scores):.3f} shifts {sorted(set(shifts))}")
    return out


# ---------------------------------------------------------------- UMM fidelity grid

def run_umm_variant(p: cf.Participant, *, covariance: str, mean: str, centre: bool) -> np.ndarray:
    """UMM as described in the paper. covariance: curr|cumu; mean: curr|opti|conf; centre: per-letter centring."""
    correct = np.zeros(len(p.x), dtype=bool)
    sum_t, sum_n, weight = 0.0, 0.0, 0.0
    pool = []
    for letter, (x, lit) in enumerate(zip(p.x, p.lit)):
        pool.append(x - x.mean(axis=0) if centre else x)
        inverse = cf.shrunk_inverse(np.concatenate(pool) if covariance == "cumu" else pool[-1])
        mu_t, mu_n = cf.hypothesis_means(x, lit)
        if mean != "curr" and weight > 0:
            delta = (sum_t + mu_t) / (weight + 1) - (sum_n + mu_n) / (weight + 1)
        else:
            delta = mu_t - mu_n
        distances = np.einsum("sf,fg,sg->s", delta, inverse, delta)
        best = int(np.argmax(distances))
        correct[letter] = best == p.target[letter]
        c = 1.0 if mean == "opti" else min(cf.confidence(distances), 1.0)
        sum_t, sum_n, weight = sum_t + c * mu_t[best], sum_n + c * mu_n[best], weight + c
    return correct


def fidelity_grid(people: list[cf.Participant]) -> dict:
    grid = {}
    for covariance in ("curr", "cumu"):
        for mean in ("curr", "opti", "conf"):
            for centre in (False, True):
                values = np.array([run_umm_variant(p, covariance=covariance, mean=mean, centre=centre).mean() for p in people])
                key = f"cov={covariance} mean={mean} centre={'letter' if centre else 'none'}"
                grid[key] = {"mean": round(float(values.mean()), 4), "min": round(float(values.min()), 4),
                             "participants_at_or_below_10pct": int((values <= 0.10).sum()),
                             "per_participant": [round(float(v), 4) for v in values]}
                print(f"  {key}: {values.mean():.3f}")
    return {"published_als_accuracy": 0.7786, "variants": grid,
            "best_variant": max(grid, key=lambda k: grid[k]["mean"]), "best_mean": max(v["mean"] for v in grid.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "latency_study_results.json")
    args = parser.parse_args()
    binned = cf.load_datasets(args.data_dir, "binned")
    results = {"shifts_ms": [k * BIN_MS for k in SHIFTS], "seed_weight": cf.SEED_WEIGHT,
               "response_similarity": response_similarity(binned)}
    print("seed study (binned)")
    results["seed_study"] = seed_study(binned)
    print("fixed site-level shift (post-hoc)")
    results["fixed_shift_check"] = fixed_shift_check(binned)
    print("UMM fidelity grid on rome_als (authors' preprocessing)")
    results["umm_fidelity"] = fidelity_grid(cf.load_datasets(args.data_dir, "authors")["rome_als"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
