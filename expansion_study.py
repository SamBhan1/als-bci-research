"""27 more people with ALS, how small a seed can be, and what makes a seed travel (PROTOCOL.md, addendum 4).

Adds bigP3BCI Study B (`etsu_b`) and Study N (`etsu_n_wet`, `etsu_n_dry`) to the three datasets of the main
study. `binned` pipeline throughout.

    python download_bigp3bci_more.py
    python expansion_study.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.covariance import ledoit_wolf

import calibration_free as cf

NEW_CALIBRATION = 12
SEED_SIZES = (1, 2, 3, 5, 8, 12, 18)
DRAWS = 10
SAME_PEOPLE = {frozenset(("etsu_n_wet", "etsu_n_dry"))}
SAME_LAB = ("etsu_als", "etsu_b", "etsu_n_wet")


def load_new(data_dir: Path) -> tuple[dict[str, list[cf.Participant]], dict[str, str]]:
    """Returns (datasets, problems). A dataset that cannot be loaded on the eight channels is left out."""
    groups: dict[str, dict[str, list[Path]]] = {"etsu_b": {}, "etsu_n_wet": {}, "etsu_n_dry": {}}
    for path in sorted((data_dir / "studyB").glob("B_*_SE001_CB_Train*.edf")):
        groups["etsu_b"].setdefault(path.name[:4], []).append(path)
    for path in sorted((data_dir / "studyN").glob("N_*_Train*.edf")):
        key = "etsu_n_wet" if "_Wet_" in path.name else "etsu_n_dry"
        groups[key].setdefault(path.name[:4], []).append(path)
    datasets, problems = {}, {}
    for name, people in groups.items():
        try:
            datasets[name] = [cf.load_etsu(paths, "binned") for paths in people.values()]
        except (ValueError, KeyError) as error:
            problems[name] = str(error)[:300]
    return datasets, problems


def mc_sign_flip_p(differences: np.ndarray, draws: int = 1_000_000, seed: int = 0) -> float:
    """Two-sided sign-flip test with random sign assignments, for samples too large to enumerate."""
    d = np.asarray(differences, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    if len(d) <= 20:
        return cf.sign_flip_p(d)
    rng, observed, hits = np.random.default_rng(seed), abs(d.sum()), 0
    for _ in range(draws // 50_000):
        signs = rng.integers(0, 2, size=(50_000, len(d))) * 2.0 - 1.0
        hits += int((np.abs(signs @ d) >= observed - 1e-12).sum())
    return float((hits + 1) / (draws + 1))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    out = cf.paired(a, b) if len(a) <= 20 else None
    if out is None:
        d = a - b
        low, high = cf.bootstrap_mean(d)
        out = {"mean_difference": round(float(d.mean()), 4), "ci95": [round(low, 4), round(high, 4)], "n": int(len(d)),
               "better_same_worse": [int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())],
               "p_sign_flip": round(mc_sign_flip_p(d), 5)}
    return out


# ---------------------------------------------------------------- A: confirmation

def confirmation(datasets: dict[str, list[cf.Participant]], base: dict[str, np.ndarray], local: dict[str, np.ndarray]) -> dict:
    out: dict = {"datasets": {}}
    for name in ("etsu_b", "etsu_n_wet", "etsu_n_dry"):
        if name not in datasets:
            continue
        correct = cf.evaluate(datasets[name], NEW_CALIBRATION)
        late = {m: cf.per_participant(c, "late", NEW_CALIBRATION) for m, c in correct.items()}
        everything = {m: cf.per_participant(c, "all", NEW_CALIBRATION) for m, c in correct.items() if m != "supervised"}
        out["datasets"][name] = {
            "participants": [p.name for p in datasets[name]], "letters": [len(p.x) for p in datasets[name]],
            "all_letters": {m: cf.summary(v) for m, v in everything.items()},
            "after_calibration_letters": {m: cf.summary(v) for m, v in late.items()},
            "seeded_umm - umm_memory": paired(everything["seeded_umm"], everything["umm_memory"]),
            "seeded_umm - transfer": paired(everything["seeded_umm"], everything["transfer"]),
            "seeded_umm - supervised (after calibration letters)": paired(late["seeded_umm"], late["supervised"]),
        }
    new = [n for n in ("etsu_b", "etsu_n_wet") if n in datasets]
    old = ("rome_als", "etsu_als")
    pool = lambda scores, names: np.concatenate([scores[n] for n in names])
    out["new_27_people"] = {"datasets": new, "seeded_umm": cf.summary(pool(local, new)), "umm_memory": cf.summary(pool(base, new)),
                            "seeded_umm - umm_memory": paired(pool(local, new), pool(base, new))}
    everyone = [*old, *new]
    out["all_people_with_als"] = {"datasets": everyone, "seeded_umm": cf.summary(pool(local, everyone)),
                                  "umm_memory": cf.summary(pool(base, everyone)),
                                  "seeded_umm - umm_memory": paired(pool(local, everyone), pool(base, everyone))}
    return out


# ---------------------------------------------------------------- B: seed size

def seed_size_curve(people: list[cf.Participant], base: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    curve = {}
    for k in SEED_SIZES:
        if k > len(people) - 1:
            continue
        per_person = []
        for p in people:
            others = [q for q in people if q.name != p.name]
            draws = 1 if k == len(others) else DRAWS
            scores = []
            for _ in range(draws):
                donors = [others[i] for i in rng.choice(len(others), size=k, replace=False)]
                scores.append(cf.run_umm(p, memory=True, seed=cf.seed_from(donors)).mean())
            per_person.append(float(np.mean(scores)))
        curve[str(k)] = cf.summary(np.array(per_person))
        print(f"  seed from {k:>2} people: {np.mean(per_person):.3f}")
    full = curve[str(max(int(k) for k in curve))]["mean"]
    enough = next((int(k) for k in curve if curve[k]["mean"] > base.mean() and curve[k]["mean"] >= full - 0.03), None)
    return {"umm_memory": round(float(base.mean()), 4), "by_number_of_donors": curve, "smallest_enough": enough}


# ---------------------------------------------------------------- C, D, E: borrowed seeds

def background_covariance(people: list[cf.Participant]) -> np.ndarray:
    return np.mean([ledoit_wolf(np.concatenate(p.x) / np.concatenate(p.x).std())[0] for p in people], axis=0)


def similarity(source: list[cf.Participant], target: list[cf.Participant]) -> tuple[float, float]:
    """(plain cosine, background-adjusted cosine) between the average responses of two datasets."""
    a, b = np.subtract(*cf.seed_from(source)), np.subtract(*cf.seed_from(target))
    inverse = np.linalg.pinv(background_covariance(target))
    plain = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    adjusted = float(a @ inverse @ b / np.sqrt((a @ inverse @ a) * (b @ inverse @ b)))
    return plain, adjusted


def borrowed(datasets: dict[str, list[cf.Participant]], base: dict[str, np.ndarray], local: dict[str, np.ndarray]) -> dict:
    pairs = {}
    for target_name, target in datasets.items():
        for source_name, source in datasets.items():
            if source_name == target_name:
                continue
            scores = np.array([cf.run_umm(p, memory=True, seed=cf.seed_from([q for q in source if q.name != p.name])).mean()
                               for p in target])
            plain, adjusted = similarity(source, target)
            pairs[f"{target_name} <- {source_name}"] = {
                "target": target_name, "source": source_name, **cf.summary(scores),
                "minus_umm_memory": round(float(scores.mean() - base[target_name].mean()), 4),
                "minus_local_seed": round(float(scores.mean() - local[target_name].mean()), 4),
                "travels": bool(scores.mean() > base[target_name].mean() and scores.mean() >= local[target_name].mean() - 0.05),
                "plain_cosine": round(plain, 4), "background_adjusted_cosine": round(adjusted, 4),
                "same_people": frozenset((target_name, source_name)) in SAME_PEOPLE,
            }
            print(f"  {target_name} <- {source_name}: {scores.mean():.3f} (no seed {base[target_name].mean():.3f}, local {local[target_name].mean():.3f})")
    usable = [v for v in pairs.values() if not v["same_people"]]
    gain = [v["minus_umm_memory"] for v in usable]
    rho_plain = spearmanr([v["plain_cosine"] for v in usable], gain)
    rho_adjusted = spearmanr([v["background_adjusted_cosine"] for v in usable], gain)
    same_lab = [v for v in usable if v["target"] in SAME_LAB and v["source"] in SAME_LAB]
    cross_lab = [v for v in usable if (v["target"] in SAME_LAB) != (v["source"] in SAME_LAB) and "healthy" not in v["target"] + v["source"]]
    return {
        "pairs": pairs,
        "what_makes_a_seed_travel": {
            "ordered_pairs": len(usable),
            "spearman_plain_cosine_vs_gain": {"rho": round(float(rho_plain.statistic), 3), "p": round(float(rho_plain.pvalue), 5)},
            "spearman_background_adjusted_vs_gain": {"rho": round(float(rho_adjusted.statistic), 3), "p": round(float(rho_adjusted.pvalue), 5)},
        },
        "same_lab_different_study": {"pairs": len(same_lab), "travelling": int(sum(v["travels"] for v in same_lab)),
                                     "mean_minus_local_seed": round(float(np.mean([v["minus_local_seed"] for v in same_lab])), 4) if same_lab else None,
                                     "mean_minus_umm_memory": round(float(np.mean([v["minus_umm_memory"] for v in same_lab])), 4) if same_lab else None},
        "different_lab_als_only": {"pairs": len(cross_lab), "travelling": int(sum(v["travels"] for v in cross_lab)),
                                   "mean_minus_local_seed": round(float(np.mean([v["minus_local_seed"] for v in cross_lab])), 4) if cross_lab else None,
                                   "mean_minus_umm_memory": round(float(np.mean([v["minus_umm_memory"] for v in cross_lab])), 4) if cross_lab else None},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / "expansion_study_results.json")
    args = parser.parse_args()
    new, problems = load_new(args.data_dir)
    datasets = {**cf.load_datasets(args.data_dir, "binned"), **new}
    print({name: len(people) for name, people in datasets.items()}, problems)
    base = {n: np.array([cf.run_umm(p, memory=True).mean() for p in people]) for n, people in datasets.items()}
    local = {n: np.array([cf.run_umm(p, memory=True, seed=cf.seed_from([q for q in people if q.name != p.name])).mean() for p in people])
             for n, people in datasets.items()}
    results = {"datasets": {n: {"participants": len(p), "umm_memory": cf.summary(base[n]), "seeded_umm": cf.summary(local[n])}
                            for n, p in datasets.items()},
               "left_out": problems}
    print("A: confirmation")
    results["confirmation"] = confirmation(datasets, base, local)
    if "etsu_b" in datasets:
        print("B: seed size (etsu_b)")
        results["seed_size"] = seed_size_curve(datasets["etsu_b"], base["etsu_b"])
    print("C-E: borrowed seeds")
    results["borrowed_seeds"] = borrowed(datasets, base, local)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
