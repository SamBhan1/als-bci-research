"""Run a reproducible participant-disjoint model benchmark and save JSON results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from advanced_experiment import bootstrap_ci, evaluate
from experiment import load_edf_data


def summarize(rows: list[dict[str, float]]) -> dict[str, object]:
    summary: dict[str, object] = {"participants": len(rows), "per_participant": rows}
    for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
        values = np.asarray([row[metric] for row in rows], dtype=float)
        low, high = bootstrap_ci(values, seed=401)
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "bootstrap_95_ci": [float(low), float(high)],
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args()

    data = load_edf_data(args.data_dir, args.max_files)
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "leave-one-participant-out",
        "epoch_averaging": args.repeats,
        "raw_epochs": int(len(data.y)),
        "participants": int(len(np.unique(data.subject))),
        "models": {},
    }
    for model in ("logistic", "xdawn_lda", "riemann"):
        print(f"Running {model}")
        result["models"][model] = summarize(evaluate(data, model, args.repeats))
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
