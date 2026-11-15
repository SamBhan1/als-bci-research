"""Research-grade P300 comparison: aggregation, xDAWN, shrinkage LDA, and LOSO."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from experiment import Dataset, features, load_edf_data


def aggregate(data: Dataset, repeats: int, seed: int = 19) -> Dataset:
    """Average same-class epochs within each participant, without crossing groups."""
    if repeats == 1:
        return data
    rng = np.random.default_rng(seed)
    x_out, y_out, s_out = [], [], []
    for subject in np.unique(data.subject):
        for label in (0, 1):
            indices = np.flatnonzero((data.subject == subject) & (data.y == label))
            rng.shuffle(indices)
            usable = len(indices) - (len(indices) % repeats)
            for group in np.split(indices[:usable], usable // repeats):
                x_out.append(data.x[group].mean(axis=0))
                y_out.append(label)
                s_out.append(subject)
    return Dataset(np.asarray(x_out), np.asarray(y_out), np.asarray(s_out))


def balanced_train_indices(y: np.ndarray, seed: int = 31) -> np.ndarray:
    """Balance only the training partition; never alter the held-out test partition."""
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    n = min(len(class_indices[0]), len(class_indices[1]))
    return np.concatenate([rng.choice(indices, n, replace=False) for indices in class_indices])


def p300_window(x: np.ndarray) -> np.ndarray:
    """Use the preregistered 250--500 ms post-stimulus interval, downsampled 4x."""
    start, end = int(0.45 * 256), int(0.70 * 256)  # epochs start 200 ms before stimulus
    return x[:, :, start:end:4]


def score(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "roc_auc": float(roc_auc_score(y, proba)),
    }


def evaluate(data: Dataset, model_name: str, repeats: int) -> list[dict[str, float]]:
    """Evaluate one model with leave-one-participant-out splits."""
    rows = []
    for subject in np.unique(data.subject):
        train = data.subject != subject
        test = ~train
        train_data = aggregate(Dataset(data.x[train], data.y[train], data.subject[train]), repeats, seed=42)
        test_data = aggregate(Dataset(data.x[test], data.y[test], data.subject[test]), repeats, seed=42)
        keep = balanced_train_indices(train_data.y, seed=17)
        x_train = p300_window(train_data.x[keep])
        x_test = p300_window(test_data.x)

        if model_name == "xdawn_lda":
            from mne.decoding import XdawnTransformer

            spatial = XdawnTransformer(n_components=4, reg="oas")
            train_transformed = spatial.fit_transform(x_train, train_data.y[keep])
            test_transformed = spatial.transform(x_test)
            train_features = train_transformed.reshape(len(train_transformed), -1)
            test_features = test_transformed.reshape(len(test_transformed), -1)
            model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        elif model_name == "riemann":
            from pyriemann.estimation import XdawnCovariances
            from pyriemann.tangentspace import TangentSpace

            covariances = XdawnCovariances(nfilter=4, estimator="oas", xdawn_estimator="oas")
            train_cov = covariances.fit_transform(x_train, train_data.y[keep])
            test_cov = covariances.transform(x_test)
            tangent = TangentSpace(metric="riemann")
            train_features = tangent.fit_transform(train_cov)
            test_features = tangent.transform(test_cov)
            model = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs")
        else:
            train_features = features(train_data.x[keep])
            test_features = features(test_data.x)
            model = LogisticRegression(max_iter=2500, class_weight="balanced", solver="lbfgs")

        model.fit(train_features, train_data.y[keep])
        pred = model.predict(test_features)
        proba = model.predict_proba(test_features)[:, 1]
        row = score(test_data.y, pred, proba)
        row["subject"] = subject
        rows.append(row)
    return rows


def bootstrap_ci(values: np.ndarray, seed: int = 77, draws: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def report(name: str, rows: list[dict[str, float]]) -> None:
    print(f"\n{name}")
    for metric in ("balanced_accuracy", "macro_f1", "roc_auc"):
        values = np.asarray([row[metric] for row in rows])
        low, high = bootstrap_ci(values)
        print(f"  {metric}: mean={values.mean():.3f}, std={values.std(ddof=1):.3f}, 95% CI=[{low:.3f}, {high:.3f}]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for memory-safe experiments")
    args = parser.parse_args()

    data = load_edf_data(args.data_dir, args.max_files)
    print(f"Real EEG epochs: {len(data.y)} | participants: {len(np.unique(data.subject))} | epoch averaging: {args.repeats}")
    logistic = evaluate(data, "logistic", args.repeats)
    xdawn = evaluate(data, "xdawn_lda", args.repeats)
    riemann = evaluate(data, "riemann", args.repeats)
    report("Time-resolved logistic baseline", logistic)
    report("xDAWN + shrinkage LDA", xdawn)
    report("xDAWN covariance + Riemannian tangent space", riemann)
    for left_name, left, right_name, right in (
        ("logistic", logistic, "xDAWN+LDA", xdawn),
        ("xDAWN+LDA", xdawn, "Riemannian", riemann),
    ):
        paired = wilcoxon(
            [row["balanced_accuracy"] for row in left],
            [row["balanced_accuracy"] for row in right],
            alternative="two-sided",
            method="auto",
        )
        print(f"\nPaired Wilcoxon {left_name} vs {right_name}: statistic={paired.statistic:.3f}, p={paired.pvalue:.4f}")
    print("Interpretation: participant-level results and uncertainty matter more than a single pooled accuracy.")


if __name__ == "__main__":
    main()
