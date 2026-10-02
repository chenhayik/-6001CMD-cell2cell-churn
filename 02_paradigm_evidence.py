"""Task 2 evidence: K-Means clusters vs churn, and fit time of each algorithm.
No train/test split here, since this only explores the data (no model is evaluated)."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import common as c


def build_table3_figure3_kmeans(df: pd.DataFrame) -> dict:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(df[c.NUMERIC_FEATURES]))
    y_true = (df[c.TARGET_COL] == "Yes").astype(int).to_numpy()

    rows = []
    labels_by_k = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=c.RANDOM_STATE)
        labels = km.fit_predict(x)
        labels_by_k[k] = labels
        sil = silhouette_score(x, labels, sample_size=min(c.SILHOUETTE_SAMPLE_SIZE, len(x)), random_state=c.RANDOM_STATE)
        rows.append({
            "k": k,
            "silhouette_10k_sample": sil,
            "inertia": km.inertia_,
            "adjusted_rand_index": adjusted_rand_score(y_true, labels),
            "normalized_mutual_info": normalized_mutual_info_score(y_true, labels),
        })
    table = pd.DataFrame(rows)
    c.save_table(
        table, 3, "kmeans_k_selection",
        "K-Means for k = 2 to 8 on standardised, median-imputed numeric features: silhouette (10,000-row "
        "sample), inertia, adjusted Rand index and normalised mutual information against Churn.",
        "Task 2",
    )

    best = table.loc[table["silhouette_10k_sample"].idxmax()]
    best_k = int(best["k"])
    labels = labels_by_k[best_k]
    overall_rate = y_true.mean()
    rate_per_cluster = pd.Series(y_true).groupby(labels).mean().sort_index()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(rate_per_cluster.index.astype(str), rate_per_cluster.values, color="C0")
    ax.axhline(overall_rate, color="black", linestyle="--", linewidth=1, label="Overall churn rate")
    ax.set_xlabel(f"Cluster (k = {best_k})")
    ax.set_ylabel("Churn rate")
    ax.set_title(f"Churn rate per cluster, best-silhouette k = {best_k}")
    ax.legend()
    c.save_fig(
        3, "churn_rate_per_cluster",
        f"Churn rate per cluster for the best-silhouette k ({best_k}), against the overall churn rate.",
        "Task 2",
    )

    return {
        "table3_best_silhouette_k": best_k,
        "table3_best_silhouette": float(best["silhouette_10k_sample"]),
        "table3_best_k_ari": float(best["adjusted_rand_index"]),
        "table3_best_k_nmi": float(best["normalized_mutual_info"]),
    }


# (name, model, time complexity, max rows or None). k = 4 is only for timing,
# not the k chosen in Table 3. n = rows, k = clusters, i = iterations, d = features.
_CLUSTERING_ALGORITHMS = {"KMeans", "MiniBatchKMeans", "AgglomerativeClustering"}
_ALGORITHMS = [
    ("KMeans", lambda: KMeans(n_clusters=4, n_init=10, random_state=c.RANDOM_STATE), "O(n k i d)", None),
    ("MiniBatchKMeans", lambda: MiniBatchKMeans(n_clusters=4, n_init=10, random_state=c.RANDOM_STATE), "O(b k i d), b = batch size", None),
    ("AgglomerativeClustering", lambda: AgglomerativeClustering(n_clusters=4), "O(n^2) memory", 10_000),
    ("LogisticRegression", lambda: LogisticRegression(max_iter=1000, random_state=c.RANDOM_STATE), "O(n d i)", None),
    ("RandomForest (100 trees)", lambda: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=c.RANDOM_STATE), "O(trees n log(n) d)", None),
    ("HistGradientBoosting", lambda: HistGradientBoostingClassifier(random_state=c.RANDOM_STATE), "O(rounds n d), histogram-based", None),
    ("SVC (RBF)", lambda: SVC(kernel="rbf", random_state=c.RANDOM_STATE), "O(n^2) to O(n^3)", 10_000),
]


def build_table4_figure4_scalability(df: pd.DataFrame) -> dict:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_all = scaler.fit_transform(imputer.fit_transform(df[c.NUMERIC_FEATURES]))
    y_all = (df[c.TARGET_COL] == "Yes").astype(int).to_numpy()
    n_available = len(x_all)

    sizes = [2_000, 5_000, 10_000, 20_000, 40_000]
    rng = np.random.RandomState(c.RANDOM_STATE)
    rows = []
    for name, factory, complexity, max_n in _ALGORITHMS:
        for n in sizes:
            if n > n_available:
                continue
            if max_n is not None and n > max_n:
                rows.append({
                    "algorithm": name, "n": n, "fit_time_seconds": np.nan,
                    "complexity": complexity, "note": f"not run: {complexity}",
                })
                continue
            idx = rng.choice(n_available, size=n, replace=False)
            x_sub = x_all[idx]
            model = factory()
            t0 = time.perf_counter()
            if name in _CLUSTERING_ALGORITHMS:
                model.fit(x_sub)
            else:
                model.fit(x_sub, y_all[idx])
            elapsed = time.perf_counter() - t0
            rows.append({"algorithm": name, "n": n, "fit_time_seconds": elapsed, "complexity": complexity, "note": ""})

    # KNN only stores data when fitting, so its prediction time is also recorded.
    for n in sizes:
        if n > n_available:
            continue
        idx = rng.choice(n_available, size=n, replace=False)
        x_sub, y_sub = x_all[idx], y_all[idx]
        knn = KNeighborsClassifier(n_neighbors=5)
        t0 = time.perf_counter()
        knn.fit(x_sub, y_sub)
        fit_t = time.perf_counter() - t0
        n_query = min(1_000, n)
        t0 = time.perf_counter()
        knn.predict(x_sub[:n_query])
        predict_t = time.perf_counter() - t0
        rows.append({
            "algorithm": "KNN (k=5)", "n": n, "fit_time_seconds": fit_t,
            "complexity": "fit O(1) lazy; predict O(n d) per query",
            "note": f"predict time for {n_query} queries: {predict_t:.4f}s",
        })

    table = pd.DataFrame(rows)
    c.save_table(
        table, 4, "fit_time_scalability",
        "Fit-time scalability across n, with textbook time complexity. AgglomerativeClustering and SVC "
        "(RBF) are capped at n = 10,000. KNN fit is near-instant; its prediction time is reported instead.",
        "Task 2",
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for algo, sub in table.groupby("algorithm", sort=False):
        sub = sub.dropna(subset=["fit_time_seconds"]).sort_values("n")
        if len(sub) == 0:
            continue
        ax.plot(sub["n"], sub["fit_time_seconds"], marker="o", label=algo)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n (rows, log scale)")
    ax.set_ylabel("Fit time (seconds, log scale)")
    ax.set_title("Fit-time scalability by algorithm")
    # legend outside the plot so it doesn't cover the lines
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    c.save_fig(4, "fit_time_scalability", "Fit time against n on log-log axes, by algorithm.", "Task 2")

    slowest = table.loc[table["fit_time_seconds"].idxmax()]
    return {
        "table4_slowest_algorithm": slowest["algorithm"],
        "table4_slowest_n": int(slowest["n"]),
        "table4_slowest_fit_time_seconds": float(slowest["fit_time_seconds"]),
    }


def main() -> None:
    args = c.make_arg_parser("Task 2 evidence: supervised vs unsupervised comparison.").parse_args()
    c.set_plot_style()
    t0 = time.time()

    df = c.load_train(quick=args.quick)

    numbers: dict = {}
    print("Task 2: Table 3 / Figure 3 K-Means vs Churn")
    numbers.update(build_table3_figure3_kmeans(df))
    print("Task 2: Table 4 / Figure 4 fit-time scalability")
    numbers.update(build_table4_figure4_scalability(df))

    numbers["script02_quick_mode"] = bool(args.quick)
    numbers["script02_runtime_seconds"] = round(time.time() - t0, 1)
    c.update_key_numbers(numbers)

    print(f"02_paradigm_evidence.py finished in {time.time() - t0:.1f}s (quick={args.quick})")


if __name__ == "__main__":
    main()
