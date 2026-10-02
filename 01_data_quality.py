"""Task 1 and Task 3 evidence: dataset overview and data quality checks.
Run: python 01_data_quality.py [--quick]"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

import common as c

# --- Task 1 ---
def build_table1_overview(df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    n_rows, n_cols = df.shape
    grp = {g: len(cols) for g, cols in c.ALL_ANALYTICAL_GROUPS.items()}
    counts = df[c.TARGET_COL].value_counts()
    yes, no = int(counts.get("Yes", 0)), int(counts.get("No", 0))
    ratio = no / yes
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    holdout_churn_nonnull = int(holdout[c.TARGET_COL].notna().sum())
    overlap = len(set(df[c.ID_COL]) & set(holdout[c.ID_COL]))

    rows = [
        ("Training rows", f"{n_rows:,}"),
        ("Training columns (total)", f"{n_cols}"),
        ("  id columns", grp["id"]),
        ("  target columns", grp["target"]),
        ("  numeric continuous columns", grp["numeric_continuous"]),
        ("  numeric count/average columns", grp["numeric_count"]),
        ("  binary Yes/No columns", grp["binary_yesno"]),
        ("  nominal categorical columns", grp["nominal_categorical"]),
        ("  ordinal categorical columns", grp["ordinal_categorical"]),
        ("Target Churn = Yes", f"{yes:,} ({yes / n_rows:.1%})"),
        ("Target Churn = No", f"{no:,} ({no / n_rows:.1%})"),
        ("Target imbalance ratio, majority:minority", f"{ratio:.2f}:1"),
        ("Training memory usage, deep (MB)", f"{mem_mb:.1f}"),
        ("Holdout rows", f"{len(holdout):,}"),
        ("Holdout columns", f"{holdout.shape[1]}"),
        ("Holdout Churn non-null count", f"{holdout_churn_nonnull} (entirely empty, not usable for modelling)"),
        ("Train/holdout CustomerID overlap", f"{overlap} (no shared customers)"),
    ]
    table = pd.DataFrame(rows, columns=["metric", "value"])
    c.save_table(
        table, 1, "dataset_overview",
        "Dataset overview: rows, columns by analytical type, target balance, memory, holdout file facts.",
        "Task 1",
    )
    return {
        "table1_train_rows": n_rows, "table1_train_cols": n_cols,
        "table1_churn_yes": yes, "table1_churn_no": no,
        "table1_churn_yes_pct": yes / n_rows, "table1_churn_no_pct": no / n_rows,
        "table1_imbalance_ratio": ratio, "table1_memory_mb": mem_mb,
        "table1_holdout_rows": len(holdout), "table1_holdout_cols": int(holdout.shape[1]),
        "table1_holdout_churn_nonnull": holdout_churn_nonnull,
        "table1_holdout_train_id_overlap": overlap,
    }


def build_table2_dictionary(df: pd.DataFrame) -> dict:
    notes = c.build_feature_notes(df)
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        nunique = int(s.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(s):
            rng = f"{s.min():.2f} to {s.max():.2f}"
        else:
            rng = f"{nunique} levels"
        rows.append({
            "name": col,
            "pandas_dtype": str(s.dtype),
            "analytical_type": c.ANALYTICAL_TYPE[col],
            "description": c.FEATURE_DESCRIPTIONS.get(col, ""),
            "non_null_count": non_null,
            "unique_count": nunique,
            "range_or_levels": rng,
            "notes": notes.get(col, ""),
        })
    table = pd.DataFrame(rows)
    caption = (
        "Feature dictionary. Kaggle provides no data dictionary for this dataset: descriptions are "
        "interpretations from column names and value patterns, not official definitions."
    )
    c.save_table(table, 2, "feature_dictionary", caption, "Task 1")
    return {"table2_n_features_documented": len(table)}


def build_figure1_churn_by_quintile(df: pd.DataFrame) -> None:
    cols = ["DroppedCalls", "BlockedCalls", "CustomerCareCalls", "MonthsInService", "CurrentEquipmentDays"]
    overall_rate = (df[c.TARGET_COL] == "Yes").mean()
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 3.4), sharey=True)
    for i, (ax, col) in enumerate(zip(axes, cols)):
        sub = df[[col, c.TARGET_COL]].dropna()
        # rank first so the many tied zeros still split into 5 groups
        quintile = pd.qcut(sub[col].rank(method="first"), 5, labels=["1", "2", "3", "4", "5"])
        rate = sub.groupby(quintile, observed=True)[c.TARGET_COL].apply(lambda s: (s == "Yes").mean())
        ax.bar(rate.index, rate.values, color="C0")
        ax.axhline(overall_rate, color="black", linestyle="--", linewidth=1, label="Overall churn rate")
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Quintile (1 = lowest)")
        if i == 0:
            ax.set_ylabel("Churn rate")
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Churn rate by quintile of service-quality indicators, vs overall churn rate", y=1.03)
    fig.tight_layout()
    c.save_fig(
        1, "churn_rate_by_quintile",
        "Churn rate by quintile of DroppedCalls, BlockedCalls, CustomerCareCalls, MonthsInService and "
        "CurrentEquipmentDays, against the overall churn rate (dashed line).",
        "Task 1",
    )


def build_figure2_target_distribution(df: pd.DataFrame) -> None:
    counts = df[c.TARGET_COL].value_counts()
    pct = counts / counts.sum()
    ratio = counts.max() / counts.min()
    fig, ax = plt.subplots(figsize=(5, 4.2))
    bars = ax.bar(counts.index, counts.values, color=["C1", "C0"])
    for bar, cnt, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{cnt:,}\n({p:.1%})",
                 ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Number of customers")
    ax.set_xlabel("Churn")
    ax.set_title(f"Target class distribution (imbalance ratio {ratio:.2f}:1)")
    ax.margins(y=0.18)
    c.save_fig(
        2, "target_distribution",
        "Target class distribution with counts, percentages and the majority:minority imbalance ratio.",
        "Task 1",
    )


# --- Task 3 ---
def build_missingness_evidence(df: pd.DataFrame) -> dict:
    n = len(df)
    rows = []
    for col in df.columns:
        true_nan = int(df[col].isna().sum())
        rule = c.PLACEHOLDER_RULES.get(col)
        if rule and rule["treat_as_missing"]:
            placeholder_count = int((df[col] == rule["placeholder_value"]).sum())
            rule_text = rule["rule"]
        else:
            placeholder_count = 0
            rule_text = rule["rule"] if rule else "not applicable: no disguised-missing rule for this column"
        if true_nan == 0 and placeholder_count == 0:
            continue
        rows.append({
            "column": col,
            "true_nan_count": true_nan, "true_nan_pct": true_nan / n,
            "placeholder_count": placeholder_count, "placeholder_pct": placeholder_count / n,
            "combined_pct": (true_nan + placeholder_count) / n,
            "placeholder_rule": rule_text,
        })
    table = pd.DataFrame(rows).sort_values("combined_pct", ascending=False).reset_index(drop=True)
    c.save_table(
        table, 5, "missing_values",
        "Missing values per column: true NaN and disguised placeholders reported separately, plus the combined rate.",
        "Task 3",
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    y = np.arange(len(table))
    ax.barh(y - 0.2, table["true_nan_pct"] * 100, height=0.4, label="True NaN", color="C0")
    ax.barh(y + 0.2, table["placeholder_pct"] * 100, height=0.4, label="Disguised placeholder", color="C1")
    ax.set_yticks(y)
    ax.set_yticklabels(table["column"])
    ax.invert_yaxis()
    ax.set_xlabel("% of rows")
    ax.set_title("True vs disguised missing values by affected column")
    ax.legend(loc="lower right")
    c.save_fig(5, "missing_true_vs_disguised", "True vs disguised missing % per affected column.", "Task 3")

    true_nan_cols = [col for col in df.columns if df[col].isna().sum() > 0]
    indicator = df[true_nan_cols].isna().astype(int)
    corr = indicator.corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(corr, ax=ax, cmap="rocket_r", vmin=0, vmax=1, square=True,
                cbar_kws={"label": "correlation of missing-value indicators"})
    ax.set_title("Missingness structure: correlation of missing-value indicators")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    c.save_fig(
        6, "missingness_structure",
        "Correlation of missing-value indicators for columns with true NaN, showing the shared 156-row block.",
        "Task 3",
    )

    block_cols = ["MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
                  "DirectorAssistedCalls", "OverageMinutes", "RoamingCalls"]
    block_rows = int(df[block_cols].isna().all(axis=1).sum())
    numbers = {
        "table5_shared_missing_block_rows": block_rows,
        "table5_percchangeminutes_nan": int(df["PercChangeMinutes"].isna().sum()),
        "table5_percchangerevenues_nan": int(df["PercChangeRevenues"].isna().sum()),
        "table5_servicearea_nan": int(df["ServiceArea"].isna().sum()),
        "table5_handsetprice_unknown": int((df["HandsetPrice"] == "Unknown").sum()),
        "table5_maritalstatus_unknown": int((df["MaritalStatus"] == "Unknown").sum()),
        "table5_agehh1_zero": int((df["AgeHH1"] == 0).sum()),
        "table5_agehh2_zero": int((df["AgeHH2"] == 0).sum()),
        "table5_incomegroup_zero": int((df["IncomeGroup"] == 0).sum()),
        "table5_n_affected_columns": int(len(table)),
    }
    return numbers


def build_table6_duplicates(df: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    exact = int(df.duplicated().sum())
    ignoring_id = int(df.drop(columns=[c.ID_COL]).duplicated().sum())
    ignoring_id_target = int(df.drop(columns=[c.ID_COL, c.TARGET_COL]).duplicated().sum())
    id_dup = int(df[c.ID_COL].duplicated().sum())
    overlap = len(set(df[c.ID_COL]) & set(holdout[c.ID_COL]))

    rows = [
        ("Exact duplicate rows (all columns)", exact),
        ("Duplicate rows ignoring CustomerID", ignoring_id),
        ("Duplicate rows on all features (ignoring CustomerID and Churn)", ignoring_id_target),
        ("CustomerID unique", "Yes" if df[c.ID_COL].is_unique else "No"),
        ("CustomerID duplicate count", id_dup),
        ("Train/holdout CustomerID overlap", overlap),
    ]
    table = pd.DataFrame(rows, columns=["check", "result"])
    c.save_table(
        table, 6, "duplicate_checks",
        "Duplicate-row and identifier-uniqueness checks, training set and holdout.",
        "Task 3",
    )
    return {
        "table6_exact_duplicates": exact, "table6_duplicates_ignoring_id": ignoring_id,
        "table6_duplicates_ignoring_id_target": ignoring_id_target,
        "table6_customerid_duplicates": id_dup, "table6_train_holdout_overlap": overlap,
    }


def build_table7_figure7_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    rows = []
    for col in c.NUMERIC_FEATURES:
        s = df[col].dropna()
        desc = s.describe()
        rows.append({
            "feature": col, "count": int(desc["count"]), "mean": desc["mean"], "std": desc["std"],
            "min": desc["min"], "25%": desc["25%"], "50%": desc["50%"], "75%": desc["75%"], "max": desc["max"],
            "skewness": s.skew(), "excess_kurtosis_fisher": s.kurtosis(),
        })
    table = pd.DataFrame(rows)
    table["_abs_skew"] = table["skewness"].abs()
    table = table.sort_values("_abs_skew", ascending=False).drop(columns="_abs_skew").reset_index(drop=True)
    c.save_table(
        table, 7, "numeric_summary_stats",
        "Summary statistics for numeric features, sorted by absolute skewness, with excess kurtosis (Fisher).",
        "Task 3",
    )

    top8 = table["feature"].head(8).tolist()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.ravel(), top8):
        s = df[col].dropna()
        ax.hist(s, bins=40, color="C0")
        ax.set_title(f"{col} (skew={s.skew():.2f})", fontsize=9)
        ax.set_xlabel(col, fontsize=8)
    fig.suptitle("Distributions of the 8 most skewed numeric features", y=1.02)
    fig.tight_layout()
    c.save_fig(7, "skewed_histograms", "Histograms of the 8 most skewed numeric features, with skew annotated.", "Task 3")

    numbers = {
        "table7_top_skew_feature": top8[0],
        "table7_top_skew_value": float(table.loc[table["feature"] == top8[0], "skewness"].iloc[0]),
    }
    return table, top8, numbers


def build_table8_figure8_outliers(df: pd.DataFrame, top8: list[str]) -> dict:
    rows = []
    for col in c.NUMERIC_FEATURES:
        s = df[col].dropna()
        lo, hi = c.iqr_bounds(s)
        iqr_mask = (s < lo) | (s > hi)
        z = (s - s.mean()) / s.std(ddof=0)
        z_mask = z.abs() > 3
        rows.append({
            "feature": col,
            "iqr_outlier_count": int(iqr_mask.sum()), "iqr_outlier_pct": float(iqr_mask.mean()),
            "z_outlier_count": int(z_mask.sum()), "z_outlier_pct": float(z_mask.mean()),
            "min": s.min(), "max": s.max(),
        })
    table = pd.DataFrame(rows).sort_values("iqr_outlier_pct", ascending=False).reset_index(drop=True)
    c.save_table(
        table, 8, "outliers",
        "Outliers per numeric feature: IQR-rule count/%, |z| > 3 count/%, and observed min/max.",
        "Task 3",
    )

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.ravel(), top8):
        s = df[col].dropna()
        ax.boxplot(s, orientation="vertical", tick_labels=[col])
        if (s.max() - s.min()) > 50:
            ax.set_yscale("symlog")
        ax.set_title(col, fontsize=9)
        ax.set_xticklabels([])
    fig.suptitle("Boxplots of the 8 most skewed numeric features (symlog axis where the range is wide)", y=1.02)
    fig.tight_layout()
    c.save_fig(8, "boxplots", "Boxplots of the 8 most skewed numeric features.", "Task 3")

    worst = table.iloc[0]
    return {"table8_worst_iqr_feature": worst["feature"], "table8_worst_iqr_pct": float(worst["iqr_outlier_pct"])}


def build_figure9_table9_correlation(df: pd.DataFrame) -> dict:
    num = df[c.NUMERIC_FEATURES]
    pearson = num.corr(method="pearson")
    spearman = num.corr(method="spearman")

    dist_arr = (1 - pearson.abs()).to_numpy(copy=True)
    np.fill_diagonal(dist_arr, 0.0)
    condensed = squareform(dist_arr, checks=False)
    z = linkage(condensed, method="average")
    order = dendrogram(z, no_plot=True)["leaves"]
    ordered_cols = [pearson.columns[i] for i in order]
    pearson_ordered = pearson.loc[ordered_cols, ordered_cols]

    fig, ax = plt.subplots(figsize=(11, 9.5))
    sns.heatmap(pearson_ordered, ax=ax, cmap="vlag", vmin=-1, vmax=1, square=True,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Pearson correlation, cluster-ordered (see Table 9 for pairs with |r| >= 0.7)")
    ax.tick_params(axis="both", labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)
    c.save_fig(
        9, "correlation_heatmap",
        "Pearson correlation heatmap for numeric features, cluster-ordered for readability.",
        "Task 3",
    )

    cols = list(num.columns)
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pr, sr = pearson.iloc[i, j], spearman.iloc[i, j]
            if abs(pr) >= 0.7 or abs(sr) >= 0.7:
                pairs.append({"feature_1": cols[i], "feature_2": cols[j], "pearson_r": pr, "spearman_r": sr})
    table9 = pd.DataFrame(pairs)
    if len(table9):
        table9 = table9.sort_values("pearson_r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    c.save_table(
        table9, 9, "high_correlation_pairs",
        "Numeric feature pairs with |r| >= 0.7 under Pearson and/or Spearman correlation.",
        "Task 3",
    )
    return {"table9_n_high_corr_pairs": int(len(table9))}


def build_figure10_table10_associations(df: pd.DataFrame) -> dict:
    cat_cols = c.CATEGORICAL_FEATURES + [c.TARGET_COL]
    m = len(cat_cols)
    v = pd.DataFrame(np.eye(m), index=cat_cols, columns=cat_cols)
    for i in range(m):
        for j in range(i + 1, m):
            val = c.cramers_v(df[cat_cols[i]], df[cat_cols[j]])
            v.iloc[i, j] = v.iloc[j, i] = val

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    sns.heatmap(v, ax=ax, cmap="rocket_r", vmin=0, vmax=1, square=True,
                cbar_kws={"label": "Cramer's V (bias-corrected)"})
    ax.set_title("Association between categorical features and Churn (Cramer's V)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.tick_params(labelsize=7)
    c.save_fig(
        10, "cramers_v_heatmap",
        "Bias-corrected Cramer's V heatmap for categorical features, including Churn.",
        "Task 3",
    )

    y = (df[c.TARGET_COL] == "Yes").astype(int)
    rows = []
    for col in c.CATEGORICAL_FEATURES:
        rows.append({"feature": col, "type": "categorical", "metric": "Cramer's V", "value": c.cramers_v(df[col], df[c.TARGET_COL])})
    for col in c.NUMERIC_FEATURES:
        sub = pd.DataFrame({"x": df[col], "y": y}).dropna()
        r, _p = stats.pointbiserialr(sub["y"], sub["x"])
        rows.append({"feature": col, "type": "numeric", "metric": "point-biserial r", "value": r})
    table10 = pd.DataFrame(rows)
    table10["_abs"] = table10["value"].abs()
    table10 = table10.sort_values("_abs", ascending=False).drop(columns="_abs").reset_index(drop=True)
    table10.insert(0, "rank", np.arange(1, len(table10) + 1))
    c.save_table(
        table10, 10, "association_with_churn",
        "Association of each feature with Churn: Cramer's V for categoricals, point-biserial r for numerics, ranked.",
        "Task 3",
    )
    top = table10.iloc[0]
    return {"table10_top_feature": top["feature"], "table10_top_value": float(top["value"])}


def build_table11_noise(df: pd.DataFrame) -> dict:
    n = len(df)
    rows = []

    # 1. Impossible negative values
    neg = {col: int((df[col] < 0).sum()) for col in ["MonthlyRevenue", "TotalRecurringCharge", "CurrentEquipmentDays"]}
    total_neg_rows = int(((df["MonthlyRevenue"] < 0) | (df["TotalRecurringCharge"] < 0) | (df["CurrentEquipmentDays"] < 0)).sum())
    rows.append({
        "issue": "Impossible negative values",
        "columns_affected": "MonthlyRevenue, TotalRecurringCharge, CurrentEquipmentDays",
        "rows_affected": "; ".join(f"{k}: {v:,}" for k, v in neg.items()),
        "pct_of_rows": f"{total_neg_rows / n:.2%} (rows with at least one)",
        "example_values": f"mins: MonthlyRevenue {df['MonthlyRevenue'].min():.2f}, "
                           f"TotalRecurringCharge {df['TotalRecurringCharge'].min():.2f}, "
                           f"CurrentEquipmentDays {df['CurrentEquipmentDays'].min():.0f}",
    })

    # 2. Decimal values in count columns (they are monthly averages)
    frac_share = {col: float((df[col].dropna() % 1 != 0).mean()) for col in c.NUMERIC_COUNT}
    top_frac = sorted(frac_share.items(), key=lambda kv: -kv[1])[:5]
    dropped_calls_frac_n = int((df["DroppedCalls"].dropna() % 1 != 0).sum())
    rows.append({
        "issue": "Fractional values in count-like columns (monthly averages, not raw counts)",
        "columns_affected": "all 22 numeric count/average columns",
        "rows_affected": f"DroppedCalls: {dropped_calls_frac_n:,}",
        "pct_of_rows": f"{frac_share['DroppedCalls']:.1%} (DroppedCalls)",
        "example_values": "; ".join(f"{k} {v:.1%}" for k, v in top_frac),
    })

    # 3. NewCellphoneUser and NotNewCellphoneUser overlap
    ct = pd.crosstab(df["NewCellphoneUser"], df["NotNewCellphoneUser"])
    both_no = int(ct.loc["No", "No"])
    rows.append({
        "issue": "NewCellphoneUser / NotNewCellphoneUser redundancy: both No means status unknown",
        "columns_affected": "NewCellphoneUser, NotNewCellphoneUser",
        "rows_affected": f"{both_no:,}",
        "pct_of_rows": f"{both_no / n:.1%}",
        "example_values": f"No/No={ct.loc['No','No']}, No/Yes={ct.loc['No','Yes']}, "
                           f"Yes/No={ct.loc['Yes','No']}, Yes/Yes={ct.loc['Yes','Yes']}",
    })

    # 4. DroppedBlockedCalls inconsistency
    valid = df["DroppedBlockedCalls"].notna() & df["DroppedCalls"].notna() & df["BlockedCalls"].notna()
    d = df.loc[valid, "DroppedBlockedCalls"]
    s_sum = df.loc[valid, "DroppedCalls"] + df.loc[valid, "BlockedCalls"]
    mismatch = ~np.isclose(d, s_sum)
    diff = (d[mismatch] - s_sum[mismatch]).abs()
    within = int((diff <= 0.1).sum())
    larger = int((diff > 0.1).sum())
    mismatch_n = int(mismatch.sum())
    rows.append({
        "issue": "DroppedBlockedCalls != DroppedCalls + BlockedCalls",
        "columns_affected": "DroppedBlockedCalls, DroppedCalls, BlockedCalls",
        "rows_affected": f"{mismatch_n:,}",
        "pct_of_rows": f"{mismatch_n / n:.2%}",
        "example_values": f"of {mismatch_n:,} mismatches: {within:,} within 0.1 (rounding of 1-decimal averages), {larger:,} larger",
    })

    # 5. Homeownership only says Known/Unknown
    unk_home = int((df["Homeownership"] == "Unknown").sum())
    rows.append({
        "issue": "Homeownership records whether ownership status is known, not ownership itself",
        "columns_affected": "Homeownership",
        "rows_affected": f"{unk_home:,}",
        "pct_of_rows": f"{unk_home / n:.2%}",
        "example_values": "only takes values Known/Unknown, never a Yes/No ownership answer",
    })

    # 6. HandsetPrice mixes text and numbers
    unk_hp = int((df["HandsetPrice"] == "Unknown").sum())
    bands = sorted(pd.to_numeric(df.loc[df["HandsetPrice"] != "Unknown", "HandsetPrice"], errors="coerce").dropna().unique())
    rows.append({
        "issue": "HandsetPrice mixes a text placeholder with a numeric price band",
        "columns_affected": "HandsetPrice",
        "rows_affected": f"{unk_hp:,}",
        "pct_of_rows": f"{unk_hp / n:.2%}",
        "example_values": f"numeric bands seen (USD): {', '.join(str(int(b)) for b in bands)}",
    })

    # 7. AgeHH zeros
    a1 = int((df["AgeHH1"] == 0).sum())
    a2 = int((df["AgeHH2"] == 0).sum())
    rows.append({
        "issue": "AgeHH1 / AgeHH2 zero is not a plausible age",
        "columns_affected": "AgeHH1, AgeHH2",
        "rows_affected": f"AgeHH1: {a1:,}; AgeHH2: {a2:,}",
        "pct_of_rows": f"AgeHH1: {a1 / n:.1%}; AgeHH2: {a2 / n:.1%}",
        "example_values": "AgeHH2 = 0 is ambiguous: may mean no second adult in the household, not a missing age",
    })

    # 8. IncomeGroup zeros
    ig0 = int((df["IncomeGroup"] == 0).sum())
    rows.append({
        "issue": "IncomeGroup 0 appears to mean unknown income",
        "columns_affected": "IncomeGroup",
        "rows_affected": f"{ig0:,}",
        "pct_of_rows": f"{ig0 / n:.2%}",
        "example_values": "bands otherwise run 1 (lowest) to 9 (highest)",
    })

    # 9. ServiceArea code structure
    sa = df["ServiceArea"].dropna()
    n_levels = int(sa.nunique())
    n_markets = int(sa.str.slice(0, 3).nunique())
    rows.append({
        "issue": "ServiceArea code encodes a market, city and telephone area code",
        "columns_affected": "ServiceArea",
        "rows_affected": f"{len(sa):,}",
        "pct_of_rows": f"{len(sa) / n:.2%}",
        "example_values": f"e.g. SEABLV425 = market SEA + city BLV + area code 425; "
                           f"{n_levels} raw levels reduce to {n_markets} markets",
    })

    table = pd.DataFrame(rows)
    c.save_table(
        table, 11, "noise_and_inconsistency",
        "Noise and inconsistency evidence, one row per issue.",
        "Task 3",
    )
    return {
        "table11_dropped_blocked_mismatch_count": mismatch_n,
        "table11_dropped_blocked_mismatch_pct": mismatch_n / n,
        "table11_dropped_blocked_within_0_1": within,
        "table11_dropped_blocked_larger": larger,
        "table11_newcellphone_both_no": both_no,
        "table11_servicearea_raw_levels": n_levels,
        "table11_servicearea_markets": n_markets,
        "table11_handsetprice_unknown": unk_hp,
    }


def build_figure11_servicearea(df: pd.DataFrame) -> None:
    sa = df["ServiceArea"].dropna()
    raw_freq = sa.value_counts().sort_values(ascending=False)
    market_freq = sa.str.slice(0, 3).value_counts().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(np.arange(1, len(raw_freq) + 1), raw_freq.values, marker=".", linestyle="none", color="C0")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Rank (log scale)")
    axes[0].set_ylabel("Customers (log scale)")
    axes[0].set_title(f"Raw ServiceArea codes ({len(raw_freq)} levels)")

    axes[1].plot(np.arange(1, len(market_freq) + 1), market_freq.values, marker=".", linestyle="none", color="C1")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Rank (log scale)")
    axes[1].set_ylabel("Customers (log scale)")
    axes[1].set_title(f"Extracted markets ({len(market_freq)} levels)")

    fig.suptitle("ServiceArea rank-frequency: long tail of raw codes vs extracted markets", y=1.03)
    fig.tight_layout()
    c.save_fig(
        11, "servicearea_rank_frequency",
        "ServiceArea rank-frequency (long tail), raw codes vs the 57 extracted markets.",
        "Task 3",
    )


# --- Run everything ---
def main() -> None:
    args = c.make_arg_parser("Task 1 and Task 3 evidence: data quality analysis.").parse_args()
    c.set_plot_style()
    t0 = time.time()

    df = c.load_train(quick=args.quick)
    holdout = c.load_holdout()
    c.validate_column_groups(df)

    numbers: dict = {}

    print("Task 1: Table 1 dataset overview")
    numbers.update(build_table1_overview(df, holdout))
    print("Task 1: Table 2 feature dictionary")
    numbers.update(build_table2_dictionary(df))
    print("Task 1: Figure 1 churn rate by quintile")
    build_figure1_churn_by_quintile(df)
    print("Task 1: Figure 2 target class distribution")
    build_figure2_target_distribution(df)

    print("Task 3: Table 5 / Figure 5 / Figure 6 missingness")
    numbers.update(build_missingness_evidence(df))
    print("Task 3: Table 6 duplicate checks")
    numbers.update(build_table6_duplicates(df, holdout))
    print("Task 3: Table 7 / Figure 7 summary statistics")
    _table7, top8, n7 = build_table7_figure7_stats(df)
    numbers.update(n7)
    print("Task 3: Table 8 / Figure 8 outliers")
    numbers.update(build_table8_figure8_outliers(df, top8))
    print("Task 3: Figure 9 / Table 9 Pearson correlation")
    numbers.update(build_figure9_table9_correlation(df))
    print("Task 3: Figure 10 / Table 10 associations with Churn")
    numbers.update(build_figure10_table10_associations(df))
    print("Task 3: Table 11 noise and inconsistency")
    numbers.update(build_table11_noise(df))
    print("Task 3: Figure 11 ServiceArea rank-frequency")
    build_figure11_servicearea(df)

    numbers["script01_quick_mode"] = bool(args.quick)
    numbers["script01_runtime_seconds"] = round(time.time() - t0, 1)
    c.update_key_numbers(numbers)

    print(f"01_data_quality.py finished in {time.time() - t0:.1f}s (quick={args.quick})")


if __name__ == "__main__":
    main()
