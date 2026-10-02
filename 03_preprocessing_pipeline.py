"""Task 4 evidence: preprocessing, resampling and leakage tests (Tables 12 to 18, Figures 12 to 17).
The 80/20 split happens first and everything is fitted on training data only, except the Table 16 leakage demo."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, make_scorer, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer, StandardScaler, TargetEncoder

import common as c

# --- Cleaning and column groups ---
def pre_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row cleaning done before the split (placeholders, Yes/No to 1/0, market).
    It never uses statistics across rows, so it cannot leak."""
    out = c.clean_placeholders(df)
    for col in c.BINARY_YESNO:
        out[col] = out[col].map({"Yes": 1.0, "No": 0.0})
    out["ServiceAreaMarket"] = c.extract_service_area_market(out["ServiceArea"])
    out = out.drop(columns=["ServiceArea", c.ID_COL])
    return out


def build_feature_lists() -> tuple[list[str], list[str], list[str], list[str]]:
    """Group the columns (numeric, binary, ordinal, nominal) after pre_clean().
    Binary 0/1 flags are imputed but not power-transformed."""
    numeric_cols = c.NUMERIC_CONTINUOUS + c.NUMERIC_COUNT + ["IncomeGroup", "HandsetPrice"]
    binary_cols = list(c.BINARY_YESNO)
    ordinal_cols = ["CreditRating"]
    nominal_cols = ["ServiceAreaMarket", "PrizmCode", "Occupation", "Homeownership", "MaritalStatus"]
    return numeric_cols, binary_cols, ordinal_cols, nominal_cols


# --- Table 12 / Figure 12: skew before and after transforms ---
def build_table12_figure12(x_train: pd.DataFrame) -> dict:
    skews = {col: x_train[col].dropna().skew() for col in c.NUMERIC_FEATURES}
    ranked = sorted(skews.items(), key=lambda kv: -abs(kv[1]))
    top_features = [f for f, _ in ranked[:6]]

    # The 6 most skewed columns all contain 0, so Box-Cox can't run on them.
    # Add the 2 most skewed positive columns so Box-Cox has results too.
    positive_extra: list[str] = []
    for feature, _skew in ranked:
        if feature in top_features:
            continue
        if x_train[feature].dropna().min() > 0:
            positive_extra.append(feature)
        if len(positive_extra) == 2:
            break
    all_features = top_features + positive_extra

    rows = []
    yj_cache = {}
    for col in all_features:
        s = x_train[col].dropna()
        raw_skew = s.skew()

        # log1p needs x > -1 for every value
        if (s > -1).all():
            log1p_skew = f"{np.log1p(s).skew():.3f}"
        else:
            log1p_skew = "not applicable: values <= -1"

        # Box-Cox needs strictly positive values
        if (s > 0).all():
            bc, _ = stats.boxcox(s.to_numpy())
            boxcox_skew = f"{pd.Series(bc).skew():.3f}"
        else:
            boxcox_skew = "not applicable: non-positive values"

        # Yeo-Johnson is defined for any real value
        yj, _ = stats.yeojohnson(s.to_numpy())
        yj_series = pd.Series(yj, index=s.index)
        yj_cache[col] = {"raw": s, "yj": yj_series}

        rows.append({
            "feature": col, "skew_raw": raw_skew, "skew_log1p": log1p_skew,
            "skew_boxcox": boxcox_skew, "skew_yeojohnson": yj_series.skew(),
        })
    table = pd.DataFrame(rows)
    c.save_table(
        table, 12, "skew_before_after",
        "Skewness before and after log1p, Box-Cox (where strictly positive) and Yeo-Johnson, for the 6 most "
        "skewed numeric features plus the 2 most skewed strictly positive numeric features (added so Box-Cox "
        "has real values to report), fit on the training split only.",
        "Task 4",
    )

    fig3 = top_features[:3]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for j, col in enumerate(fig3):
        axes[0, j].hist(yj_cache[col]["raw"], bins=40, color="C0")
        axes[0, j].set_title(f"{col} (raw)", fontsize=9)
        axes[1, j].hist(yj_cache[col]["yj"], bins=40, color="C1")
        axes[1, j].set_title(f"{col} (Yeo-Johnson)", fontsize=9)
    fig.suptitle("Before and after Yeo-Johnson transform", y=1.02)
    fig.tight_layout()
    c.save_fig(12, "before_after_histograms", "Before and after histograms for 3 of the most skewed features (Yeo-Johnson transform).", "Task 4")

    return {
        "table12_most_skewed_feature": top_features[0],
        "table12_most_skewed_raw_skew": float(skews[top_features[0]]),
        "table12_positive_extra_features": positive_extra,
    }


def build_figure17_typical_skew(x_train: pd.DataFrame) -> dict:
    """Figure 17: Yeo-Johnson on typical skewed columns that are not mostly zeros.
    Picks columns with under 5% zeros and 50+ distinct values, fitted on training data only."""
    candidates = []
    for col in c.NUMERIC_FEATURES:
        s = x_train[col].dropna()
        if (s == 0).mean() < 0.05 and s.nunique() >= 50:
            candidates.append((col, s.skew()))
    chosen = [col for col, _ in sorted(candidates, key=lambda kv: -abs(kv[1]))[:3]]

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    numbers = {}
    for j, col in enumerate(chosen):
        s = x_train[col].dropna()
        yj, _ = stats.yeojohnson(s.to_numpy())
        yj = pd.Series(yj)
        axes[0, j].hist(s, bins=40, color="C0")
        axes[0, j].set_title(f"{col} (raw, skew = {s.skew():.2f})", fontsize=9)
        axes[1, j].hist(yj, bins=40, color="C1")
        axes[1, j].set_title(f"{col} (Yeo-Johnson, skew = {yj.skew():.2f})", fontsize=9)
        axes[1, j].set_xlabel("transformed value")
        numbers[f"figure17_{col}_skew_raw"] = float(s.skew())
        numbers[f"figure17_{col}_skew_yeojohnson"] = float(yj.skew())
    axes[0, 0].set_ylabel("customers")
    axes[1, 0].set_ylabel("customers")
    fig.suptitle("Before and after Yeo-Johnson: skewed columns that are not zero-inflated", y=1.02)
    fig.tight_layout()
    c.save_fig(
        17, "before_after_typical_skew",
        "Before and after Yeo-Johnson for the three most skewed continuous columns with fewer than 5% zeros "
        "(training split only). Contrast with Figure 12, where zero-inflated columns barely change.",
        "Task 4",
    )
    numbers["figure17_features"] = chosen
    return numbers


# --- Table 13 / Figure 13: PCA on usage and call columns ---
def build_table13_figure13_pca(x_train: pd.DataFrame, numeric_cols: list[str]) -> dict:
    # Usage and call columns only. Age, handset age and income describe the customer, not usage.
    usage_call_cols = [col for col in numeric_cols if col in (c.NUMERIC_CONTINUOUS + c.NUMERIC_COUNT)
                        and col not in ("AgeHH1", "AgeHH2", "CurrentEquipmentDays")]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(x_train[usage_call_cols]))

    pca = PCA(n_components=None, random_state=c.RANDOM_STATE)
    pca.fit(x)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    table = pd.DataFrame({
        "component": np.arange(1, len(evr) + 1),
        "explained_variance_ratio": evr,
        "cumulative_explained_variance": cum,
    })
    n80 = int(np.searchsorted(cum, 0.80) + 1)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    caption = (
        f"PCA on {len(usage_call_cols)} standardised numeric usage/call features, fit on the training split "
        f"only: {n80} components reach 80% cumulative explained variance, {n90} reach 90%, {n95} reach 95%."
    )
    c.save_table(table, 13, "pca_explained_variance", caption, "Task 4")

    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    ax1.bar(table["component"], table["explained_variance_ratio"], color="C0", label="Per-component")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax2 = ax1.twinx()
    ax2.plot(table["component"], table["cumulative_explained_variance"], color="C1", marker="o", label="Cumulative")
    for level in (0.80, 0.90, 0.95):
        ax2.axhline(level, color="grey", linestyle=":", linewidth=1)
    ax2.set_ylabel("Cumulative explained variance")
    ax1.set_title("PCA scree plot: usage and call-volume features")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    c.save_fig(13, "pca_scree", "Scree plot with cumulative explained variance; dotted lines mark 80/90/95%.", "Task 4")

    return {
        "table13_n_input_features": len(usage_call_cols),
        "table13_n_components_80pct": n80,
        "table13_n_components_90pct": n90,
        "table13_n_components_95pct": n95,
    }


# --- Table 14: encoding options ---
def build_table14(x_train: pd.DataFrame, y_train: pd.Series, service_area_raw_train: pd.Series) -> dict:
    def te_columns(frame: pd.DataFrame) -> int:
        te = TargetEncoder(target_type="binary", cv=5)  # cross-fitting, so a row is never encoded using its own target
        te.fit(frame, y_train)
        return int(te.transform(frame).shape[1])

    other_cats = ["PrizmCode", "Occupation", "Homeownership", "MaritalStatus", "CreditRating"]
    rows = []

    raw_levels = int(service_area_raw_train.nunique(dropna=True))
    raw_te = te_columns(service_area_raw_train.to_frame(name="ServiceArea"))
    rows.append({"feature": "ServiceArea (raw)", "distinct_levels": raw_levels,
                 "onehot_columns": raw_levels, "frequency_encoding_columns": 1, "target_encoder_columns": raw_te})

    market_levels = int(x_train["ServiceAreaMarket"].nunique(dropna=True))
    market_te = te_columns(x_train[["ServiceAreaMarket"]])
    rows.append({"feature": "ServiceArea (extracted market)", "distinct_levels": market_levels,
                 "onehot_columns": market_levels, "frequency_encoding_columns": 1, "target_encoder_columns": market_te})

    other_levels_sum = 0
    for col in other_cats:
        levels = int(x_train[col].nunique(dropna=True))
        te_cols = te_columns(x_train[[col]])
        rows.append({"feature": col, "distinct_levels": levels, "onehot_columns": levels,
                     "frequency_encoding_columns": 1, "target_encoder_columns": te_cols})
        other_levels_sum += levels

    n_features_total = len(other_cats) + 1
    total_raw = raw_levels + other_levels_sum
    total_market = market_levels + other_levels_sum
    rows.append({"feature": "TOTAL: all categoricals, one-hot with raw ServiceArea", "distinct_levels": np.nan,
                 "onehot_columns": total_raw, "frequency_encoding_columns": n_features_total,
                 "target_encoder_columns": n_features_total})
    rows.append({"feature": "TOTAL: all categoricals, one-hot with extracted market", "distinct_levels": np.nan,
                 "onehot_columns": total_market, "frequency_encoding_columns": n_features_total,
                 "target_encoder_columns": n_features_total})

    table = pd.DataFrame(rows)
    caption = (
        "Encoding options for ServiceArea and the other categoricals, fit on the training split only: "
        "resulting column counts for one-hot (raw ServiceArea and extracted market), frequency encoding, "
        "and scikit-learn TargetEncoder (which cross-fits internally)."
    )
    c.save_table(table, 14, "encoding_options", caption, "Task 4")
    return {
        "table14_servicearea_raw_levels": raw_levels, "table14_servicearea_market_levels": market_levels,
        "table14_total_onehot_raw": total_raw, "table14_total_onehot_market": total_market,
        "table14_target_encoder_columns_per_feature": raw_te,
    }


# --- Pipeline building blocks (Tables 15 to 18) ---
def build_preprocessor(numeric_cols: list[str], binary_cols: list[str], ordinal_cols: list[str], nominal_cols: list[str]) -> ColumnTransformer:
    """Impute, Yeo-Johnson the numbers and encode the categories. Scaling is added later, once.
    Used for every sampler except SMOTE-NC."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False)),
    ])
    binary_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[c.CREDIT_RATING_ORDER], handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    nominal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("bin", binary_pipe, binary_cols),
        ("ord", ordinal_pipe, ordinal_cols),
        ("nom", nominal_pipe, nominal_cols),
    ])


def build_smotenc_preprocessor(numeric_cols: list[str], binary_cols: list[str], ordinal_cols: list[str], nominal_cols: list[str]) -> tuple[ColumnTransformer, SMOTENC]:
    """Preprocessing plus SMOTE-NC, before one-hot encoding.
    Binary flags and nominal columns are marked categorical, so new rows copy real values."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("yeojohnson", PowerTransformer(method="yeo-johnson", standardize=False)),
    ]).set_output(transform="pandas")
    binary_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ]).set_output(transform="pandas")
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[c.CREDIT_RATING_ORDER], handle_unknown="use_encoded_value", unknown_value=-1)),
    ]).set_output(transform="pandas")
    nominal_impute_only = SimpleImputer(strategy="constant", fill_value="Missing")

    pre = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("bin", binary_pipe, binary_cols),
            ("ord", ordinal_pipe, ordinal_cols),
            ("nom", nominal_impute_only, nominal_cols),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    smotenc = SMOTENC(categorical_features=binary_cols + nominal_cols, random_state=c.RANDOM_STATE)
    return pre, smotenc


def build_smotenc_pipeline(classifier_name: str, numeric_cols: list[str], binary_cols: list[str], ordinal_cols: list[str], nominal_cols: list[str]) -> ImbPipeline:
    """Full SMOTE-NC pipeline: preprocess, resample, one-hot encode, scale, classify."""
    pre, smotenc = build_smotenc_preprocessor(numeric_cols, binary_cols, ordinal_cols, nominal_cols)

    onehot_after = ColumnTransformer(
        [("oh", OneHotEncoder(handle_unknown="ignore"), nominal_cols)],
        remainder="passthrough",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )

    steps = [
        ("preprocess", pre),
        ("smotenc", smotenc),
        ("encode_nominal", onehot_after),
        ("scale", StandardScaler()),
        ("clf", make_classifier(classifier_name)),
    ]
    return ImbPipeline(steps)


def verify_smotenc_binary_flags(x_train: pd.DataFrame, y_train: pd.Series, numeric_cols: list[str], binary_cols: list[str], ordinal_cols: list[str], nominal_cols: list[str]) -> dict:
    """Check that the binary flags are still exactly 0 or 1 after SMOTE-NC."""
    pre, smotenc = build_smotenc_preprocessor(numeric_cols, binary_cols, ordinal_cols, nominal_cols)
    x_res, _ = ImbPipeline([("preprocess", pre), ("smotenc", smotenc)]).fit_resample(x_train, y_train)
    seen = sorted(pd.unique(x_res[binary_cols].to_numpy().ravel()).tolist())
    ok = set(seen) <= {0.0, 1.0}
    assert ok, f"SMOTENC produced non-binary values in the flag columns: {seen}"
    print(f"    SMOTENC binary-flag check: PASS ({len(binary_cols)} flags, values seen = {seen})")
    return {
        "table15_smotenc_binary_flags_ok": bool(ok),
        "table15_smotenc_binary_flags_checked": len(binary_cols),
        "table15_smotenc_binary_flags_values_seen": seen,
    }


def make_sampler(name: str):
    if name == "none":
        return None
    if name == "RandomOverSampler":
        return RandomOverSampler(random_state=c.RANDOM_STATE)
    if name == "RandomUnderSampler":
        return RandomUnderSampler(random_state=c.RANDOM_STATE)
    if name == "SMOTE":
        return SMOTE(random_state=c.RANDOM_STATE)
    if name == "ADASYN":
        return ADASYN(random_state=c.RANDOM_STATE)
    raise ValueError(f"unknown sampler {name}")


def make_classifier(name: str):
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=c.RANDOM_STATE)
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=c.RANDOM_STATE)
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=c.RANDOM_STATE)
    raise ValueError(f"unknown classifier {name}")


def make_scoring() -> dict:
    return {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "specificity": make_scorer(recall_score, pos_label=0),
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }


def build_pipeline(sampler_name: str, classifier_name: str, numeric_cols: list[str], binary_cols: list[str], ordinal_cols: list[str], nominal_cols: list[str]) -> ImbPipeline:
    if sampler_name == "SMOTENC":
        return build_smotenc_pipeline(classifier_name, numeric_cols, binary_cols, ordinal_cols, nominal_cols)
    steps = [
        ("preprocess", build_preprocessor(numeric_cols, binary_cols, ordinal_cols, nominal_cols)),
        ("scale", StandardScaler()),
    ]
    sampler = make_sampler(sampler_name)
    if sampler is not None:
        steps.append(("sample", sampler))
    steps.append(("clf", make_classifier(classifier_name)))
    return ImbPipeline(steps)


# --- Table 15 / Figure 14: resampling methods x classifiers ---
def build_table15(x_train: pd.DataFrame, y_train: pd.Series, numeric_cols, binary_cols, ordinal_cols, nominal_cols) -> pd.DataFrame:
    samplers = ["none", "RandomOverSampler", "RandomUnderSampler", "SMOTE", "SMOTENC", "ADASYN"]
    classifiers = ["LogisticRegression", "RandomForest", "HistGradientBoosting"]
    scoring = make_scoring()
    skf = StratifiedKFold(n_splits=c.CV_FOLDS, shuffle=True, random_state=c.RANDOM_STATE)

    rows = []
    for clf_name in classifiers:
        for sampler_name in samplers:
            pipe = build_pipeline(sampler_name, clf_name, numeric_cols, binary_cols, ordinal_cols, nominal_cols)
            t0 = time.perf_counter()
            cv_res = cross_validate(pipe, x_train, y_train, cv=skf, scoring=scoring, n_jobs=1, error_score="raise")
            elapsed = time.perf_counter() - t0
            row = {"classifier": clf_name, "sampler": sampler_name, "cv_folds": c.CV_FOLDS, "fit_eval_time_seconds": elapsed}
            for m in scoring:
                arr = cv_res[f"test_{m}"]
                row[f"{m}_mean"] = float(arr.mean())
                row[f"{m}_std"] = float(arr.std())
            rows.append(row)
            print(f"    {clf_name:22s} + {sampler_name:18s} recall={row['recall_mean']:.3f} "
                  f"f1={row['f1_mean']:.3f} pr_auc={row['pr_auc_mean']:.3f} ({elapsed:.1f}s)")

    table = pd.DataFrame(rows)
    caption = (
        f"Stratified {c.CV_FOLDS}-fold CV on the training split only. Each row is an imblearn pipeline: "
        "clean placeholders, impute, Yeo-Johnson skewed numerics (unscaled), encode, scale, sampler, "
        "classifier. Metrics are for the Churn = Yes class except specificity (Churn = No recall), balanced "
        "accuracy and accuracy. Mean and standard deviation across folds."
    )
    c.save_table(table, 15, "resampling_comparison", caption, "Task 4")
    return table


def build_figure14(table15: pd.DataFrame) -> None:
    metrics = ["recall_mean", "f1_mean", "pr_auc_mean"]
    classifiers = list(table15["classifier"].unique())
    samplers = list(table15["sampler"].unique())
    fig, axes = plt.subplots(1, len(classifiers), figsize=(15, 5.2), sharey=True)
    x = np.arange(len(samplers))
    width = 0.25
    for ax, clf_name in zip(axes, classifiers):
        sub = table15[table15["classifier"] == clf_name].set_index("sampler").loc[samplers]
        for i, m in enumerate(metrics):
            ax.bar(x + (i - 1) * width, sub[m], width=width, label=m.replace("_mean", ""))
        ax.set_xticks(x)
        ax.set_xticklabels(samplers, rotation=30, ha="right", fontsize=8)
        ax.set_title(clf_name, fontsize=9)
    axes[0].set_ylabel("CV mean score")
    axes[0].legend(fontsize=8)
    fig.suptitle("Recall, F1 and PR-AUC by sampler, grouped by classifier (CV mean)", y=1.03)
    fig.tight_layout()
    c.save_fig(14, "sampler_comparison", "Recall, F1 and PR-AUC by resampling method, grouped by classifier.", "Task 4")


# --- Table 16: leakage test ---
def build_table16(table15: pd.DataFrame, x_train: pd.DataFrame, y_train: pd.Series, numeric_cols, binary_cols, ordinal_cols, nominal_cols) -> tuple[pd.DataFrame, dict]:
    """Leakage test on the same folds: A fits everything before CV (wrong),
    B fits only SMOTE inside CV, C fits everything inside CV (correct)."""
    scoring = make_scoring()
    skf = StratifiedKFold(n_splits=c.CV_FOLDS, shuffle=True, random_state=c.RANDOM_STATE)

    # Arm A (WRONG, on purpose): preprocessing, scaling and SMOTE fitted once before CV,
    # so validation rows leak into training.
    pre_a = build_preprocessor(numeric_cols, binary_cols, ordinal_cols, nominal_cols)
    x_scaled_a = StandardScaler().fit_transform(pre_a.fit_transform(x_train))
    x_res_a, y_res_a = SMOTE(random_state=c.RANDOM_STATE).fit_resample(x_scaled_a, y_train)
    arm_a_res = cross_validate(make_classifier("RandomForest"), x_res_a, y_res_a, cv=skf, scoring=scoring, n_jobs=1)

    # Arm B: preprocessing and scaling fitted before CV, but SMOTE refitted inside each fold.
    pre_b = build_preprocessor(numeric_cols, binary_cols, ordinal_cols, nominal_cols)
    x_scaled_b = StandardScaler().fit_transform(pre_b.fit_transform(x_train))
    pipe_b = ImbPipeline([
        ("sample", SMOTE(random_state=c.RANDOM_STATE)),
        ("clf", make_classifier("RandomForest")),
    ])
    arm_b_res = cross_validate(pipe_b, x_scaled_b, y_train, cv=skf, scoring=scoring, n_jobs=1)

    # Arm C (CORRECT): everything refitted inside each fold.
    # This is the Table 15 SMOTE + RandomForest row, so it is reused.
    correct_row = table15[(table15["sampler"] == "SMOTE") & (table15["classifier"] == "RandomForest")].iloc[0]

    rows = []
    for m in scoring:
        a_mean = float(arm_a_res[f"test_{m}"].mean())
        b_mean = float(arm_b_res[f"test_{m}"].mean())
        c_mean = float(correct_row[f"{m}_mean"])
        rows.append({
            "metric": m,
            "arm_a_wrong_mean": a_mean,
            "arm_b_mean": b_mean,
            "arm_c_correct_mean": c_mean,
            "a_minus_b_smote_timing": a_mean - b_mean,
            "b_minus_c_preprocessing_timing": b_mean - c_mean,
            "a_minus_c_total": a_mean - c_mean,
        })
    table = pd.DataFrame(rows)
    caption = (
        f"Three-arm leakage ablation (RandomForest classifier), same {len(x_train):,}-row training split and "
        "the same StratifiedKFold splits in every arm. Arm A (WRONG) fits preprocessing, scaling and SMOTE "
        "once on the whole training split, before cross-validation. Arm B fits preprocessing and scaling once "
        "on the training split, same as arm A, but refits SMOTE inside each CV fold. Arm C (CORRECT) refits "
        "preprocessing, scaling and SMOTE inside each fold; identical to the Table 15 SMOTE + RandomForest "
        "row. A minus B isolates the effect of fitting SMOTE before the folds rather than inside them; B "
        "minus C isolates the effect of fitting preprocessing and scaling before the folds rather than inside "
        "them; A minus C is the total effect of getting both wrong. A positive difference means the "
        "earlier-fitted, more leaky arm is optimistic."
    )
    c.save_table(table, 16, "leakage_demonstration", caption, "Task 4")

    numbers = {}
    for _, r in table.iterrows():
        m = r["metric"]
        numbers[f"table16_arm_a_wrong_{m}_mean"] = float(r["arm_a_wrong_mean"])
        numbers[f"table16_arm_b_{m}_mean"] = float(r["arm_b_mean"])
        numbers[f"table16_arm_c_correct_{m}_mean"] = float(r["arm_c_correct_mean"])
        numbers[f"table16_diff_a_minus_b_{m}"] = float(r["a_minus_b_smote_timing"])
        numbers[f"table16_diff_b_minus_c_{m}"] = float(r["b_minus_c_preprocessing_timing"])
        numbers[f"table16_diff_a_minus_c_{m}"] = float(r["a_minus_c_total"])
    return table, numbers


# --- Table 17: effect of scaling ---
def build_table17(x_train: pd.DataFrame, y_train: pd.Series, numeric_cols, binary_cols, ordinal_cols, nominal_cols) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=c.CV_FOLDS, shuffle=True, random_state=c.RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "f1": "f1"}
    rows = []
    for clf_name in ("KNN", "RandomForest"):
        for scale in (True, False):
            # The StandardScaler below is the only difference between the two arms.
            steps = [("preprocess", build_preprocessor(numeric_cols, binary_cols, ordinal_cols, nominal_cols))]
            if scale:
                steps.append(("scale", StandardScaler()))
            clf = KNeighborsClassifier(n_neighbors=5) if clf_name == "KNN" else make_classifier("RandomForest")
            steps.append(("clf", clf))
            pipe = Pipeline(steps)
            res = cross_validate(pipe, x_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
            rows.append({
                "classifier": clf_name, "standardised": scale,
                "roc_auc_mean": float(res["test_roc_auc"].mean()), "roc_auc_std": float(res["test_roc_auc"].std()),
                "f1_mean": float(res["test_f1"].mean()), "f1_std": float(res["test_f1"].std()),
            })
    table = pd.DataFrame(rows)
    caption = (
        "KNN and RandomForest with and without standardisation, CV ROC-AUC and F1 on the training split. "
        "Both arms impute and Yeo-Johnson transform the numeric block with PowerTransformer(standardize=False), "
        "so the power transform itself never scales anything; the scaled arm then applies an explicit "
        "StandardScaler to the whole encoded matrix, and the unscaled arm applies no scaler at all. That is "
        "the only difference between the two arms; the table reports whatever direction the data shows."
    )
    c.save_table(table, 17, "scaling_sensitivity", caption, "Task 4")
    return table


# --- Table 18 / Figure 15: final test-set evaluation ---
def build_table18_figure15(table15, x_train, y_train, x_test, y_test, numeric_cols, binary_cols, ordinal_cols, nominal_cols):
    # Two candidates picked from Table 15 before touching the test set: best PR-AUC and best F1.
    # Each is trained once and tested once.
    candidates = [
        ("a", "best CV PR-AUC in Table 15", table15.loc[table15["pr_auc_mean"].idxmax()]),
        ("b", "best CV F1 (Churn = Yes) in Table 15", table15.loc[table15["f1_mean"].idxmax()]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    table_rows = []
    results = {}
    for ax, (key, chosen_by, row) in zip(axes, candidates):
        sampler_name, clf_name = row["sampler"], row["classifier"]
        pipe = build_pipeline(sampler_name, clf_name, numeric_cols, binary_cols, ordinal_cols, nominal_cols)
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        y_proba = pipe.predict_proba(x_test)[:, 1]

        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "specificity": recall_score(y_test, y_pred, pos_label=0),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
        }
        label = f"({key}) {chosen_by}"
        table_rows.append({"candidate": label, "sampler": sampler_name, "classifier": clf_name, **metrics})
        results[key] = {"metrics": metrics, "sampler": sampler_name, "classifier": clf_name, "chosen_by": chosen_by}

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        ax.set_xlabel("Predicted Churn")
        ax.set_ylabel("Actual Churn")
        ax.set_title(f"({key}) {clf_name} + {sampler_name}\n{chosen_by}", fontsize=9)

    table = pd.DataFrame(table_rows)
    caption = (
        "Held-out test-set metrics for two candidates pre-declared from Table 15's CV results before the "
        "test set was touched: (a) best CV PR-AUC (ranking quality) and (b) best CV F1 for the Churn = Yes "
        "class (operating-point quality). Each pre-declared candidate is fit once on the training split and "
        "scored once on the test split."
    )
    c.save_table(table, 18, "held_out_test_metrics", caption, "Task 4")

    fig.suptitle("Held-out test confusion matrices, two pre-declared candidates", y=1.04)
    fig.tight_layout()
    c.save_fig(
        15, "confusion_matrix",
        "Held-out test confusion matrices for the two candidates pre-declared from Table 15's CV results: "
        "(a) best CV PR-AUC and (b) best CV F1 for Churn = Yes. Each candidate touched the test set once.",
        "Task 4",
    )

    return results


# --- Figure 16: pipeline flowchart ---
def build_figure16_flowchart() -> None:
    steps = [
        "Split\n(train/test)",
        "Clean\n(duplicates,\ninconsistencies)",
        "Impute\n(median / mode)",
        "Transform skewed\nnumerics\n(Yeo-Johnson)",
        "Encode\n(ordinal, one-hot)",
        "Scale\n(standardise)",
        "Select / reduce\n(PCA, optional)",
        "Resample\n(training folds\nonly)",
        "Model\n(fit / predict)",
    ]
    n = len(steps)
    box_w, box_h, gap = 1.0, 0.9, 0.5
    step_x = box_w + gap
    fig, ax = plt.subplots(figsize=(15, 3.2))
    for i, label in enumerate(steps):
        x = i * step_x
        rect = plt.Rectangle((x, 0), box_w, box_h, facecolor="C0", alpha=0.15, edgecolor="C0", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, box_h / 2, label, ha="center", va="center", fontsize=8)
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap, box_h / 2), xytext=(x + box_w, box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2))
    ax.set_xlim(-0.2, (n - 1) * step_x + box_w + 0.2)
    ax.set_ylim(-0.2, box_h + 0.2)
    ax.axis("off")
    ax.set_title("Leakage-safe pipeline order (resampling runs on training folds only)")
    c.save_fig(
        16, "pipeline_flowchart",
        "Ordered leakage-safe pipeline: split, clean, impute, transform skewed features, encode, scale, "
        "select/reduce, resample on training folds only, model.",
        "Task 4",
    )


# --- Run everything ---
def main() -> None:
    args = c.make_arg_parser("Task 4 evidence: preprocessing strategy, PCA, encoding, resampling comparison.").parse_args()
    c.set_plot_style()
    t0 = time.time()

    df = c.load_train(quick=args.quick)
    cleaned = pre_clean(df)
    numeric_cols, binary_cols, ordinal_cols, nominal_cols = build_feature_lists()

    x_all = cleaned[numeric_cols + binary_cols + ordinal_cols + nominal_cols]
    y_all = (cleaned[c.TARGET_COL] == "Yes").astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.2, stratify=y_all, random_state=c.RANDOM_STATE
    )
    service_area_raw_train = df.loc[x_train.index, "ServiceArea"]
    print(f"  train={len(x_train)}, test={len(x_test)}, cv_folds={c.CV_FOLDS}")

    numbers: dict = {"script03_train_rows": int(len(x_train)), "script03_test_rows": int(len(x_test))}

    print("Task 4: Table 12 / Figure 12 skew before/after transforms")
    numbers.update(build_table12_figure12(x_train))
    print("Task 4: Figure 17 Yeo-Johnson on typical (not zero-inflated) skewed columns")
    numbers.update(build_figure17_typical_skew(x_train))
    print("Task 4: Table 13 / Figure 13 PCA")
    numbers.update(build_table13_figure13_pca(x_train, numeric_cols))
    print("Task 4: Table 14 encoding options")
    numbers.update(build_table14(x_train, y_train, service_area_raw_train))

    print("Task 4: Table 15 resampling x classifier comparison (slowest step)")
    table15 = build_table15(x_train, y_train, numeric_cols, binary_cols, ordinal_cols, nominal_cols)
    print("Task 4: Figure 14 grouped bars by sampler")
    build_figure14(table15)

    print("Task 4: SMOTENC binary-flag check")
    numbers.update(verify_smotenc_binary_flags(x_train, y_train, numeric_cols, binary_cols, ordinal_cols, nominal_cols))

    print("Task 4: Table 16 leakage demonstration (three-arm ablation)")
    _, table16_numbers = build_table16(table15, x_train, y_train, numeric_cols, binary_cols, ordinal_cols, nominal_cols)
    numbers.update(table16_numbers)

    print("Task 4: Table 17 scaling sensitivity")
    build_table17(x_train, y_train, numeric_cols, binary_cols, ordinal_cols, nominal_cols)

    print("Task 4: Table 18 / Figure 15 held-out test evaluation")
    results = build_table18_figure15(
        table15, x_train, y_train, x_test, y_test, numeric_cols, binary_cols, ordinal_cols, nominal_cols
    )
    numbers["table18_chosen_sampler"] = results["a"]["sampler"]
    numbers["table18_chosen_classifier"] = results["a"]["classifier"]
    numbers["table18_chosen_by"] = results["a"]["chosen_by"]
    for k, v in results["a"]["metrics"].items():
        numbers[f"table18_test_{k}"] = float(v)
    numbers["table18_b_chosen_sampler"] = results["b"]["sampler"]
    numbers["table18_b_chosen_classifier"] = results["b"]["classifier"]
    numbers["table18_b_chosen_by"] = results["b"]["chosen_by"]
    for k, v in results["b"]["metrics"].items():
        numbers[f"table18_b_test_{k}"] = float(v)

    print("Task 4: Figure 16 pipeline flowchart")
    build_figure16_flowchart()

    numbers["script03_cv_folds"] = c.CV_FOLDS
    numbers["script03_quick_mode"] = bool(args.quick)
    numbers["script03_runtime_seconds"] = round(time.time() - t0, 1)
    c.update_key_numbers(numbers)

    print(f"03_preprocessing_pipeline.py finished in {time.time() - t0:.1f}s (quick={args.quick})")


if __name__ == "__main__":
    main()
