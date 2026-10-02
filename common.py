"""Shared helpers for the Cell2Cell churn analysis: loading, column groups, placeholders, plotting and saving.
Column descriptions are my own reading of the column names, since Kaggle gives no data dictionary."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# --- Settings ---
RANDOM_STATE = 42
QUICK_SAMPLE_SIZE = 10_000  # rows used for a quick test run
SILHOUETTE_SAMPLE_SIZE = 10_000  # rows used for the silhouette score in Task 2
CV_FOLDS = 5  # 5-fold cross-validation (full run takes about 15 minutes)

# --- File paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
REGISTER_STATE_PATH = OUTPUT_DIR / "_register_state.csv"  # internal list of saved outputs
REGISTER_MD_PATH = OUTPUT_DIR / "figure_table_register.md"
KEY_NUMBERS_PATH = OUTPUT_DIR / "key_numbers.json"

TRAIN_PATH = DATA_DIR / "cell2celltrain.csv"
HOLDOUT_PATH = DATA_DIR / "cell2cellholdout.csv"

for _d in (OUTPUT_DIR, FIGURE_DIR, TABLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Column groups ---
# Each of the 58 columns belongs to exactly one group, which decides how it is preprocessed.
ID_COL = "CustomerID"
TARGET_COL = "Churn"

NUMERIC_CONTINUOUS = [
    "MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge",
    "DirectorAssistedCalls", "OverageMinutes", "RoamingCalls",
    "PercChangeMinutes", "PercChangeRevenues", "CurrentEquipmentDays",
    "AgeHH1", "AgeHH2",
]

NUMERIC_COUNT = [
    "DroppedCalls", "BlockedCalls", "UnansweredCalls", "CustomerCareCalls",
    "ThreewayCalls", "ReceivedCalls", "OutboundCalls", "InboundCalls",
    "PeakCallsInOut", "OffPeakCallsInOut", "DroppedBlockedCalls",
    "CallForwardingCalls", "CallWaitingCalls", "MonthsInService",
    "UniqueSubs", "ActiveSubs", "Handsets", "HandsetModels",
    "RetentionCalls", "RetentionOffersAccepted", "ReferralsMadeBySubscriber",
    "AdjustmentsToCreditRating",
]

BINARY_YESNO = [
    "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable", "TruckOwner",
    "RVOwner", "BuysViaMailOrder", "RespondsToMailOffers", "OptOutMailings",
    "NonUSTravel", "OwnsComputer", "HasCreditCard", "NewCellphoneUser",
    "NotNewCellphoneUser", "OwnsMotorcycle", "MadeCallToRetentionTeam",
]

# Categories with no natural order.
NOMINAL_CATEGORICAL = ["ServiceArea", "PrizmCode", "Occupation", "Homeownership", "MaritalStatus"]

# Categories with a natural order (IncomeGroup is stored as a number).
ORDINAL_CATEGORICAL = ["CreditRating", "HandsetPrice", "IncomeGroup"]

ALL_ANALYTICAL_GROUPS = {
    "id": [ID_COL],
    "target": [TARGET_COL],
    "numeric_continuous": NUMERIC_CONTINUOUS,
    "numeric_count": NUMERIC_COUNT,
    "binary_yesno": BINARY_YESNO,
    "nominal_categorical": NOMINAL_CATEGORICAL,
    "ordinal_categorical": ORDINAL_CATEGORICAL,
}
ANALYTICAL_TYPE = {col: grp for grp, cols in ALL_ANALYTICAL_GROUPS.items() for col in cols}

# Numeric vs text columns for the Task 3 statistics.
# IncomeGroup is stored as a number, so it counts as numeric here.
NUMERIC_FEATURES = NUMERIC_CONTINUOUS + NUMERIC_COUNT + ["IncomeGroup"]
CATEGORICAL_FEATURES = BINARY_YESNO + NOMINAL_CATEGORICAL + ["CreditRating", "HandsetPrice"]

# CreditRating order from lowest to highest, used for ordinal encoding.
CREDIT_RATING_ORDER = ["7-Lowest", "6-VeryLow", "5-Low", "4-Medium", "3-Good", "2-High", "1-Highest"]

# Possible leakage: retention-team contact may happen after a customer already plans to leave.
LEAKAGE_RISK_COLS = ["RetentionCalls", "RetentionOffersAccepted", "MadeCallToRetentionTeam"]

# --- Placeholder rules ---
# Disguised missing values. treat_as_missing=True turns them into NaN, False keeps them as a real category.
PLACEHOLDER_RULES = {
    "HandsetPrice": {
        "placeholder_value": "Unknown",
        "treat_as_missing": True,
        "rule": "text 'Unknown' in an otherwise numeric USD handset price band: parse remaining values to float, set 'Unknown' to NaN",
    },
    "MaritalStatus": {
        "placeholder_value": "Unknown",
        "treat_as_missing": True,
        "rule": "text 'Unknown' alongside Yes/No responses: set to NaN",
    },
    "Homeownership": {
        "placeholder_value": "Unknown",
        "treat_as_missing": False,
        "rule": "only takes Known/Unknown: the column records whether ownership status is known, not ownership itself, so 'Unknown' is kept as a legitimate category, not converted to NaN",
    },
    "AgeHH1": {
        "placeholder_value": 0,
        "treat_as_missing": True,
        "rule": "0 is not a plausible age for the first adult in the household: set to NaN",
    },
    "AgeHH2": {
        "placeholder_value": 0,
        "treat_as_missing": True,
        "rule": "0 may mean no second adult in the household or an unrecorded age (ambiguous): set to NaN for the age value",
    },
    "IncomeGroup": {
        "placeholder_value": 0,
        "treat_as_missing": True,
        "rule": "0 appears to denote unknown income: bands otherwise run 1 (lowest) to 9 (highest), set to NaN",
    },
    "Occupation": {
        "placeholder_value": "Other",
        "treat_as_missing": False,
        "rule": "'Other' is a vague catch-all category: reported but not treated as missing",
    },
    "PrizmCode": {
        "placeholder_value": "Other",
        "treat_as_missing": False,
        "rule": "'Other' is a vague catch-all category: reported but not treated as missing",
    },
}


def clean_placeholders(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()
    for col, rule in PLACEHOLDER_RULES.items():
        if col not in out.columns or not rule["treat_as_missing"]:
            continue
        mask = out[col] == rule["placeholder_value"]
        out.loc[mask, col] = np.nan
    if "HandsetPrice" in out.columns:
        # With 'Unknown' gone, the rest ("30", "150", ...) can be read as numbers.
        out["HandsetPrice"] = pd.to_numeric(out["HandsetPrice"], errors="coerce")
    return out


def extract_service_area_market(series: pd.Series) -> pd.Series:
    """Keep the first 3 letters (the market), e.g. SEABLV425 -> SEA. Cuts 747 levels to 57."""
    return series.str.slice(0, 3)


# --- Plain-English column descriptions (Table 2) ---
FEATURE_DESCRIPTIONS = {
    "CustomerID": "unique subscriber identifier",
    "Churn": "target: whether the customer churned (Yes/No)",
    "MonthlyRevenue": "mean monthly revenue (USD)",
    "MonthlyMinutes": "mean monthly minutes of use",
    "TotalRecurringCharge": "mean total monthly recurring charge (USD)",
    "DirectorAssistedCalls": "mean number of directory-assisted calls per month",
    "OverageMinutes": "mean monthly minutes used beyond the plan allowance",
    "RoamingCalls": "mean number of roaming calls per month",
    "PercChangeMinutes": "percentage change in minutes of use versus the previous period",
    "PercChangeRevenues": "percentage change in revenue versus the previous period",
    "DroppedCalls": "mean number of dropped voice calls per month",
    "BlockedCalls": "mean number of blocked (failed connection) calls per month",
    "UnansweredCalls": "mean number of unanswered calls per month",
    "CustomerCareCalls": "mean number of calls placed to customer care per month",
    "ThreewayCalls": "mean number of three-way calls per month",
    "ReceivedCalls": "mean number of inbound calls received per month",
    "OutboundCalls": "mean number of outbound calls placed per month",
    "InboundCalls": "mean number of inbound calls per month",
    "PeakCallsInOut": "mean number of inbound and outbound calls during peak hours",
    "OffPeakCallsInOut": "mean number of inbound and outbound calls during off-peak hours",
    "DroppedBlockedCalls": "mean number of dropped or blocked calls per month (intended to equal DroppedCalls + BlockedCalls)",
    "CallForwardingCalls": "mean number of call-forwarding calls per month",
    "CallWaitingCalls": "mean number of call-waiting calls per month",
    "MonthsInService": "months since account activation",
    "UniqueSubs": "number of unique subscribers on the account",
    "ActiveSubs": "number of active subscribers on the account",
    "ServiceArea": "service area code: 3-letter market + 3-letter city + 3-digit telephone area code",
    "Handsets": "number of handsets issued to the account",
    "HandsetModels": "number of distinct handset models issued to the account",
    "CurrentEquipmentDays": "days since the current handset was issued",
    "AgeHH1": "age of the first adult in the household (years)",
    "AgeHH2": "age of the second adult in the household (years), 0 if none or unrecorded",
    "ChildrenInHH": "whether children are present in the household",
    "HandsetRefurbished": "whether the current handset is refurbished",
    "HandsetWebCapable": "whether the current handset can browse the web",
    "TruckOwner": "whether the household owns a truck",
    "RVOwner": "whether the household owns a recreational vehicle",
    "Homeownership": "whether home-ownership status is known (Known/Unknown), not ownership itself",
    "BuysViaMailOrder": "whether the customer buys via mail order",
    "RespondsToMailOffers": "whether the customer responds to mail offers",
    "OptOutMailings": "whether the customer opted out of marketing mailings",
    "NonUSTravel": "whether the customer travels outside the United States",
    "OwnsComputer": "whether the household owns a computer",
    "HasCreditCard": "whether the customer has a credit card on file",
    "RetentionCalls": "number of calls made to the retention team",
    "RetentionOffersAccepted": "number of retention offers accepted",
    "NewCellphoneUser": "whether the customer is a first-time cellphone user",
    "NotNewCellphoneUser": "whether the customer is not a first-time cellphone user",
    "ReferralsMadeBySubscriber": "number of referrals made by the subscriber",
    "IncomeGroup": "household income band, 1 (lowest) to 9 (highest); 0 appears to mean unknown",
    "OwnsMotorcycle": "whether the household owns a motorcycle",
    "AdjustmentsToCreditRating": "number of adjustments made to the customer's credit rating",
    "HandsetPrice": "handset price band in USD",
    "MadeCallToRetentionTeam": "whether the customer ever called the retention team",
    "CreditRating": "ordinal credit-rating band, 1-Highest to 7-Lowest",
    "PrizmCode": "geodemographic segment (PRIZM cluster code)",
    "Occupation": "customer's occupation category",
    "MaritalStatus": "whether the customer is married (Yes/No); Unknown where not captured",
}


def build_feature_notes(df: pd.DataFrame) -> dict:
    """Notes for Table 2 (placeholders, leakage, redundancy).
    Counts are calculated from the data, not typed in."""
    n = len(df)
    notes = {}
    for col in LEAKAGE_RISK_COLS:
        notes[col] = "possible leakage: contact with the retention team may reflect churn intent recorded close to the churn decision"
    both_no = int(((df["NewCellphoneUser"] == "No") & (df["NotNewCellphoneUser"] == "No")).sum())
    redundancy_note = (
        f"redundant pair: both No in {both_no:,} rows ({both_no / n:.1%}) means new/not-new status is unknown, not that the customer is neither"
    )
    notes["NewCellphoneUser"] = redundancy_note
    notes["NotNewCellphoneUser"] = redundancy_note
    notes["HandsetPrice"] = f"placeholder: 'Unknown' text mixed with a numeric USD price band ({int((df['HandsetPrice'] == 'Unknown').sum()):,} rows)"
    notes["MaritalStatus"] = f"placeholder: 'Unknown' text alongside Yes/No ({int((df['MaritalStatus'] == 'Unknown').sum()):,} rows)"
    notes["Homeownership"] = "records whether ownership status is known, not ownership itself"
    notes["AgeHH1"] = f"placeholder: 0 is not a plausible age ({int((df['AgeHH1'] == 0).sum()):,} rows)"
    notes["AgeHH2"] = f"placeholder: 0 is not a plausible age ({int((df['AgeHH2'] == 0).sum()):,} rows); ambiguous, may mean no second adult"
    notes["IncomeGroup"] = f"placeholder: 0 appears to mean unknown income ({int((df['IncomeGroup'] == 0).sum()):,} rows); bands otherwise run 1 to 9"
    occ_other_pct = (df["Occupation"] == "Other").mean()
    notes["Occupation"] = f"'Other' is a vague catch-all ({occ_other_pct:.1%} of rows), reported but not treated as missing"
    prizm_other_pct = (df["PrizmCode"] == "Other").mean()
    notes["PrizmCode"] = f"'Other' is a vague catch-all ({prizm_other_pct:.1%} of rows), reported but not treated as missing"
    notes["ServiceArea"] = f"{df['ServiceArea'].nunique()} levels; first 3 characters encode a {df['ServiceArea'].dropna().str.slice(0, 3).nunique()}-level market, see Task 3 Figure 11"
    valid = df["DroppedBlockedCalls"].notna() & df["DroppedCalls"].notna() & df["BlockedCalls"].notna()
    mismatch_pct = 1 - np.isclose(
        df.loc[valid, "DroppedBlockedCalls"], df.loc[valid, "DroppedCalls"] + df.loc[valid, "BlockedCalls"]
    ).mean()
    notes["DroppedBlockedCalls"] = f"inconsistent with DroppedCalls + BlockedCalls in about {mismatch_pct:.1%} of rows, see Table 11"
    notes["CustomerID"] = "identifier, excluded from modelling"
    return notes


def validate_column_groups(df: pd.DataFrame) -> None:
    """Check that every column is in exactly one group."""
    declared = {ID_COL, TARGET_COL} | set(NUMERIC_CONTINUOUS) | set(NUMERIC_COUNT) | set(BINARY_YESNO) | set(NOMINAL_CATEGORICAL) | set(ORDINAL_CATEGORICAL)
    actual = set(df.columns)
    assert declared == actual, f"column group / dataframe mismatch: {declared ^ actual}"
    text_cols = set(df.select_dtypes(exclude="number").columns)
    declared_text = set(BINARY_YESNO) | set(NOMINAL_CATEGORICAL) | {"CreditRating", "HandsetPrice", TARGET_COL}
    assert text_cols == declared_text, f"text dtype / declared categorical mismatch: {text_cols ^ declared_text}"


# --- Data loading ---
def load_train(quick: bool = False) -> pd.DataFrame:
    """Load the training file. With --quick, use a 10,000-row sample."""
    df = pd.read_csv(TRAIN_PATH)
    if quick:
        from sklearn.model_selection import train_test_split

        df, _ = train_test_split(
            df, train_size=QUICK_SAMPLE_SIZE, stratify=df[TARGET_COL], random_state=RANDOM_STATE
        )
        df = df.reset_index(drop=True)
    return df


def load_holdout() -> pd.DataFrame:
    """Load the holdout file. It has no Churn labels, so it is never used for modelling."""
    return pd.read_csv(HOLDOUT_PATH)


# --- Plot style ---
def set_plot_style() -> None:
    """Colourblind-friendly colours and readable font sizes for all plots."""
    sns.set_theme(context="notebook", style="whitegrid", palette="colorblind")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "axes.titleweight": "bold",
            "figure.constrained_layout.use": False,
        }
    )


# --- Saving figures, tables and key numbers ---
_REGISTER_COLUMNS = ["kind", "number", "file", "caption", "task", "script"]


def _caller_script() -> str:
    """Name of the script that called this module."""
    this_file = Path(__file__).name
    for frame in inspect.stack():
        name = Path(frame.filename).name
        if name != this_file:
            return name
    return "unknown"


def _load_register() -> pd.DataFrame:
    if REGISTER_STATE_PATH.exists():
        return pd.read_csv(REGISTER_STATE_PATH)
    return pd.DataFrame(columns=_REGISTER_COLUMNS)


def _write_register(reg: pd.DataFrame) -> None:
    reg = reg.sort_values(["kind", "number"]).reset_index(drop=True)
    reg.to_csv(REGISTER_STATE_PATH, index=False)
    lines = [
        "# Figure and table register",
        "",
        "Generated automatically by common.py; do not edit by hand.",
        "",
        "| Kind | Number | File | Caption | Task | Script |",
        "|---|---|---|---|---|---|",
    ]
    def _esc(value: str) -> str:
        # escape "|" so captions like "|z| > 3" don't break the table
        return str(value).replace("|", "\\|")

    for _, r in reg.iterrows():
        lines.append(
            f"| {_esc(r['kind'])} | {int(r['number'])} | {_esc(r['file'])} | {_esc(r['caption'])} | "
            f"{_esc(r['task'])} | {_esc(r['script'])} |"
        )
    REGISTER_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record_output(kind: str, num: int, filename: str, caption: str, task: str) -> None:
    reg = _load_register()
    reg = reg[~((reg["kind"] == kind) & (reg["number"] == num))]
    new_row = pd.DataFrame(
        [{"kind": kind, "number": int(num), "file": filename, "caption": caption, "task": task, "script": _caller_script()}]
    )
    reg = pd.concat([reg, new_row], ignore_index=True)
    _write_register(reg)


def save_fig(num: int, slug: str, caption: str, task: str, fig=None) -> Path:
    """Save a figure as outputs/figures/FigNN_name.png."""
    fig = fig if fig is not None else plt.gcf()
    filename = f"Fig{int(num):02d}_{slug}.png"
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    _record_output("Figure", num, f"figures/{filename}", caption, task)
    print(f"  saved {filename}")
    return path


def save_table(df: pd.DataFrame, num: int, slug: str, caption: str, task: str, index: bool = False) -> Path:
    """Save a table as outputs/tables/TableNN_name.csv."""
    filename = f"Table{int(num):02d}_{slug}.csv"
    path = TABLE_DIR / filename
    df.to_csv(path, index=index)
    _record_output("Table", num, f"tables/{filename}", caption, task)
    print(f"  saved {filename}")
    return path


def _json_safe(obj):
    """Convert numpy/pandas values to plain Python so they can be saved as JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def update_key_numbers(new_numbers: dict) -> None:
    """Add this script's headline numbers to key_numbers.json.
    Numbers from the other scripts are kept."""
    existing = {}
    if KEY_NUMBERS_PATH.exists():
        existing = json.loads(KEY_NUMBERS_PATH.read_text(encoding="utf-8"))
    existing.update(_json_safe(new_numbers))
    KEY_NUMBERS_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


# --- Statistics helpers ---
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V: strength of link between two categorical columns (0 to 1)."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.to_numpy().sum()
    if n <= 1:
        return np.nan
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def iqr_bounds(series: pd.Series) -> tuple[float, float]:
    """IQR outlier limits: Q1 - 1.5*IQR and Q3 + 1.5*IQR."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


# --- Command-line options ---
def make_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Run on a stratified 10,000-row subsample of the training file for a fast smoke test.",
    )
    return p
