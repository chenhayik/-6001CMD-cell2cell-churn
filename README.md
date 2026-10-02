# Cell2Cell churn analysis (Coventry 6001CMD CW1)

Reproducible Python analysis code and evidence (figures, tables, key numbers)
for a Machine Learning report on the Cell2Cell telecom churn dataset:
problem formulation, supervised vs unsupervised comparison, data quality
analysis, and a preprocessing strategy. This repository is code and evidence
only; it does not contain the report itself.

## Data provenance

- Source used here: Kaggle, "Cell2Cell: Churn Datasets for Telecom Company",
  https://www.kaggle.com/datasets/jpacse/datasets-for-churn-telecom.
  Licence as listed on Kaggle: Unknown.
- Original source: the Duke University Teradata Center for Customer
  Relationship Management churn modelling tournament. Cited as: Neslin, S.A.,
  Gupta, S., Kamakura, W., Lu, J. and Mason, C.H. (2006) "Defection
  Detection: Measuring and Understanding the Predictive Accuracy of Customer
  Churn Models", Journal of Marketing Research, 43(2), pp. 204 to 211.
  doi:10.1509/jmkr.43.2.204.
- Files (not included in this repository because the Kaggle licence is
  unknown: download them from the Kaggle link above into `data/`; no script
  modifies them):
  - `data/cell2celltrain.csv`: 51,047 rows x 58 columns, target `Churn`
    (Yes/No), used throughout.
  - `data/cell2cellholdout.csv`: 20,000 rows x 58 columns, `Churn` entirely
    empty, no `CustomerID` overlap with the training file. Described only
    (Table 1), never used for modelling: it has no labels.

## Environment setup

```
python -m venv <path-to-venv>
<path-to-venv>\Scripts\activate     # Windows
pip install -r requirements.txt
```

`requirements.txt` pins the seven packages this project uses (numpy, pandas,
matplotlib, seaborn, scipy, scikit-learn, imbalanced-learn) to the exact
versions the scripts were developed and tested against. No other package is
required or used.

All scripts assume they are run from the project root (the directory
containing `common.py`), for example:

```
<path-to-venv>\Scripts\python.exe 01_data_quality.py
```

## Run order

The scripts are independent (each loads its own data) but are numbered in
report order and are intended to be run in sequence, so that the figure/table
register and `key_numbers.json` end up complete:

1. `01_data_quality.py`: Task 1 tables/figures and all Task 3 evidence.
2. `02_paradigm_evidence.py`: Task 2 (supervised vs unsupervised) evidence.
3. `03_preprocessing_pipeline.py`: Task 4 (preprocessing strategy) evidence.

Each script can also be re-run on its own; it only overwrites its own
figures, tables and key-number entries, identified by their fixed
Fig/Table number.

## The `--quick` flag

Every script accepts `--quick`, which runs on a stratified 10,000-row
subsample of the training file (see `common.QUICK_SAMPLE_SIZE`) for a faster
smoke test than a full run. Measured quick-mode runtimes on the development
machine:

| Script | `--quick` runtime |
|---|---|
| `01_data_quality.py` | 12.3 seconds |
| `02_paradigm_evidence.py` | 28.9 seconds |
| `03_preprocessing_pipeline.py` | 223.1 seconds (about 3.7 minutes) |

`03_preprocessing_pipeline.py --quick` is not "well under a minute": Table 15
still cross-validates 18 pipeline configurations (with `CV_FOLDS = 5`) and
Table 18 fits two pre-declared candidates, just on a 10,000-row subsample
instead of the full 51,047 rows, so it is the slowest script even in quick
mode. Use it while developing or checking the code runs end to end:

```
python 01_data_quality.py --quick
python 02_paradigm_evidence.py --quick
python 03_preprocessing_pipeline.py --quick
```

**The figures, tables, register and `key_numbers.json` used in the report
must come from full runs (no `--quick`).** Numbers computed on a 10,000-row
subsample (for example Table 1's counts, or Table 11's rare-event counts)
will not match the facts the report quotes if `--quick` was used for the
final run. Re-run all three scripts without `--quick` before taking any
number from `outputs/` into the report.

## Runtime (full runs, no `--quick`, measured on the development machine)

| Script | Runtime |
|---|---|
| `01_data_quality.py` | 19.9 seconds |
| `02_paradigm_evidence.py` | 37.3 seconds |
| `03_preprocessing_pipeline.py` | 742.3 seconds (about 12.4 minutes) |

`03_preprocessing_pipeline.py` is dominated by Table 15: a stratified
cross-validated comparison of 6 resampling strategies x 3 classifiers = 18
pipeline configurations. `common.CV_FOLDS = 5` (the textbook default) is used
throughout, including Tables 15, 16 and 17: the measured full run took about
12.4 minutes in total with 5 folds, comfortably inside the assignment's ~20
minute allowance for this step, so the fallback to 3 folds ("if the full run
would take more than about 20 minutes, reduce to 3 folds via a constant") is
not needed. Each script also prints its own elapsed time on completion, and
that figure is written to `key_numbers.json` as `script0N_runtime_seconds`.

## Output map

Everything is written under `outputs/`:

```
outputs/
  figures/            Fig01_....png to Fig16_....png (dpi 200, bbox tight)
  tables/              Table01_....csv to Table18_....csv
  figure_table_register.md   number, file, caption, task, script for every output
  key_numbers.json           every headline number the report would quote
  _register_state.csv        internal bookkeeping for the register (not a deliverable)
```

`outputs/figure_table_register.md` and `outputs/key_numbers.json` are
regenerated (merged, not wiped) every time any script runs; they are the
single source of truth for "which script produced this number/figure/table".

| Script | Produces |
|---|---|
| `01_data_quality.py` | Table 1, 2, 5 to 11; Figure 1, 2, 5 to 11 |
| `02_paradigm_evidence.py` | Table 3, 4; Figure 3, 4 |
| `03_preprocessing_pipeline.py` | Table 12 to 18; Figure 12 to 16 |

See `outputs/figure_table_register.md` after running the scripts for the
full list with captions, and `outputs/key_numbers.json` for every number in
machine-readable form.

## Design notes for the report writer

- `common.py` holds everything shared across scripts: data loading, the
  column-group / analytical-type map (id, target, numeric continuous,
  numeric count/average, binary Yes/No, nominal categorical, ordinal
  categorical), the placeholder map for disguised missing values
  (`PLACEHOLDER_RULES`), the plotting style, and the `save_fig` / `save_table`
  helpers. Kaggle provides no data dictionary for this dataset: every
  plain-English feature description in Table 2 is an interpretation of the
  column name and observed values, not an official definition; this is
  stated again in Table 2's own caption.
- `random_state = 42` everywhere a seed is accepted.
- Leakage safety: in `03_preprocessing_pipeline.py`, the stratified 80/20
  train/test split happens first. Every imputer, transformer, encoder,
  scaler, PCA and sampler used afterwards (Tables 12, 13, 14, 15, 17, 18) is
  fitted on the training split, or, inside cross-validation, on training
  folds only. Table 18 pre-declares two candidates from the Table 15 CV
  results ((a) best CV PR-AUC, (b) best CV F1 for Churn = Yes) before the
  test set is touched at all; each is fit once on the training split and the
  test split is used exactly once per candidate. Table 16 is the sole
  deliberate exception: a three-arm ablation, all three arms fitted on the
  same training rows with the same StratifiedKFold splits, that separates
  two leakage sources instead of blurring them into one. Arm A (WRONG) fits
  preprocessing, scaling and SMOTE once on the whole training split before
  cross-validating. Arm B fits preprocessing and scaling once beforehand,
  same as arm A, but refits SMOTE inside each fold. Arm C (CORRECT) refits
  preprocessing, scaling and SMOTE inside each fold, and is identical to the
  Table 15 SMOTE + RandomForest row. Arm A minus arm B isolates SMOTE's own
  timing effect; arm B minus arm C isolates preprocessing and scaling's own
  timing effect.
- Scaling is explicit and happens exactly once: `PowerTransformer` always
  uses `standardize=False` in the numeric branch of the preprocessing
  `ColumnTransformer` (`build_preprocessor` / `build_smotenc_pipeline` in
  `03_preprocessing_pipeline.py`), so that step never scales anything by
  itself. A single, separate `StandardScaler` step, added by the caller, is
  the only place any pipeline scales its features, so there is no hidden
  double scaling and each step has one job. Binary Yes/No flags (already
  mapped to 0/1) get their own branch: impute most-frequent, no power
  transform, since Yeo-Johnson has nothing meaningful to do to a 0/1 flag;
  they are still scaled later along with everything else by that one
  `StandardScaler` step, which is fine.
- Placeholder cleaning (`common.clean_placeholders`, called from
  `03_preprocessing_pipeline.pre_clean`) converts disguised-missing values
  (e.g. HandsetPrice "Unknown", AgeHH1/AgeHH2 zero, IncomeGroup zero,
  MaritalStatus "Unknown") to NaN using a fixed, row-independent rule before
  the train/test split. Every rule depends only on a row's own value in one
  column, never on a statistic computed across rows, so applying it before
  the split cannot leak information between the eventual training and test
  rows; see the comment on `pre_clean` in `03_preprocessing_pipeline.py`.
  Homeownership's "Unknown" and Occupation/PrizmCode's "Other" are
  deliberately *not* converted: they are documented as legitimate categories
  in Table 2 and Table 11, not Table 5, since they are not treated as
  missing (see `PLACEHOLDER_RULES` in `common.py`).
- `CustomerID` is dropped before any modelling step. The holdout file is
  never used for modelling; it has no labels.
- No em dashes or en dashes are used anywhere in the code, comments,
  captions or this README; commas, colons, parentheses or "to" are used
  instead.
