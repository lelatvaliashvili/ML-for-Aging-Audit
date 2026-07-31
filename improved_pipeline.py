import pandas as pd
import numpy as np
import config

#Issue 9: one global seed so the whole run is reproducible, not just the pieces that already had random_state
np.random.seed(config.RANDOM_STATE)

# Read core tables
patients = pd.read_csv('mimic_demo/PATIENTS.csv')
admissions = pd.read_csv('mimic_demo/ADMISSIONS.csv')
icustays = pd.read_csv('mimic_demo/ICUSTAYS.csv')

# Preview the shape and first rows of each table
print("PATIENTS.csv shape:", patients.shape)
print(patients.head())

print("\nADMISSIONS.csv shape:", admissions.shape)
print(admissions.head())

print("\nICUSTAYS.csv shape:", icustays.shape)
print(icustays.head())

# Merge patients and admissions on 'subject_id'
adm_pat = pd.merge(admissions, patients, on='subject_id', how='left')

# Merge the result with icustays on ['subject_id', 'hadm_id']
adm_pat_icu = pd.merge(adm_pat, icustays, on=['subject_id', 'hadm_id'], how='left')

# Check the shape and sample rows of the merged table
print("Merged table shape:", adm_pat_icu.shape)
print(adm_pat_icu.head())

# Check the distribution of the target variable
print("\nHospital Expire Flag distribution:")
print(adm_pat_icu['hospital_expire_flag'].value_counts())

# Calculate missing value rate for each column
missing_rate = adm_pat_icu.isnull().mean().sort_values(ascending=False)
print("Missing value rate per column:\n", missing_rate)

# Separate numeric and categorical columns
numeric_cols = adm_pat_icu.select_dtypes(include=['number']).columns.tolist()
categorical_cols = adm_pat_icu.select_dtypes(include=['object']).columns.tolist()

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Describe numeric variables
print("Descriptive statistics for numeric variables:")
print(adm_pat_icu[numeric_cols].describe())

# Frequency count for categorical variables (top 3 for each)
for col in categorical_cols:
    print(f"\nValue counts for categorical variable '{col}':")
    print(adm_pat_icu[col].value_counts(dropna=False).head(3))

import matplotlib.pyplot as plt


# Plot distribution of the target variable
adm_pat_icu['hospital_expire_flag'].value_counts().plot(kind='bar')
plt.title('Distribution of Hospital Expire Flag')
plt.xlabel('Hospital Expire Flag (0=Survived, 1=Died)')
plt.ylabel('Count')
plt.show()

# Plot gender distribution
adm_pat_icu['gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Plot admission_type distribution
adm_pat_icu['admission_type'].value_counts().plot(kind='bar')
plt.title('Admission Type Distribution')
plt.xlabel('Admission Type')
plt.ylabel('Count')
plt.show()

# Plot insurance distribution
adm_pat_icu['insurance'].value_counts().plot(kind='bar')
plt.title('Insurance Distribution')
plt.xlabel('Insurance')
plt.ylabel('Count')
plt.show()

# Plot ethnicity distribution
adm_pat_icu['ethnicity'].value_counts().plot(kind='bar')
plt.title('Ethnicity Distribution')
plt.xlabel('Ethnicity')
plt.ylabel('Count')
plt.show()

# Histogram for length of stay (los)
adm_pat_icu['los'].plot(kind='hist', bins=20)
plt.title('Distribution of Length of Stay (LOS)')
plt.xlabel('Length of Stay (days)')
plt.ylabel('Frequency')
plt.show()

# Boxplot for LOS by hospital_expire_flag
adm_pat_icu.boxplot(column='los', by='hospital_expire_flag')
plt.title('LOS by Hospital Expire Flag')
plt.xlabel('Hospital Expire Flag (0=Survived, 1=Died)')
plt.ylabel('Length of Stay (days)')
plt.suptitle('')
plt.show()

# Death rate by admission type
death_by_admtype = adm_pat_icu.groupby('admission_type')['hospital_expire_flag'].mean()
death_by_admtype.plot(kind='bar')
plt.title('Death Rate by Admission Type')
plt.xlabel('Admission Type')
plt.ylabel('Death Rate')
plt.show()

# Death rate by insurance
death_by_insurance = adm_pat_icu.groupby('insurance')['hospital_expire_flag'].mean()
death_by_insurance.plot(kind='bar')
plt.title('Death Rate by Insurance')
plt.xlabel('Insurance')
plt.ylabel('Death Rate')
plt.show()



#######
# Issue 4: Domain Validation
# Audit Fix:
# Add validation of the dataset before preprocessing by checking temporal consistency
# and clinically meaningful constraints such as admission before discharge, ICU admission before ICU discharge, non-negative LOS values.
# Therefore, EDA is followed by validation before cleaning
# Age was not relevant as MIMIC intentionally shifts dates for privacy, so deriving age this way is unreliable,
# therefore ensuring plausible age range is intentionally omitted even though initial audit report included checking realistic age.
#######
date_columns= [
    'admittime', 'dischtime',
    'intime', 'outtime',
    'dob', 'dod', 'deathtime'
]

for col in date_columns:
    if col in adm_pat_icu.columns:
        adm_pat_icu[col] = pd.to_datetime(adm_pat_icu[col])

#Check that for all entries, patient admission occurs before discharge for all records
invalid_admissions = adm_pat_icu[
    adm_pat_icu["admittime"] > adm_pat_icu["dischtime"]
]

print("number of invalid dates for admission and discharge: Admissions after discharge:", len(invalid_admissions))

#Ensure only valid records remain
adm_pat_icu = adm_pat_icu[
    adm_pat_icu["admittime"] <= adm_pat_icu["dischtime"]
]

# Check that for all entries, ICU admission is before ICU discharge
invalid_icu = adm_pat_icu[
    adm_pat_icu["intime"] > adm_pat_icu["outtime"]
]

print("ICU discharge before ICU admission:", len(invalid_icu))

#Ensure only valid records remain
adm_pat_icu = adm_pat_icu[
    adm_pat_icu["intime"] <= adm_pat_icu["outtime"]]

# Check for Negative length of stay
negative_los = adm_pat_icu[
    adm_pat_icu["los"] < 0
]


print("Negative LOS:", len(negative_los))

#Ensure only valid records remain
adm_pat_icu = adm_pat_icu[
    adm_pat_icu["los"] >= 0
]

########
# The current dataset does not contain records that violate these constraints, but this step ensures
# that logically inconsistent observations are excluded if they are encountered in future datasets.
#######

#######
# Audit Fix:
# Issue 2, Issue 9: Target Leakage, Feature Selection, Identifier Variables
# Remove variables that would not be available at prediction time
# (target leakage) together with identifier attributes that uniquely
# identify patients or admissions but do not contain predictive
# clinical information.
#######

#remove the columns needed for domain validation only or convert them back to strings
validation_only_columns = [
    "admittime",
    "intime",
    "dob"
]

#adm_pat_icu = adm_pat_icu.drop(columns=validation_only_columns)

#adm_pat_icu["admittime"] = adm_pat_icu["admittime"].astype(str)
#adm_pat_icu["intime"] = adm_pat_icu["intime"].astype(str)
#adm_pat_icu["dob"] = adm_pat_icu["dob"].astype(str)

leakage_columns = [
    "dod",
    "dod_hosp",
    "dod_ssn",
    "deathtime",
    "dischtime",
    "outtime",
    "discharge_location"
]

id_columns = [
    "hadm_id",
    "icustay_id",
    "row_id_x",
    "row_id_y",
    "row_id"
]

#Issue 3: subject_id stays for now, GroupShuffleSplit needs it; dropped from features after the split.
columns_to_drop = [
    col
    for col in leakage_columns + id_columns
    if col in adm_pat_icu.columns
]

adm_pat_icu.drop(columns = columns_to_drop, inplace=True)

#Split the raw data before preprocessing to prevent data leakage.
X = adm_pat_icu.drop(columns=["hospital_expire_flag"])
y = adm_pat_icu["hospital_expire_flag"]

from sklearn.model_selection import GroupShuffleSplit

######
# Issue 1: Split data before preprocessing to prevent data leakage.
# Issue 3: split by subject_id, not by row, so a patient can't end up in both sets.
# Tradeoff: GroupShuffleSplit can't also stratify by y like train_test_split did.
# so the death rate may differ by chance, repeated StratifiedGroupKFold gives a more balanced and reliable evaluation.

#####
gss = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=X["subject_id"]))

#kept for the repeated group CV in the evaluation section further down
groups_train = X.iloc[train_idx]["subject_id"]
groups_test = X.iloc[test_idx]["subject_id"]

X_train = X.iloc[train_idx].drop(columns=["subject_id"])
X_test = X.iloc[test_idx].drop(columns=["subject_id"])
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

########
# Fill missing values in numeric variables with median
# Issue 6: Missing Value Handling
# Audit Fix:
# A missingness pattern is inspected before choosing an imputation strategy. We do not assume missing all missing values occur randomly.
# edregtime/edouttime are blank for every ELECTIVE admission (no ED visit), so missingness itself is a signal. That is why missingness indicators are created from these variables.
# For variables such as language and religion, no similar pattern was identified. Therefore, they are assumed to be missing at random based on general domain knowledge
# rather than a formal MCAR/MAR/MNAR analysis, given the dataset is small.
# Other columns exceeding the missing-value threshold are removed as they contain insufficient information for modeling reliably. The same columns are then excluded from both the training and test sets for feature consistency
########

missing_rate = X_train.isnull().mean().sort_values(ascending=False)

#inspect missing values to make informed decision about dropping
print("missing value percentage: ")
print((missing_rate * 100).round(1))

# Drop columns with more than 50% missing data
threshold = config.MISSING_VALUE_THRESHOLD
cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()

X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

for col in ["edregtime", "edouttime"]:
    if col in X_train.columns:
        X_train[f"{col}_missing"] = X_train[col].isna().astype(int)
        X_test[f"{col}_missing"] = X_test[col].isna().astype(int)

#######
# Audit Fix:
# Issue 5: Outlier Analysis
# Inspecting outliers with IQR rule and comparing it with existing 0.99th percentile clipping
# demonstrated that 14 patients (around 10% of the dataset) get marked as outliers with IQR, while many correspond to plausible long stays due to clinical reasons.
# As the 99th percentile only clips 2 extreme samples which motivated moving towards retaining all observations. Therefore, IQR is computed
# solely for exploration purposes. This further motivated to apply log1p transformation for reducing skewness instead of removing or clipping the values. Since it does not have fitted parameters,
# log1p transform can be applied to both train and test sets without resulting in data leakage.
#######

#inspect distribution
print(X_train["los"].describe())

#For 99th percentile, check how many observations would be clipped
upper_pct = X_train["los"].quantile(config.LOS_CLIP_QUANTILE)
n_outliers_pct = (X_train["los"] > upper_pct).sum()

#check for extreme values
plt.figure(figsize=(6,4))
plt.boxplot(X_train["los"], vert=False)
plt.xlabel("Length of Stay (days)")
plt.title("LOS Distribution Before Clipping")
plt.show()

print(f"99th percentile: {upper_pct:.2f}")
print(f"Number of values above 99th percentile: {n_outliers_pct}")

#IQR rule, for exploration only
Q1 = X_train["los"].quantile(0.25)
Q3 = X_train["los"].quantile(0.75)

IQR = Q3 - Q1

lower_iqr = Q1 - config.IQR_MULTIPLIER * IQR
upper_iqr = Q3 + config.IQR_MULTIPLIER * IQR

print(f"IQR upper bound: {upper_iqr:.2f}")
print("Outliers detected with IQR:",
      (X_train["los"] > upper_iqr).sum())

X_train['los'] = np.log1p(X_train['los'])
X_test['los'] = np.log1p(X_test['los'])

#Issue 1: imputer, scaler and encoder now live inside a Pipeline (below) instead of
# being fit on all of X_train up front, so GridSearchCV refits them per CV fold
# Identifier columns were removed earlier and are therefore excluded.
num_cols = X_train.select_dtypes(include='number').columns.tolist()

#skip one-hot encoding for high-cardinality columns (e.g. raw admittime/intime/dob
# strings) so they don't blow up into hundreds of dummy columns
categorical_cols_all = X_train.select_dtypes(include='object').columns.tolist()
high_cardinality_cols = [
    col for col in categorical_cols_all
    if X_train[col].nunique() > config.ONE_HOT_MAX_CARDINALITY
]
categorical_cols = [col for col in categorical_cols_all if col not in high_cardinality_cols]

print("High-cardinality columns excluded from one-hot encoding:", high_cardinality_cols)

X_train = X_train.drop(columns=high_cardinality_cols)
X_test = X_test.drop(columns=high_cardinality_cols)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder


def make_preprocessor():
    #Issue 5: RobustScaler (median/IQR) instead of StandardScaler, less thrown off by outliers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


#####

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#Class imbalance (69/31 survived/died): class_weight="balanced" so the minority class isn't ignored.
#SMOTE was considered too, but skipped. 136 rows is small to synthesize minority examples from, and it needs an extra imblearn
# dependency for what class_weight/scale_pos_weight already handle here.
# Set up grid search for Logistic Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', make_preprocessor()),
    ('classifier', LogisticRegression(max_iter=config.LOGREG_MAX_ITER, random_state=config.RANDOM_STATE, class_weight='balanced'))
])
param_grid_lr = {'classifier__C': config.LR_C_VALUES}
grid_lr = GridSearchCV(lr_pipeline, param_grid_lr, cv=config.CV_FOLDS, scoring='roc_auc')
grid_lr.fit(X_train, y_train)

print("Best parameters (Logistic Regression):", grid_lr.best_params_)
logreg_best = grid_lr.best_estimator_

from sklearn.ensemble import RandomForestClassifier

# Set up grid search for Random Forest
rf_pipeline = Pipeline(steps=[
    ('preprocessor', make_preprocessor()),
    ('classifier', RandomForestClassifier(random_state=config.RANDOM_STATE, class_weight='balanced'))
])
param_grid_rf = {
    'classifier__n_estimators': config.RF_N_ESTIMATORS,
    'classifier__max_depth': config.RF_MAX_DEPTH
}
grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=config.CV_FOLDS, scoring='roc_auc')
grid_rf.fit(X_train, y_train)

print("Best parameters (Random Forest):", grid_rf.best_params_)
rf_best = grid_rf.best_estimator_

import xgboost as xgb

#XGBoost's equivalent of class_weight is scale_pos_weight: negative/positive count ratio
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_pipeline = Pipeline(steps=[
    ('preprocessor', make_preprocessor()),
    ('classifier', xgb.XGBClassifier(random_state=config.RANDOM_STATE, eval_metric='logloss', scale_pos_weight=scale_pos_weight))
])
param_grid_xgb = {
    'classifier__n_estimators': config.XGB_N_ESTIMATORS,
    'classifier__max_depth': config.XGB_MAX_DEPTH
}
grid_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, cv=config.CV_FOLDS, scoring='roc_auc')
grid_xgb.fit(X_train, y_train)

print("Best parameters (XGBoost):", grid_xgb.best_params_)
xgb_best = grid_xgb.best_estimator_

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, make_scorer
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

#majority-class baseline, so the real models have something to beat
baseline = DummyClassifier(strategy='most_frequent', random_state=config.RANDOM_STATE)
baseline.fit(X_train, y_train)

models = {
    'Baseline (Majority Class)': baseline,
    'Logistic Regression': logreg_best,
    'Random Forest': rf_best,
    'XGBoost': xgb_best
}

#Issue 7: test set is small, so re-run grouped stratified CV a few times over the
# whole dataset and average each metric instead of trusting one split. Reusing
# X_test is fine here, GridSearchCV above already tuned using X_train only.
X_full = pd.concat([X_train, X_test])
y_full = pd.concat([y_train, y_test])
groups_full = pd.concat([groups_train, groups_test])

#note: precision/recall/f1 here use each fold's default 0.5 threshold, not the
# per-model threshold picked below
CV_SCORING = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'f1': make_scorer(f1_score, zero_division=0),
    'roc_auc': 'roc_auc'
}


def repeated_cv_scores(pipeline, n_repeats=config.CV_REPEATS):
    scores = {metric: [] for metric in CV_SCORING}
    for repeat in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE + repeat)
        cv_result = cross_validate(pipeline, X_full, y_full, groups=groups_full, cv=cv, scoring=CV_SCORING)
        for metric in CV_SCORING:
            scores[metric].extend(cv_result[f'test_{metric}'])
    return {metric: (np.mean(vals), np.std(vals)) for metric, vals in scores.items()}


results = {}

#Plotting bug fix: ConfusionMatrixDisplay.plot() creates and activates a new
# figure each call, so they were on top of each other and only the last one was visible. 
# Dedicated figures/axes created up front and plotted onto explicitly fixes both.
n_models = len(models)
n_cols = 2
n_rows = -(-n_models // n_cols)  # ceil division
cm_fig, cm_axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.2 * n_rows))
cm_axes = cm_axes.ravel()

roc_fig, roc_ax = plt.subplots(figsize=(6, 6))

for i, (name, model) in enumerate(models.items()):
    print(f"Evaluating {name}...")
    y_proba = model.predict_proba(X_test)[:,1]

    #Issue 7: pick a threshold from the precision-recall curve hitting at least
    # config.MIN_RECALL_TARGET recall, instead of silently defaulting to 0.5.
    if name == 'Baseline (Majority Class)':
        threshold = 0.5
    else:
        precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
        candidates = [i for i in range(len(pr_thresholds)) if recalls[i] >= config.MIN_RECALL_TARGET]
        threshold = pr_thresholds[max(candidates, key=lambda i: precisions[i])] if candidates else 0.5

    y_pred = (y_proba >= threshold).astype(int)

    if name == 'Baseline (Majority Class)':
        cv_scores = {metric: (np.nan, np.nan) for metric in CV_SCORING}
    else:
        cv_scores = repeated_cv_scores(model)

    print("Probabilities done. Metrics:")
    results[name] = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc_test': roc_auc_score(y_test, y_proba),
    }
    for metric, (mean, std) in cv_scores.items():
        results[name][f'{metric}_cv_mean'] = mean
        results[name][f'{metric}_cv_std'] = std
    print(f"\n{name}:")
    print(results[name])

    # Confusion matrix, plotted onto its own slot in the shared grid
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=cm_axes[i], colorbar=False)
    r = results[name]
    cm_axes[i].set_title(
        f"{name} (threshold={threshold:.2f})\n"
        f"Acc={r['accuracy']:.2f}  Prec={r['precision']:.2f}  "
        f"Rec={r['recall']:.2f}  F1={r['f1']:.2f}  AUC={r['roc_auc_test']:.2f}",
        fontsize=10,
    )

    # ROC curve, plotted onto the dedicated shared ROC axes
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_ax.plot(fpr, tpr, label=f'{name} (AUC={results[name]["roc_auc_test"]:.2f})')

# hide any unused confusion-matrix subplot slots (e.g. odd number of models)
for ax in cm_axes[n_models:]:
    ax.axis("off")
cm_fig.suptitle('Confusion Matrices — All Models', fontsize=14)
cm_fig.tight_layout()

roc_ax.plot([0,1], [0,1], 'k--')
roc_ax.set_xlabel('False Positive Rate')
roc_ax.set_ylabel('True Positive Rate')
roc_ax.set_title('ROC Curve (All Models)')
roc_ax.legend()
plt.show()

