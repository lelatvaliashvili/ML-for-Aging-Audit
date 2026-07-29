import pandas as pd
import config

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
# Domain Validation (Issue #4 form the audit report)
# Audit Fix: Add validation of the dataset beofre preprocessing by checking temporal consistency
# and clinically meaningful constraints. This follows the EDA process discussed in the lecture.
# Therefore, EDA is followed by validation before cleaning
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

# Check that for all entries, ICU admission is before ICU discharge
invalid_icu = adm_pat_icu[
    adm_pat_icu["intime"] > adm_pat_icu["outtime"]
]

print("ICU discharge before ICU admission:", len(invalid_icu))

# Check for Negative length of stay
negative_los = adm_pat_icu[
    adm_pat_icu["los"] < 0
]

print("Negative LOS:", len(negative_los))

#age was not relevant cause MIMIC intentionally shifts dates for privacy, so deriving age this way is unreliable,
# therefore this check is intentionally omitted.
"""
adm_pat_icu["age"] = (
    adm_pat_icu["admittime"].dt.year
    - adm_pat_icu["dob"].dt.year
)"""
#end of issue 4 fix

#######
# Issue 2, Issue 8, Issue 10: Feature Selection, Target Leakage, Identifier Variables
# AUDIT FIX:
# Remove identifier variables and leakage features that reveal information unavailble during prediction time
#######

#remove the columns needed for domain validation only or convert them back to strings
validation_only_columns = [
    "admittime",
    "intime",
    "dob"
]

#adm_pat_icu = adm_pat_icu.drop(columns=validation_only_columns)

adm_pat_icu["admittime"] = adm_pat_icu["admittime"].astype(str)
adm_pat_icu["intime"] = adm_pat_icu["intime"].astype(str)
adm_pat_icu["dob"] = adm_pat_icu["dob"].astype(str)

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
    "subject_id",
    "hadm_id",
    "icustay_id",
    "row_id_x",
    "row_id_y",
    "row_id"
]


columns_to_drop = [
    col
    for col in leakage_columns + id_columns
    if col in adm_pat_icu.columns
]

adm_pat_icu.drop(columns = columns_to_drop, inplace=True)

#Split the raw data before preprocessing to prevent data leakage.
X = adm_pat_icu.drop(columns=["hospital_expire_flag"])
y = adm_pat_icu["hospital_expire_flag"]

from sklearn.model_selection import train_test_split

######
# Issue 1:  Data Preprocessing
# AUDIT FIX: Split data before preprocessing to prevent data leakage.
#####
# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config.TEST_SIZE, random_state=42, stratify=y
)

########
# Issue 8: Missing Value Handling
# AUDIT FIX: Determine missing value rate using only the training data. This ensures that the information from the test set does not influence the preprocessing decisions.
# Missingness pattern is inspected before imputation, variables such as emergency department timestamps (edregtime, edouttime) are likely related to observed clinical characteristics, while language, religion, etc are more likely missing at random.
# Features with more than 50% missing observations are removed to avoid relying on majority class. The same columns are then excluded from both the training and test sets for feature consistency.
########

missing_rate = X_train.isnull().mean().sort_values(ascending=False)

#AUDIT FIX: inspect missing values to make informed decision about dropping
print("missing value percentage: ")
print((missing_rate * 100).round(1))

# Drop columns with more than 50% missing data
threshold = config.MISSING_VALUE_THRESHOLD
cols_to_drop = missing_rate[missing_rate > threshold].index.tolist()

X_train = X_train.drop(columns=cols_to_drop)
X_test = X_test.drop(columns=cols_to_drop)

# Fill missing values in categorical variables with 'Unknown'
for col in X_train.select_dtypes(include='object').columns:
    X_train[col] = X_train[col].fillna('Unknown')
    X_test[col] = X_test[col].fillna("Unknown")

# Fill missing values in numeric variables with median
######
# AUDIT FIX: Data Leakage
# Learn median values from the training data only and use the same statistics to impute both datasets,
# preventing data leakage.
for col in X_train.select_dtypes(include='number').columns:
    median = X_train[col].median()
    X_train[col] = X_train[col].fillna(median)
    X_test[col] = X_test[col].fillna(median)


# Cap LOS at 99th percentile to handle extreme outliers
######
# AUDIT FIX: Data Leakage
# Learn the clipping threshold from the training data only and then apply the same threshold to the test data to avoid leakage.#

#inspect distribution

print(X_train["los"].describe())

#how many observations would be clipped
upper = X_train["los"].quantile(0.99)
n_outliers = (X_train["los"] > upper).sum()

#check for extreme values
plt.figure(figsize=(6,4))
plt.boxplot(X_train["los"], vert=False)
plt.xlabel("Length of Stay (days)")
plt.title("LOS Distribution Before Clipping")
plt.show()

print(f"99th percentile: {upper:.2f}")
print(f"Number of values above threshold: {n_outliers}")

#IQR rule
Q1 = X_train["los"].quantile(0.25)
Q3 = X_train["los"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"Upper bound: {upper:.2f}")

print("Outliers detected with IQR:",
      (X_train["los"] > upper).sum())

X_train["los"] = X_train["los"].clip(lower=lower, upper=upper)
X_test["los"] = X_test["los"].clip(lower=lower, upper=upper)


#los_cap = X_train['los'].quantile(config.LOS_CLIP_QUANTILE)
#X_train['los'] = X_train['los'].clip(upper=los_cap)
#X_test['los'] = X_test['los'].clip(upper=los_cap)


# List of categorical columns (excluding the target)
#categorical_cols = [col for col in adm_pat_icu_clean.select_dtypes(include='object').columns if col != 'hospital_expire_flag']

#AUDIT FIX: The target variable (hospital_expire_flag) has already been separated from the feature matrix.
# Therefore, if condition (if col != 'hospital_expire_flag') from original implementation is no longer necessary
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
# Apply one-hot encoding
#adm_pat_icu_encoded = pd.get_dummies(adm_pat_icu_clean, columns=categorical_cols, drop_first=True)

########
# AUDIT FIX:
# Perform one-hot encoding separately on the training
# and test data

X_train = pd.get_dummies(
    X_train,
    columns=categorical_cols,
    drop_first=True
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_cols,
    drop_first=True
)

#AUDIT FIX: Aligning encoded feature matrices ensures the training and test sets contain exactly the same features after one-hot encoding
X_train, X_test = X_train.align(
    X_test,
    join="left",
    axis=1,
    fill_value=0
)

from sklearn.preprocessing import StandardScaler

# AUDIT FIX: num_cols becomes obsolete as we removed identifier columns before splitting
# Select numeric columns (excluding target and IDs)
"""
num_cols = [col for col in adm_pat_icu_encoded.columns
            if adm_pat_icu_encoded[col].dtype != 'object'
            and col not in ['hospital_expire_flag', 'subject_id', 'hadm_id', 'icustay_id', 'row_id_x', 'row_id_y', 'row_id']]
"""
#AUDIT FIX: Identifier columns were removed earlier and are therefore excluded.
num_cols = X_train.select_dtypes(include='number').columns.tolist()

######
# AUDIT FIX:
# Learn scaling parameters from the training data only
# and apply the same transformation to the test data.
scaler = StandardScaler()
scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

#the line below is also obsolete
#adm_pat_icu_encoded[num_cols] = scaler.fit_transform(adm_pat_icu_encoded[num_cols])

# Remove features with zero variance or that are duplicates
nunique = X_train.nunique()
zero_var_cols = nunique[nunique <= 1].index.tolist()
X_train = X_train.drop(columns=zero_var_cols)
X_test= X_test.drop(columns=zero_var_cols)


#####
#Model Development
#
#AUDIT FIX overview:
# At this stage the dataset has been:
# - validated for data quality
# - stripped of leakage and identifier variables
# - split into training and testing sets
# - preprocessed using statistics learned only from the
#   training data to avoid data leakage.
# The processed training data are now used for
# hyperparameter tuning and model training.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Set up grid search for Logistic Regression
logreg = LogisticRegression(max_iter=config.LOGREG_MAX_ITER, random_state=42)
param_grid_lr = {'C': config.LR_C_VALUES }
grid_lr = GridSearchCV(logreg, param_grid_lr, cv=config.CV_FOLDS, scoring='roc_auc')
grid_lr.fit(X_train, y_train)

print("Best parameters (Logistic Regression):", grid_lr.best_params_)
logreg_best = grid_lr.best_estimator_

from sklearn.ensemble import RandomForestClassifier

# Set up grid search for Random Forest
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': config.RF_N_ESTIMATORS,
    'max_depth': config.RF_MAX_DEPTH
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=config.CV_FOLDS, scoring='roc_auc')
grid_rf.fit(X_train, y_train)

print("Best parameters (Random Forest):", grid_rf.best_params_)
rf_best = grid_rf.best_estimator_

import xgboost as xgb

xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
param_grid_xgb = {
    'n_estimators': config.XGB_N_ESTIMATORS,
    'max_depth': config.XGB_MAX_DEPTH
}
grid_xgb = GridSearchCV(xgb_clf, param_grid_xgb, cv=config.CV_FOLDS, scoring='roc_auc')
grid_xgb.fit(X_train, y_train)

print("Best parameters (XGBoost):", grid_xgb.best_params_)
xgb_best = grid_xgb.best_estimator_

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve

models = {
    'Logistic Regression': logreg_best,
    'Random Forest': rf_best,
    'XGBoost': xgb_best
}

results = {}

for name, model in models.items():
    print(f"Evaluating {name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    print("Probabilities done. Metrics:")
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    print(f"\n{name}:")
    print(results[name])

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix: {name}')
    plt.show(block=False)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC={results[name]["roc_auc"]:.2f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (All Models)')
plt.legend()
plt.show()

