#####
#AUDIT FIX
#Issue 9: To improve readability and maintainability, hard-coded numerical constants were replaced with named configuration parameters provided in this file
####

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

MISSING_VALUE_THRESHOLD = 0.50
LOS_CLIP_QUANTILE = 0.99

LOGREG_MAX_ITER = 5000
LR_C_VALUES = [0.01, 0.1, 1, 10, 100]

RF_N_ESTIMATORS = [50, 100, 200]
RF_MAX_DEPTH = [None, 5, 10]

IQR_MULTIPLIER = 1.5

XGB_N_ESTIMATORS = [50, 100, 200]
XGB_MAX_DEPTH = [3, 5, 7]
MIN_RECALL_TARGET = 0.70
CV_REPEATS = 5
ONE_HOT_MAX_CARDINALITY = 20

DIAGNOSIS_MIN_COUNT = 3