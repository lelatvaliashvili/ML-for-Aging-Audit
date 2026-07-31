Project: Improved ML Pipeline for Hospital Mortality Prediction
Final Project: Code Auditing and Improvement

Lela Tvaliashvili - - MSc. Data Science @ FU
Enise Irem Colak - - MSc. Data Science @ FU

Description: 
This project audits and improves a flawed machine learning pipeline that predicts
 in-hospital mortality (hospital_expire_flag) using the dataset (PATIENTS, ADMISSIONS, ICUSTAYS). 
 The original pipeline contained several critical and major issues, including data leakage, non-independent 
 train/test splits, unhandled class imbalance, and improper preprocessing order. These are documented 
 in audit_report.pdf, along with the fixes applied in improved_pipeline.py.

Files included:
audit_report.pdf: full audit report (Part A) and results summary/reflection (Part C)
improved_pipeline.py: corrected pipeline (Part B)
requirements.txt: required Python packages
README.txt: this file

How to run:

Install the required packages:
pip install -r requirements.txt
Make sure the mimic_demo folder (containing PATIENTS.csv, ADMISSIONS.csv, ICUSTAYS.csv) is in the same directory as improved_pipeline.py.

Run the script:
python improved_pipeline.py

The script runs end-to-end without modification and prints intermediate outputs 
(data shapes, missing value analysis, outlier exploration, model results) to the console, 
and displays plots (EDA charts, confusion matrices, ROC curve).

Notes:

config.py holds all configurable constants (random seed, thresholds, hyperparameter grids) 
used by improved_pipeline.py, and must be kept in the same folder.
Random seed is fixed (see config.py) so results are reproducible.