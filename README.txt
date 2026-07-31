Project: Improved ML Pipeline for Hospital Mortality Prediction
Final Project: Code Auditing and Improvement

Lela Tvaliashvili - 5597957 - MSc. Data Science @ FU
Enise Irem Colak - 5593394 - MSc. Data Science @ FU

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
config.py: Configuration constants for pipeline
README.txt

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



CONTRIBUTIONS
=============
Group members: Lela, Enise
+--------+---------------------------------+------------------------------------------------------------+
| Name   | Section                         | Contribution                                                |
+--------+---------------------------------+------------------------------------------------------------+
| Lela   | Task & Dataset Understanding    | Explored the three tables, checked shapes and column        
|        |                                 | meanings, identified the potential pitfalls and merge structure
+--------+---------------------------------+------------------------------------------------------------+
| Enise  | Task & Dataset Understanding    | Reviewed for consistency, rearranged wording to fit the      
|        |                                 | word limit                                                    
+--------+---------------------------------+------------------------------------------------------------+
+--------+---------------------------------+------------------------------------------------------------+
| Lela   | Issue Identification            |Reviewed for consistency, Exploration of Issues 1,2,4,5,8,9                                                            
+--------+---------------------------------+------------------------------------------------------------+
| Enise  | Issue Identification            |Reviewed for consistency, Exploration of Issues 1,2,3,6,7                                                              
+--------+---------------------------------+------------------------------------------------------------+
+--------+---------------------------------+------------------------------------------------------------+
| Lela   | Methodological Critique         | Reviewed for consistency, rearranged wording to fit the      
|        |                                 | word limit                                                                
+--------+---------------------------------+------------------------------------------------------------+
| Enise  | Methodological Critique         | Explored through the pipeline to identify methodological choices 
|                                          | and did go through to the relevant lecture material                                                           
+--------+---------------------------------+------------------------------------------------------------+
+--------+---------------------------------+------------------------------------------------------------+
| Lela   | Improvement Plan                | Created improvement plan based on the issues
+--------+---------------------------------+------------------------------------------------------------+
| Enise  | Improvement Plan                | Reviewed for consistency, rearranged wording to fit the      
|        |                                 | word limit                                                            
+--------+---------------------------------+------------------------------------------------------------+
+--------+---------------------------------+------------------------------------------------------------+
| Lela   | Improved Implementation         | Moved preprocessing after the train/test split and Redesigned
|                                          | reprocessing workflow (Issue 1), removed identifier variables (Issue 8),
|                                          | removed target leakage (Issue 2), replaced hard-coded constants in pipeline
|                                          | (Issue 9),added domain validation (Issue 4), added feature alignment after
|                                          | one-hot encoding,Performed exploratory comparison of outlier handling strategies
+--------+---------------------------------+------------------------------------------------------------+
| Enise  | Improved Implementation         | Patient-level split (Issue 3), ED flags for missing value
|                                          | handling (Issue 6), RobustScaler (related Issue 5), 
|                                          | Cardinality-aware encoding (data-quality fix),
|                                          | Leakage fix (Issue 1), class imbalance handling (Issue 7), 
|                                          | Outlier handling (Issue 5), corrected comparison plots, the README.
|                                          | evaluation fixes: a baseline model, repeated cross-validation, 
|                                          | an classification threshold, and a global random seed.                                                         
+--------+---------------------------------+------------------------------------------------------------+
+--------+---------------------------------+------------------------------------------------------------+
| Lela   | Results Summary and Reflection  | Explained the overall results, added Tables and issue related explanation
+--------+---------------------------------+------------------------------------------------------------+
| Enise  | Results Summary and Reflection  | Provided results table and graphics, reviewed for consistency                                                               
+--------+---------------------------------+------------------------------------------------------------+
