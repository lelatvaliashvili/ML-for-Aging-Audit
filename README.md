# Hospital Mortality Prediction using MIMIC Clinical Data

## Overview

This project implements a machine learning pipeline for predicting **in-hospital mortality** using patient records from the **MIMIC (Medical Information Mart for Intensive Care)** clinical database. The objective is to classify whether a patient will survive or die during a hospital admission based on demographic, admission, and intensive care unit (ICU) information.

The project demonstrates a complete supervised machine learning workflow, including data integration, exploratory data analysis (EDA), preprocessing, model training, hyperparameter optimization, and model evaluation.

---

## Problem Statement

Early prediction of hospital mortality can assist healthcare professionals in identifying high-risk patients, supporting clinical decision-making, optimizing resource allocation, and improving patient care.

Given patient information collected during hospital admission and ICU stay, the goal is to predict the binary outcome:

- **0** – Patient survived the hospital admission
- **1** – Patient died during the hospital admission

This is a **supervised binary classification** problem.

---

## Dataset

The project uses three relational tables from the MIMIC clinical database.

### PATIENTS

Contains patient demographic and mortality information, including:

- Patient ID
- Gender
- Date of birth
- Date of death
- Mortality indicators

### ADMISSIONS

Contains hospital admission information, including:

- Admission and discharge times
- Admission type
- Insurance
- Ethnicity
- Diagnosis
- Hospital mortality label (`hospital_expire_flag`)

### ICUSTAYS

Contains ICU-related information, including:

- ICU stay identifier
- Care unit
- ICU admission and discharge times
- Length of stay (LOS)

The three datasets are merged using the common identifiers `subject_id` and `hadm_id` to create a single dataset suitable for machine learning.

---

## Machine Learning Pipeline

The implemented workflow consists of the following steps:

1. Load clinical datasets
2. Merge patient, admission, and ICU records
3. Perform exploratory data analysis (EDA)
4. Handle missing values
5. Encode categorical variables
6. Scale numerical features
7. Split data into training and testing sets
8. Train multiple machine learning models
9. Perform hyperparameter optimization using Grid Search with cross-validation
10. Evaluate model performance

---

## Models

The following supervised learning algorithms are implemented:

- Logistic Regression
- Random Forest
- XGBoost

Hyperparameters are optimized using **GridSearchCV**.

---

## Evaluation Metrics

Model performance is evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The project also visualizes:

- Confusion Matrices
- ROC Curves
- Target class distribution
- Feature distributions
- Length of Stay (LOS) analysis

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost

---

## Project Structure

```text
.
├── mimic_demo/
│   ├── PATIENTS.csv
│   ├── ADMISSIONS.csv
│   └── ICUSTAYS.csv
│
├── HospitalMortalityPrediction.py
├── requirements.txt
└── README.md
```

---

##  Objectives

- Data integration from multiple relational tables
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Supervised binary classification
- Hyperparameter tuning
- Model comparison
- Performance evaluation using standard classification metrics
- End-to-end machine learning workflow in a healthcare setting