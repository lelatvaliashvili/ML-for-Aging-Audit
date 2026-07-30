# Hospital Mortality Prediction using MIMIC Clinical Data

# Group Members:
Lela Tvaliashvili
Enise Irem Cokal
## Overview

This project implements a machine learning pipeline for predicting **in-hospital mortality** using patient records from the **MIMIC (Medical Information Mart for Intensive Care)** clinical database. The objective is to classify whether a patient will survive or die during a hospital admission based on demographic, admission, and intensive care unit (ICU) information.

The project demonstrates a complete supervised machine learning workflow, including data integration, exploratory data analysis (EDA), preprocessing, model training, hyperparameter optimization, and model evaluation.

---


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



##  Objectives

- Data integration from multiple relational tables
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Supervised binary classification
- Hyperparameter tuning
- Model comparison
- Performance evaluation using standard classification metrics
- End-to-end machine learning workflow in a healthcare setting