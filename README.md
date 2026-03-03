# Road Accident Severity Prediction

**Predicting accident severity (Fatal / Serious / Slight) using UK road accident data (2021–2022)**

![UK Road Accident](https://img.shields.io/badge/Data-UK%20Road%20Safety-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Project Overview

This repository contains an end-to-end data analysis and machine learning project that:

- Explores **UK road accident data** (~300,000+ records)
- Performs extensive **EDA** and feature engineering
- Builds a **Random Forest classifier** to predict accident severity
- Uses **SMOTE** to handle severe class imbalance (Fatal ~1%, Serious ~13%)
- Includes **scenario simulation** and a simple **risk alerting logic**
- Exports a trained model ready for deployment

Main notebook: **`Road_Accident_Analysis.ipynb`**

## Repository Contents

| File                                      | Description                                      |
|-------------------------------------------|--------------------------------------------------|
| `Road_Accident_Analysis.ipynb`            | Complete analysis, modeling & evaluation         |
| `Road_Accident_Data_Cleaned.csv.gz`       | Cleaned & preprocessed dataset (compressed)      |
| `road_accident_model.pkl`                 | Trained Random Forest model (after SMOTE)        |
| `severity_encoder.pkl`                    | Label encoder for severity classes               |
| `model_features.pkl`                      | List of features used by the model               |
| `README.md`                               | This file                                        |

## Key Findings (from the notebook)

- Infrastructure factors (**speed limit**, **road type**) are more predictive of severity than weather or light conditions
- Most accidents happen in **fine weather** → possible **driver complacency** effect
- **60–70 mph** zones show the highest proportion of **Serious** and **Fatal** outcomes
- SMOTE significantly improves recall for the minority classes (Serious & Fatal)

Final model performance (SMOTE version):

- Overall accuracy ≈ **80–85%**
- Much better detection of **Serious** cases compared to the imbalanced baseline

## Getting Started

### 1. Prerequisites

- Python 3.10+
- Jupyter Notebook / JupyterLab

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib openpyxl
