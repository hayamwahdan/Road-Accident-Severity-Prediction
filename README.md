# Road Accident Severity Prediction

**Predicting accident severity (Fatal / Serious / Slight) using UK road accident data (2021-2022)**

![UK Road Accident](https://img.shields.io/badge/Data-UK%20Road%20Safety-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## ## Project Overview

This repository contains an end-to-end data analysis and machine learning project that:

- Explores **UK road accident data** (~300,000+ records)
- Performs extensive **EDA** and feature engineering
- Builds a **Random Forest classifier** to predict accident severity
- Uses **SMOTE** to handle severe class imbalance (Fatal ~1%, Serious ~13%)
- Includes **scenario simulation** and a simple **risk alerting logic**
- Exports a trained model ready for deployment

Main notebook: **`Road_Accident_Analysis.ipynb`**

## ## Repository Contents

| File | Description |
| :--- | :--- |
| `Road_Accident_Analysis.ipynb` | Complete analysis, preprocessing, modeling & evaluation |
| `Road_Accident_Data_Cleaned.csv.gz` | The processed dataset ready for visualization tools |
| `road_accident_model.pkl` | Saved Random Forest model (SMOTE-balanced) |
| `severity_encoder.pkl` | LabelEncoder for mapping (0: Fatal, 1: Serious, 2: Slight) |
| `model_features.pkl` | List of feature columns required for model inference |

## ## Key Insights & Results

### 1. Performance
*   **Accuracy:** Achieved **~80%** overall accuracy.
*   **Class Balancing:** By applying SMOTE, the model significantly improved its ability to detect "Serious" and "Fatal" incidents compared to a baseline model.

### 2. Top Predictors
The model identified the following as the most influential factors in determining accident severity:
1.  **Spatial Coordinates (Lat/Long):** Specific high-risk geographic locations.
2.  **Speed Limit:** Higher speed zones (60-70mph) are the strongest indicators of fatal outcomes.
3.  **Hour of Day:** Peak accident times occur during afternoon commute hours (3 PM - 5 PM).

### 3. Environmental Paradox
Analysis revealed that the vast majority of accidents occur in **"Fine" weather** and **"Daylight"**. This suggests that driver complacency in good conditions is a higher risk factor than adverse weather.

## ## Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
