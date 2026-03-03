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
3. Run the notebook

Open Road_Accident_Analysis.ipynb
Update the file path in Chapter 2 if your data is not located at:Pythondf = pd.read_excel(r"D:\Hayam Wahdan\Road Accident Data.xlsx")
Run all cells sequentially

4. Expected outputs

Many visualizations (time-of-day, weather, road type, severity heatmaps…)
Feature importance plot
Confusion matrices (before & after SMOTE)
Cross-validation scores (~85% mean accuracy)
Saved model files (*.pkl)
Compressed cleaned dataset

Reproducing the cleaned dataset
If you only have the original .xlsx file:
Bashjupyter nbconvert --to notebook --execute Road_Accident_Analysis.ipynb
The last-but-one cell exports Road_Accident_Data_Cleaned.csv.gz
Model Usage Example (Inference)
Pythonimport joblib
import pandas as pd

rf = joblib.load('road_accident_model.pkl')
le = joblib.load('severity_encoder.pkl')
features = joblib.load('model_features.pkl')

# Create a single-row DataFrame with the same columns & order
new_data = pd.DataFrame({...}, columns=features)

pred = rf.predict(new_data)
prob = rf.predict_proba(new_data)

print("Predicted severity:", le.inverse_transform(pred)[0])
print("Probabilities:", dict(zip(le.classes_, prob[0])))
Future Work Ideas

Try gradient boosting (XGBoost / LightGBM / CatBoost)
Add spatial analysis (clustering, hotspot detection)
Build a simple Streamlit / Gradio web interface
Include vehicle manoeuvres & driver age (if more detailed data becomes available)

License
MIT License
Feel free to use the code, model and cleaned data for learning, research or personal projects.
Happy & safe driving! 🛣️

Last major update: March 2025
