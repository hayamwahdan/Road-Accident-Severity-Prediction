🚗 Road Accident Severity Prediction & Risk Analysis (A to Z)
📌 Project Overview
This project provides a comprehensive Spatio-Temporal analysis and a predictive Machine Learning model for road accidents. Using a massive dataset of over 300,000 historical records, the project identifies high-risk road profiles and predicts accident severity (Slight, Serious, or Fatal) based on environmental, infrastructure, and temporal factors.
📊 Key Insights from EDA
Complacency Factor: Over 80% of accidents occur during "Fine" weather conditions, suggesting that driver تهور (recklessness) increases when conditions appear safe.
Temporal Peak: Accident frequency peaks significantly between 3:00 PM and 5:00 PM (Rush Hour).
Speed Impact: While low-speed zones have higher accident volumes, 60 mph and 70 mph zones exponentially increase the probability of Fatal and Serious outcomes.
🧠 Machine Learning & Model Performance
The project utilizes a Random Forest Multi-Classifier.
Class Imbalance Correction: Because fatal accidents represent only ~1.2% of the data, I implemented SMOTE (Synthetic Minority Over-sampling Technique).
Results: The SMOTE-enhanced model achieved an overall accuracy of 80% and tripled the recall for fatal incidents compared to the baseline model.
Stability: A 5-fold Cross-Validation confirmed a stable mean accuracy of 85.4%.
📂 Data Handling & Usage Instruction
The primary dataset has been cleaned and compressed to ensure it fits within GitHub's file limits while maintaining all original attributes.
⚠️ IMPORTANT NOTE FOR USERS:
If you download this project, you do not need to manually unzip or extract the data. You should use the Python code from the method below to open the file. Pandas handles the decompression automatically in your computer's memory.
code
Python
import pandas as pd

# Open the compressed dataset directly
df = pd.read_csv('Road_Accident_Data_Cleaned.csv.gz')

# View the first 5 rows
print(df.head())
🛠️ Project Roadmap (A to Z)
Data Acquisition: Loaded 307k+ rows from Excel (.xlsx).
Preprocessing: Handled underscores, missing values (98% missingness in hazards), and mitigated data leakage.
Feature Engineering: Extracted Hour using string splitting and performed One-Hot Encoding (62 total features).
EDA: Visualized risk drivers using Log-Scales and automated Data Labels.
Modeling: Built a Random Forest Classifier with class_weight='balanced'.
Optimization: Applied SMOTE for minority class augmentation.
System Design: Developed an Automated Risk Warning System based on probability thresholds.
⚙️ Requirements
To run this project, you will need the following libraries:
pandas
numpy
seaborn
matplotlib
scikit-learn
imbalanced-learn
openpyxl (for reading the original Excel source)
👤 Author
Hayam Wahdan — BI & Data Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/hayamwahdan)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:hayamm.wahdan@gmail.com)

--
