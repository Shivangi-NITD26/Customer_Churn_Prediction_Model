# Customer Churn Prediction Model

This machine learning project aims to predict whether a customer will leave (churn) a bank based on their demographic and account-related features. Accurately identifying churn helps banks take proactive steps to improve customer retention and reduce financial losses.

## Dataset Overview 
* **Source:** Kaggle – [Bank Customer Churn Prediction Dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
* **Records:** 10,000 bank customers
* **Target Variable:** `Exited` (1 = churned, 0 = retained)

## Data Preprocessing
* Removed unnecessary identifier columns (`RowNumber`, `CustomerId`, `Surname`).
* Encoded categorical variables (`Geography`, `Gender`) using One-Hot Encoding (`pd.get_dummies` with `drop_first=True`).
* Verified there were no missing values in the dataset.

## Exploratory Data Analysis (EDA)
* Visualized churn distribution using count plots.
* Analyzed continuous variables (`Age`, `Tenure`, `Balance`, `EstimatedSalary`) using histograms (with mean/median lines) and boxplots.
* Generated a correlation heatmap to identify relationships between features.

## Handling Imbalanced Data
* The dataset was heavily imbalanced (approx. 80% retained, 20% churned).
* Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the classes.
* *Note: SMOTE was applied inside an `imblearn.pipeline.Pipeline` during cross-validation to prevent data leakage.*

## Model Building & Evaluation
Three models were evaluated using 5-fold Stratified Cross-Validation:
1. Decision Tree Classifier
2. Random Forest Classifier
3. **XGBoost Classifier (Best Performing)**

## Model Optimization
* Used `GridSearchCV` to fine-tune the hyperparameters of the XGBoost model.
* **Tuned Parameters:** `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`.

## Final Results (XGBoost)
* **Accuracy:** 81.85%
* **ROC-AUC Score:** 0.842
* **Cross-Validation Best F1-Score:** 0.596

## Technologies Used
* **Language:** Python
* **Data Manipulation:** Pandas
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Handling Imbalance:** Imbalanced-learn (SMOTE)

## Output
The final, optimized model was exported using Pickle and saved as: 
 `xgboost_model.pkl`
