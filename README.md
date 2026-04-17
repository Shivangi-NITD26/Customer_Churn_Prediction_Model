# Customer_Churn_Prediction_Model

This machine learning project aims to predict whether a customer will leave (churn) a bank based on their demographic and account-related features. Accurately identifying churn helps banks take proactive steps to improve customer retention and reduce losses.

Dataset Overview
Source: Kaggle – Bank Customer Churn Prediction Dataset(https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)<br>
Records: 10,000 bank customers<br>
Target Variable: Exited (1 = churned, 0 = retained)

### Data Preprocessing
- Removed unnecessary columns (RowNumber, CustomerId, Surname)
- Encoded categorical variables (Geography, Gender)
- Verified no missing values

### Exploratory Data Analysis (EDA)
- Visualized churn distribution
- Plotted histograms and boxplots
- Generated correlation heatmap

### Handling Imbalanced Data
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance classes

### Model Building
- Decision Tree Classifier
- Random Forest Classifier

### Model Optimization
- Used GridSearchCV for hyperparameter tuning
- Selected best Random Forest model

### Results
- Accuracy: 82.25%
- ROC-AUC Score: 0.83

### Technologies Used
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)

### Output
- Trained model saved as: random_forest_model.pkl
