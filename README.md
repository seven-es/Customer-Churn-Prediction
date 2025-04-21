# 💼 Customer Balance Threshold Classifier

This project builds machine learning models to **predict whether a customer's will chrun based on a certain threshold**

> ⚠️ **Important:** This project does **not** directly predict customer churn. Instead, it identifies whether a customer has a **high balance** (above a threshold) — which can later be used as a proxy to assess churn risk or drive targeted retention strategies.

---

## 🎯 Objective

Determine if a customer's account balance exceeds a certain threshold using classification models. Customers with **balance ≥ 73,057.376** are labeled as **"high balance"** (1), while others are labeled **"low balance"** (0).

This binary classification serves as the prediction target.

---

## 📊 Dataset Overview

The dataset contains various customer features including:

- Age
- Tenure
- Number of Products
- Credit Score
- Estimated Salary
- Geography, Gender,......

A preprocessing pipeline was applied to clean, transform, and prepare the data for modeling.

---

## 🛠️ Data Preparation

- Dropped irrelevant identifiers (e.g. RowNumber)
- Filled missing values with median/mode/rounded mean
- Converted binary columns (`HasCrCard`, `IsActiveMember`, `Exited`) into "Yes/No" and then into numeric
- Removed outliers using the IQR method
- Encoded categorical variables (`Geography`, `Gender`)
- threshold is when a customer has Salary  "73057.376" or higher
- Binarized the `Balance` column:


  
  ```python
  df["Balance"] = df["Balance"].apply(lambda x: 1 if x >= 73057.376 else 0)



 ## 🧠 Models Trained

Two classifiers were trained to predict **whether the customer has a balance above the threshold**.

### 1. Decision Tree Classifier

- **Accuracy**: `68.93%`
- **Confusion Matrix**:
 [[ 693  427]
 [ 460 1275]]

- **True Negatives (TN)** = 693 → Predicted **Low Balance**, actually **Low Balance**
- **False Positives (FP)** = 427 → Predicted **High Balance**, actually **Low Balance**
- **False Negatives (FN)** = 460 → Predicted **Low Balance**, actually **High Balance**
- **True Positives (TP)** = 1275 → Predicted **High Balance**, actually **High Balance**

---

### 2. Naive Bayes (Numerical Data Only)

- **Accuracy**: `61.09%`
- **Confusion Matrix**:

 [[  12 1108]
 [   3 1732]]

- **True Negatives (TN)** = 12 → Predicted **Low Balance**, actually **Low Balance**
- **False Positives (FP)** = 1108 → Predicted **High Balance**, actually **Low Balance**
- **False Negatives (FN)** = 3 → Predicted **Low Balance**, actually **High Balance**
- **True Positives (TP)** = 1732 → Predicted **High Balance**, actually **High Balance**

---



## 📈 Insights

- The **Decision Tree Classifier** performs better overall, with more balanced detection of both high and low balance customers.
- The **Naive Bayes model** shows a strong bias toward predicting high balance customers, leading to a high false positive rate.
- Customers with high balances might exhibit distinct behavioral or demographic patterns, which can be leveraged in future churn prediction models.
- This model setup is useful for **early segmentation** — identifying high-value customers who may need targeted interventions.

---
## 🚀 Future Improvements

- Train a separate model using **`Exited` (churn)** as the actual target variable instead of balance threshold.
- Incorporate **feature selection and engineering** to improve model performance and interpretability.
- Evaluate **advanced classification models** such as **Random Forest**, **XGBoost**, and other ensemble techniques.
- Address potential **class imbalance** challenges when modeling churn directly to avoid biased predictions.

---

## 🛠️ Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## 📂 File Structure

- `Reg_problem.py`: Main code file for data preparation, modeling, and evaluation



---
