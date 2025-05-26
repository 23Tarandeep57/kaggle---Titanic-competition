# Titanic - Machine Learning from Disaster

This repository contains my solution to the **Titanic: Machine Learning from Disaster** competition on Kaggle — my very first Kaggle competition!

I focused on **feature engineering, exploratory data analysis, and building models from scratch** using pure logic and standard Python ML libraries (no AutoML or scikit-learn pipelines). The solution uses a blend of domain knowledge, statistical intuition, and classical machine learning models.

---

## What I Did

- **Exploratory Data Analysis (EDA)** using Seaborn and Matplotlib.
- **Preprocessing & Feature Engineering**:
  - Extracted **Title** from names (`Mr`, `Miss`, `Dr`, etc.)
  - Created a binary flag for **numeric vs alphanumeric ticket numbers**
  - Handled missing data using **median imputation** for `Age` and `Fare`, and dropped missing `Embarked`
-  **Normalization**:
  - Used `StandardScaler` to normalize `Fare`, `Age`, and `SibSp`
-  **Feature Selection**:
  - Used a combination of `Pclass`, `Sex`, `Age`, `Fare`, `Embarked`, `Parch`, `SibSp`, `Title`, and `Ticket Type`
- **Modeling**:
  - Trained and cross-validated multiple models:
    - Logistic Regression
    - Decision Trees
    - Naive Bayes
    - Random Forest
    - K-Nearest Neighbors
    - Support Vector Machine (SVM)
  - Combined them using a **Voting Classifier** for final predictions

---

##  Dataset

- **train.csv** and **test.csv** were used from the official Titanic dataset available on [Kaggle](https://www.kaggle.com/c/titanic).

---

## Evaluation

- Evaluation Metric: **Accuracy**
- Used **5-fold Cross-Validation** to evaluate performance
- Achieved good baseline accuracy using Logistic Regression and Decision Trees

---

## Getting Started

### Prerequisites

Make sure you have the following installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
