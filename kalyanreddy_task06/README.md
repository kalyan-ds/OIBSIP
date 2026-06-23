# 🍷 Wine Quality Prediction using Machine Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-green?style=for-the-badge&logo=pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/Seaborn-Visualization-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Matplotlib-EDA-yellow?style=for-the-badge">
</p>

---

## 📌 Project Overview

This project focuses on predicting wine quality based on its physicochemical properties using Machine Learning classification algorithms.

The dataset contains various chemical characteristics such as acidity, density, alcohol content, sulphates, and chlorides. Three different classification models were implemented and evaluated to determine the best-performing model.

---

## 🎯 Objective

- Analyze wine quality data.
- Perform Exploratory Data Analysis (EDA).
- Build Machine Learning models.
- Compare model performances.
- Identify the most important factors affecting wine quality.

---

## 📂 Dataset
Dataset Link: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset


Dataset used:

**Wine Quality Dataset (WineQT.csv)**

Features include:

- Fixed Acidity
- Volatile Acidity
- Citric Acid
- Residual Sugar
- Chlorides
- Free Sulfur Dioxide
- Total Sulfur Dioxide
- Density
- pH
- Sulphates
- Alcohol

Target Variable:

- Quality

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 📊 Exploratory Data Analysis

Generated visualizations:

- Quality Distribution
- Correlation Heatmap
- Alcohol vs Quality
- Volatile Acidity vs Quality
- Alcohol Distribution
- Density Distribution
- Sulphates vs Quality
- Pairplot

---

## 🤖 Machine Learning Models

### 🌲 Random Forest Classifier

Accuracy:

```text
92.14%
```

---

### ⚙ Support Vector Classifier (SVC)

Accuracy:

```text
89.96%
```

---

### 🚀 Stochastic Gradient Descent (SGD)

Accuracy:

```text
88.21%
```

---

## 📈 Model Evaluation

Generated:

- Random Forest Confusion Matrix
- SVC Confusion Matrix
- SGD Confusion Matrix
- ROC Curve
- Feature Importance Plot
- Model Accuracy Comparison

---

## 🏆 Best Model

### Random Forest Classifier

Accuracy:

```text
92.14%
```

Random Forest achieved the highest accuracy and provided the best performance for wine quality classification.

---

## 📁 Project Structure

```text
kalyanreddy_task06
│
├── wine_quality_prediction.py
├── README.md
├── requirements.txt
│
└── outputs
    ├── quality_distribution.png
    ├── correlation_heatmap.png
    ├── alcohol_vs_quality.png
    ├── acidity_vs_quality.png
    ├── alcohol_distribution.png
    ├── density_distribution.png
    ├── sulphates_vs_quality.png
    ├── pairplot.png
    ├── confusion_matrix_rf.png
    ├── confusion_matrix_svc.png
    ├── confusion_matrix_sgd.png
    ├── feature_importance_rf.png
    ├── model_accuracy_comparison.png
    └── roc_curve_rf.png
```

---

## 📌 Key Insights

- Alcohol content positively influences wine quality.
- Higher volatile acidity negatively affects quality.
- Random Forest outperformed SVC and SGD classifiers.
- Feature importance analysis revealed alcohol and sulphates as significant predictors.

---

## 🚀 Future Improvements

- Hyperparameter tuning using GridSearchCV.
- Cross-validation for better generalization.
- XGBoost and LightGBM implementation.
- Multi-class classification approach.

---

## 👨‍💻 Author

**Byreddy Kalyan Reddy**

B.Tech CSE (AI & DS)

Swami Vivekanandha Institute of Technology

GitHub: https://github.com/kalyan-ds

LinkedIn: https://www.linkedin.com/in/kalyan-reddy-byreddy-559b6b344

---