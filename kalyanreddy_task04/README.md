# 🧹 Data Cleaning and Preprocessing using Python

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Pandas-Data%20Analysis-green?style=for-the-badge&logo=pandas">
<img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-orange?style=for-the-badge&logo=numpy">
<img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-red?style=for-the-badge&logo=scikitlearn">
<img src="https://img.shields.io/badge/Seaborn-Visualization-blueviolet?style=for-the-badge">
<img src="https://img.shields.io/badge/Matplotlib-EDA-yellow?style=for-the-badge">
</p>

---

# 📌 Project Overview

Data cleaning is one of the most important stages in the data analysis pipeline. Poor-quality data leads to inaccurate insights and unreliable machine learning models.

This project demonstrates a complete data preprocessing workflow on the **New York City Airbnb Open Dataset**, including:

✔ Missing Value Handling  
✔ Duplicate Removal  
✔ Outlier Detection and Removal  
✔ Feature Standardization  
✔ Data Integrity Validation  
✔ Data Visualization  
✔ Exporting a Clean Dataset

---

# 🎯 Objectives

- Improve data quality and consistency.
- Handle missing values intelligently.
- Detect and remove outliers using the IQR method.
- Standardize numerical features.
- Create meaningful visualizations.
- Export a cleaned dataset for downstream analysis.

---

# 📂 Dataset Information

### Dataset Used

**New York City Airbnb Open Data (AB_NYC_2019.csv)**

Dataset contains information about:

- Hosts
- Room Types
- Prices
- Availability
- Reviews
- Neighborhood Groups
- Geographic Coordinates

---

# 🛠 Technologies Used

| Category | Tools |
|------------|------|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | Scikit-Learn |
| IDE | VS Code |

---

# ⚙ Data Cleaning Pipeline

```text
Raw Dataset
     ↓
Missing Value Handling
     ↓
Duplicate Removal
     ↓
Outlier Detection (IQR)
     ↓
Feature Standardization
     ↓
Visualization
     ↓
Clean Dataset Export
```

---

# 🔍 Missing Value Handling

Handled missing values in:

- name
- host_name
- last_review
- reviews_per_month

### Missing Values Before Cleaning

![](outputs/missing_values_before.png)

### Missing Values After Cleaning

![](outputs/missing_values_after.png)

---

# 📊 Outlier Detection

Outliers were identified using the **Interquartile Range (IQR)** method.

### Before Outlier Removal

![](outputs/outlier_boxplot_before.png)

### After Outlier Removal

![](outputs/outlier_boxplot_after.png)

---

# 📈 Price Distribution

### Before Cleaning

![](outputs/price_distribution_before.png)

### After Cleaning

![](outputs/price_distribution_after.png)

---

# 🔥 Correlation Heatmap

Understanding relationships among numerical features.

![](outputs/correlation_heatmap.png)

---

# 🏠 Room Type Distribution

![](outputs/room_type_distribution.png)

---

# 🌆 Top Neighbourhood Distribution

![](outputs/neighbourhood_distribution.png)

---

# ⚡ Feature Standardization

StandardScaler was applied on:

- Price
- Minimum Nights
- Number of Reviews
- Reviews per Month
- Availability 365

This ensures features are on the same scale for machine learning applications.

---

# 📁 Project Structure

```text
kalyanreddy_task04
│
├── data_cleaning.py
├── README.md
├── requirements.txt
│
└── outputs
    ├── missing_values_before.png
    ├── missing_values_after.png
    ├── outlier_boxplot_before.png
    ├── outlier_boxplot_after.png
    ├── price_distribution_before.png
    ├── price_distribution_after.png
    ├── correlation_heatmap.png
    ├── room_type_distribution.png
    ├── neighbourhood_distribution.png
```

---

# 📈 Results

✅ Successfully handled missing values

✅ No duplicate records found

✅ Removed outliers using IQR

✅ Standardized important numerical features

✅ Generated visual insights

✅ Exported a cleaned dataset for future analysis

---

# 🚀 Key Learnings

- Data Integrity
- Missing Value Imputation
- Duplicate Detection
- Outlier Treatment
- Feature Scaling
- Data Visualization
- Exploratory Data Analysis
- Dataset Preparation for Machine Learning

---

# 🔮 Future Improvements

- Automated Data Cleaning Pipeline
- Advanced Anomaly Detection
- Feature Engineering
- Interactive Dashboard using Power BI
- Data Validation Framework

---

# 📦 Requirements

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

# 👨‍💻 Author

## Byreddy Kalyan Reddy

🎓 B.Tech CSE (AI & DS)  
🏫 Swami Vivekanandha Institute of Technology  
💻 Aspiring Data Scientist | AI Engineer | ML Enthusiast

### Connect With Me

- GitHub: https://github.com/kalyan-ds
- LinkedIn: https://www.linkedin.com/in/kalyan-reddy-byreddy-559b6b344

---

⭐ If you found this project useful, please consider giving it a star!
