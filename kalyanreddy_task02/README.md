# Customer Segmentation Analysis
# Customer Segmentation Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![KMeans](https://img.shields.io/badge/KMeans-Clustering-red)
![Status](https://img.shields.io/badge/Status-Completed-success)
## 📌 Objective

Perform customer segmentation analysis for an e-commerce company using clustering techniques to group customers based on purchasing behavior and spending patterns. These insights help businesses design targeted marketing strategies and improve customer satisfaction.

---

## 📂 Dataset

Dataset: iFood Customer Data

Rows: 2205

Columns: 39

Features include:

* Income
* Age
* Purchase Amount
* Web Purchases
* Store Purchases
* Catalog Purchases
* Customer Response
* Campaign Acceptance

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📊 Exploratory Data Analysis

* Dataset Exploration
* Missing Value Analysis
* Duplicate Removal
* Descriptive Statistics
* Customer Behavior Analysis

---

## 🤖 Machine Learning Technique

### K-Means Clustering

Feature Selection:

* Income
* Age
* MntTotal
* NumWebPurchases
* NumStorePurchases

### Data Preprocessing

* StandardScaler

### Cluster Optimization

* Elbow Method

### Number of Clusters Selected

K = 3

---

# 📈 Visualizations

### Customer Age Distribution

<img src="outputs/customer_age_distribution.png" width="600">

### Customer Spending Distribution

<img src="outputs/customer_spending_distribution.png" width="600">

### Correlation Heatmap

<img src="outputs/correlation_heatmap.png" width="700">

### Elbow Method

<img src="outputs/elbow_method.png" width="600">

### Customer Clusters

<img src="outputs/customer_clusters.png" width="700">

### Cluster Distribution

<img src="outputs/cluster_distribution.png" width="600">

### Customer Segment Distribution

<img src="outputs/customer_segment_distribution.png" width="600">

### Income Distribution

<img src="outputs/income_distribution.png" width="600">

### Spending by Segment

<img src="outputs/spending_by_segment.png" width="600">

### Segment Income Analysis

<img src="outputs/segment_income.png" width="600">


---

## 👥 Customer Segments

### ⭐ High Value Customers

* High income
* High spending
* Frequent purchases

### 🟢 Medium Value Customers

* Moderate spending
* Growth potential

### 🔵 Low Value Customers

* Low spending
* Suitable for promotional campaigns

---

## 💡 Business Recommendations

### High Value Customers

* Loyalty rewards
* Premium memberships
* Personalized recommendations

### Medium Value Customers

* Bundle offers
* Email marketing campaigns

### Low Value Customers

* Discount coupons
* Promotional campaigns
* Social media engagement

---

## 📁 Project Structure

kalyanreddy_task02

├── customer_segmentation.py

├── requirements.txt

├── README.md

├── ifood_df.csv

└── outputs/

```
├── customer_age_distribution.png

├── customer_spending_distribution.png

├── correlation_heatmap.png

├── elbow_method.png

├── customer_clusters.png

├── cluster_distribution.png

├── customer_segment_distribution.png

├── income_distribution.png

├── spending_by_segment.png

├── segment_income.png

└── customer_cluster_profiles.csv
```

---

## 🎯 Outcome

Successfully segmented customers into three groups using K-Means clustering and generated business insights to support targeted marketing strategies.
