# ==========================================
# Customer Segmentation Analysis
# Oasis Infobyte Data Analytics Internship
# Author : Byreddy Kalyan Reddy
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_style("whitegrid")

df = pd.read_csv("customer_segmentation_dataset.csv")

print("Dataset Loaded Successfully!")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Basic Information
# ==========================================

print("\nData Types:")
print(df.dtypes)

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================
# Descriptive Statistics
# ==========================================

print("\nDescriptive Statistics:")
print(df.describe())

print("\nAverage Customer Age:")
print(round(df["Age"].mean(),2))

print("\nAverage Total Purchase:")
print(round(df["MntTotal"].mean(),2))

print("\nAverage Web Purchases:")
print(round(df["NumWebPurchases"].mean(),2))

print("\nAverage Store Purchases:")
print(round(df["NumStorePurchases"].mean(),2))

print("\nAverage Catalog Purchases:")
print(round(df["NumCatalogPurchases"].mean(),2))

# ==========================================
# Customer Age Distribution
# ==========================================

plt.figure(figsize=(8,6))
sns.histplot(df["Age"], bins=20)
plt.title("Customer Age Distribution")
plt.savefig("outputs/customer_age_distribution.png")
plt.close()


# ==========================================
# Total Spending Distribution
# ==========================================

plt.figure(figsize=(8,6))
sns.histplot(df["MntTotal"], bins=20)
plt.title("Customer Spending Distribution")
plt.savefig("outputs/customer_spending_distribution.png")
plt.close()


# ==========================================
# Correlation Heatmap
# ==========================================

plt.figure(figsize=(10,8))

columns = [
'Income',
'Age',
'MntTotal',
'NumWebPurchases',
'NumStorePurchases',
'NumCatalogPurchases'
]

sns.heatmap(
df[columns].corr(),
annot=True,
cmap="coolwarm"
)

plt.savefig("outputs/correlation_heatmap.png")
plt.close()

# ==========================================
# Feature Selection
# ==========================================

features = [
    'Income',
    'Age',
    'MntTotal',
    'NumWebPurchases',
    'NumStorePurchases'
]

X = df[features]

# ==========================================
# Feature Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# Elbow Method
# ==========================================

wcss = []

for i in range(1,11):

    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))

plt.plot(range(1,11),wcss,marker='o')

plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.savefig(
    "outputs/elbow_method.png"
)

plt.close()

# ==========================================
# K-Means Clustering
# ==========================================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Counts:")
print(df["Cluster"].value_counts())

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Income",
    y="MntTotal",
    hue="Cluster",
    palette="Set1"
)

plt.title("Customer Segments")

plt.savefig(
    "outputs/customer_clusters.png"
)

plt.close()

plt.figure(figsize=(8,6))

sns.countplot(
    x="Cluster",
    data=df
)

plt.title("Cluster Distribution")

plt.savefig(
    "outputs/cluster_distribution.png"
)

plt.close()

print("\nCluster Profiles:")

cluster_summary = df.groupby("Cluster")[[
    "Income",
    "Age",
    "MntTotal",
    "NumWebPurchases",
    "NumStorePurchases"
]].mean()

print(cluster_summary)

cluster_summary.to_csv(
    "outputs/customer_cluster_profiles.csv"
)

segment_names = {
    0: "High Value",
    1: "Low Value",
    2: "Medium Value"
}

df["Segment"] = df["Cluster"].map(segment_names)

print("\nCustomer Segments:")
print(df["Segment"].value_counts())

plt.figure(figsize=(8,6))

sns.countplot(
    x="Segment",
    data=df,
    palette="Set2"
)

plt.title("Customer Segment Distribution")

plt.savefig(
    "outputs/customer_segment_distribution.png"
)

plt.close()

print("\n===== BUSINESS RECOMMENDATIONS =====")

print("""
High Value Customers:
- Provide loyalty rewards.
- Offer premium memberships.
- Personalized recommendations.

Medium Value Customers:
- Encourage frequent purchases.
- Promote bundle offers.
- Target with email campaigns.

Low Value Customers:
- Offer discounts and coupons.
- Run promotional campaigns.
- Increase engagement through social media.
""")

plt.figure(figsize=(8,6))
sns.histplot(df["Income"], bins=20)
plt.title("Income Distribution")
plt.savefig("outputs/income_distribution.png")
plt.close()

plt.figure(figsize=(8,6))

sns.boxplot(
    x="Segment",
    y="MntTotal",
    data=df
)

plt.title("Total Spending by Segment")

plt.savefig(
    "outputs/spending_by_segment.png"
)

plt.close()

segment_income = df.groupby("Segment")["Income"].mean()

plt.figure(figsize=(8,6))

segment_income.plot(kind="bar")

plt.title("Average Income per Segment")

plt.savefig(
    "outputs/segment_income.png"
)

plt.close()