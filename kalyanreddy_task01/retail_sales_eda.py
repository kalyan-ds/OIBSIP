# ==========================================
# Retail Sales Exploratory Data Analysis
# Oasis Infobyte Data Analytics Internship
# Author : Byreddy Kalyan Reddy
# ==========================================



# ==========================================
# Import Libraries
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Display all columns
pd.set_option('display.max_columns', None)

# Plot style
sns.set_style("whitegrid")

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("retail_sales_dataset.csv")

print("\nDataset Loaded Successfully!\n")

# ==========================================
# Basic Information
# ==========================================

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# ==========================================
# Data Cleaning
# ==========================================

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Remove rows with missing values
df.dropna(inplace=True)

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])

print("\nAfter Cleaning:")
print("Shape:", df.shape)

print("\nRemaining Missing Values:")
print(df.isnull().sum())

# ==========================================
# Descriptive Statistics
# ==========================================

print("\nDescriptive Statistics")
print(df.describe())

print("\nMean Age:")
print(df["Age"].mean())

print("\nMedian Age:")
print(df["Age"].median())

print("\nMode of Age:")
print(df["Age"].mode())

print("\nStandard Deviation of Total Amount:")
print(df["Total_Amount"].std())

# ==========================================
# Customer Analysis
# ==========================================

print("\nGender Distribution:")
print(df["Gender"].value_counts())

print("\nCustomer Segment Distribution:")
print(df["Customer_Segment"].value_counts())

print("\nIncome Distribution:")
print(df["Income"].value_counts())

print("\nAverage Age:")
print(round(df["Age"].mean(),2))

print("\nTop 10 Countries:")
print(df["Country"].value_counts().head(10))

# ==========================================
# Product Analysis
# ==========================================

print("\nProduct Category Distribution:")
print(df["Product_Category"].value_counts())

print("\nTop Product Brands:")
print(df["Product_Brand"].value_counts().head(10))

print("\nTop Product Types:")
print(df["Product_Type"].value_counts().head(10))

# ==========================================
# Payment Analysis
# ==========================================

print("\nPayment Methods:")
print(df["Payment_Method"].value_counts())

print("\nOrder Status:")
print(df["Order_Status"].value_counts())

print("\nCustomer Feedback:")
print(df["Feedback"].value_counts())

# ==========================================
# Time Series Analysis
# ==========================================

monthly_sales = df.groupby("Month")["Total_Amount"].sum()

month_order = [
'January','February','March','April',
'May','June','July','August',
'September','October','November','December'
]

monthly_sales = monthly_sales.reindex(month_order)

plt.figure(figsize=(12,6))
monthly_sales.plot(marker='o')
plt.title("Monthly Sales Trend")
plt.ylabel("Total Sales")
plt.grid(True)
plt.savefig("outputs/monthly_sales_trend.png")
plt.close()

# ==========================================
# Visualization Settings
# ==========================================

plt.figure(figsize=(8,6))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.savefig("outputs/gender_distribution.png")
plt.close()


plt.figure(figsize=(8,6))
sns.countplot(x="Customer_Segment", data=df)
plt.title("Customer Segment Distribution")
plt.savefig("outputs/customer_segment_distribution.png")
plt.close()


plt.figure(figsize=(8,6))
sns.histplot(df["Age"], bins=20)
plt.title("Age Distribution")
plt.savefig("outputs/age_distribution.png")
plt.close()


plt.figure(figsize=(10,6))
df["Product_Category"].value_counts().plot(kind="bar")
plt.title("Product Category Distribution")
plt.xticks(rotation=45)
plt.savefig("outputs/product_category_distribution.png")
plt.close()


plt.figure(figsize=(10,6))
df["Payment_Method"].value_counts().plot(kind="bar")
plt.title("Payment Methods")
plt.xticks(rotation=45)
plt.savefig("outputs/payment_methods.png")
plt.close()


plt.figure(figsize=(10,6))
df["Feedback"].value_counts().plot(kind="bar")
plt.title("Customer Feedback")
plt.xticks(rotation=45)
plt.savefig("outputs/customer_feedback.png")
plt.close()


plt.figure(figsize=(10,6))
df["Order_Status"].value_counts().plot(kind="bar")
plt.title("Order Status")
plt.xticks(rotation=45)
plt.savefig("outputs/order_status.png")
plt.close()

plt.figure(figsize=(10,8))

numeric_columns = [
'Age',
'Total_Purchases',
'Amount',
'Total_Amount',
'Ratings'
]

sns.heatmap(
df[numeric_columns].corr(),
annot=True,
cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()



plt.figure(figsize=(10,6))

df["Product_Brand"].value_counts().head(10).plot(kind='bar')

plt.title("Top 10 Product Brands")
plt.xticks(rotation=45)

plt.savefig("outputs/top_product_brands.png")
plt.close()

# ==========================================
# Business Insights and Recommendations
# ==========================================

print("\n========== BUSINESS INSIGHTS ==========")

print("\n1. Electronics is the highest-selling category.")
print("Recommendation: Increase inventory and promotions for Electronics.")

print("\n2. Regular customers form the majority.")
print("Recommendation: Launch loyalty programs to retain them.")

print("\n3. Credit Card is the most preferred payment method.")
print("Recommendation: Offer cashback and card-based discounts.")

print("\n4. Excellent and Good feedback dominate.")
print("Recommendation: Maintain product quality and customer service.")

print("\n5. Delivered orders are significantly higher.")
print("Recommendation: Continue optimizing logistics.")

print("\n6. USA contributes the highest number of customers.")
print("Recommendation: Focus marketing campaigns in the USA market.")

print("\n7. Male customers dominate the dataset.")
print("Recommendation: Introduce targeted campaigns for female customers.")

print("\n8. Medium-income customers are the largest segment.")
print("Recommendation: Design products and pricing around medium-income buyers.")

print("\n========== END OF ANALYSIS ==========")
# ==========================================
# EDA Completed
# ==========================================

print("\n====================================")
print("Retail Sales EDA Completed Successfully!")
print("All visualizations saved in outputs folder.")
print("====================================")