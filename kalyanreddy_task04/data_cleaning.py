import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")



df = pd.read_csv("AB_NYC_2019.csv")

print("Dataset Loaded Successfully!")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nDescriptive Statistics:")
print(df.describe())

# ==============================
# PHASE 2 : MISSING VALUE HANDLING
# ==============================

print("\n==============================")
print("MISSING VALUE HANDLING")
print("==============================")

# Visualize Missing Values Before Cleaning
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values Before Cleaning")
plt.savefig("outputs/missing_values_before.png")
plt.show()

# Handle Missing Values

# Fill missing names
df['name'] = df['name'].fillna('Unknown')

# Fill missing host names
df['host_name'] = df['host_name'].fillna('Unknown')

# Convert last_review to datetime
df['last_review'] = pd.to_datetime(df['last_review'])

# Fill missing last_review dates
df['last_review'] = df['last_review'].fillna(pd.Timestamp('2000-01-01'))

# Fill missing reviews_per_month using median
df['reviews_per_month'] = df['reviews_per_month'].fillna(
    df['reviews_per_month'].median()
)

# Check Missing Values After Cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Visualize Missing Values After Cleaning
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title("Missing Values After Cleaning")
plt.savefig("outputs/missing_values_after.png")
plt.show()


# ==============================
# PHASE 3 : DUPLICATES + OUTLIER DETECTION
# ==============================

print("\n==============================")
print("DUPLICATE REMOVAL & OUTLIER DETECTION")
print("==============================")

# Duplicate rows before cleaning
duplicates_before = df.duplicated().sum()
print("\nDuplicate Rows Before Cleaning:", duplicates_before)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Duplicate rows after cleaning
duplicates_after = df.duplicated().sum()
print("Duplicate Rows After Cleaning:", duplicates_after)

# -------------------------------
# Boxplots BEFORE removing outliers
# -------------------------------

plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
sns.boxplot(y=df['price'])
plt.title("Price")

plt.subplot(1,3,2)
sns.boxplot(y=df['minimum_nights'])
plt.title("Minimum Nights")

plt.subplot(1,3,3)
sns.boxplot(y=df['number_of_reviews'])
plt.title("Number of Reviews")

plt.tight_layout()
plt.savefig("outputs/outlier_boxplot_before.png")
plt.show()


# -------------------------------
# Price Distribution BEFORE cleaning
# -------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution Before Outlier Removal")
plt.savefig("outputs/price_distribution_before.png")
plt.show()


# -------------------------------
# IQR Method
# -------------------------------

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5*IQR
upper_bound = Q3 + 1.5*IQR

print("\nLower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Remove outliers
df = df[(df['price'] >= lower_bound) &
        (df['price'] <= upper_bound)]

print("\nShape After Outlier Removal:")
print(df.shape)


# -------------------------------
# Boxplots AFTER removing outliers
# -------------------------------

plt.figure(figsize=(15,6))

plt.subplot(1,3,1)
sns.boxplot(y=df['price'])
plt.title("Price")

plt.subplot(1,3,2)
sns.boxplot(y=df['minimum_nights'])
plt.title("Minimum Nights")

plt.subplot(1,3,3)
sns.boxplot(y=df['number_of_reviews'])
plt.title("Number of Reviews")

plt.tight_layout()
plt.savefig("outputs/outlier_boxplot_after.png")
plt.show()


# -------------------------------
# Price Distribution AFTER cleaning
# -------------------------------

plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Price Distribution After Outlier Removal")
plt.savefig("outputs/price_distribution_after.png")
plt.show()


# ==============================
# PHASE 4 : STANDARDIZATION & VISUALIZATION
# ==============================

from sklearn.preprocessing import StandardScaler

print("\n==============================")
print("STANDARDIZATION & VISUALIZATION")
print("==============================")

# ----------------------------------
# Correlation Heatmap
# ----------------------------------

numeric_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()


# ----------------------------------
# Room Type Distribution
# ----------------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    x='room_type',
    data=df,
    order=df['room_type'].value_counts().index
)

plt.title("Room Type Distribution")
plt.xticks(rotation=15)

plt.savefig("outputs/room_type_distribution.png")

plt.show()


# ----------------------------------
# Top 10 Neighbourhoods
# ----------------------------------

top_neighbourhoods = df['neighbourhood'].value_counts().head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    x=top_neighbourhoods.values,
    y=top_neighbourhoods.index
)

plt.title("Top 10 Neighbourhoods")

plt.savefig("outputs/neighbourhood_distribution.png")

plt.show()


# ----------------------------------
# Standardization
# ----------------------------------

scaler = StandardScaler()

columns_to_scale = [
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'availability_365'
]

df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

print("\nStandardization Completed Successfully!")

# ----------------------------------
# Save Cleaned Dataset
# ----------------------------------

df.to_csv(
    "outputs/cleaned_dataset.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully!")