import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("ggplot")


# ===============================
# Load datasets
# ===============================

apps_df = pd.read_csv("apps.csv")
reviews_df = pd.read_csv("user_reviews.csv")

print("Datasets Loaded Successfully!")

# ===============================
# Apps Dataset Information
# ===============================

print("\n========== APPS DATASET ==========")

print("\nShape:")
print(apps_df.shape)

print("\nColumns:")
print(apps_df.columns)

print("\nFirst 5 Rows:")
print(apps_df.head())

print("\nMissing Values:")
print(apps_df.isnull().sum())

print("\nDuplicate Rows:")
print(apps_df.duplicated().sum())

print("\nData Types:")
print(apps_df.dtypes)

print("\nDescriptive Statistics:")
print(apps_df.describe(include="all"))

# ===============================
# Reviews Dataset Information
# ===============================

print("\n========== USER REVIEWS DATASET ==========")

print("\nShape:")
print(reviews_df.shape)

print("\nColumns:")
print(reviews_df.columns)

print("\nFirst 5 Rows:")
print(reviews_df.head())

print("\nMissing Values:")
print(reviews_df.isnull().sum())

print("\nDuplicate Rows:")
print(reviews_df.duplicated().sum())

print("\nData Types:")
print(reviews_df.dtypes)

print("\nDescriptive Statistics:")
print(reviews_df.describe(include="all"))

# ====================================
# PHASE 2 : DATA CLEANING
# ====================================

print("\n==============================")
print("DATA CLEANING")
print("==============================")

# ------------------------
# Apps Dataset Cleaning
# ------------------------

print("\nDuplicate Rows Before Cleaning (Apps):")
print(apps_df.duplicated().sum())

# Remove missing ratings
apps_df['Rating'] = apps_df['Rating'].fillna(apps_df['Rating'].median())

# Remove missing sizes
apps_df['Size'] = apps_df['Size'].fillna(apps_df['Size'].median())

# Fill missing Current Ver and Android Ver
apps_df['Current Ver'] = apps_df['Current Ver'].fillna("Unknown")
apps_df['Android Ver'] = apps_df['Android Ver'].fillna("Unknown")

# Convert Installs column
apps_df['Installs'] = apps_df['Installs'].str.replace('+', '', regex=False)
apps_df['Installs'] = apps_df['Installs'].str.replace(',', '', regex=False)
apps_df['Installs'] = apps_df['Installs'].astype(int)

# Convert Price column
apps_df['Price'] = apps_df['Price'].str.replace('$', '', regex=False)
apps_df['Price'] = apps_df['Price'].astype(float)

print("\nMissing Values After Cleaning (Apps):")
print(apps_df.isnull().sum())

# ------------------------
# Reviews Dataset Cleaning
# ------------------------

print("\nDuplicate Rows Before Cleaning (Reviews):")
print(reviews_df.duplicated().sum())

# Remove duplicate rows
reviews_df.drop_duplicates(inplace=True)

# Remove rows with missing sentiment values
reviews_df.dropna(inplace=True)

print("\nShape After Cleaning Reviews:")
print(reviews_df.shape)

print("\nMissing Values After Cleaning:")
print(reviews_df.isnull().sum())

print("\nDuplicate Rows After Cleaning:")
print(reviews_df.duplicated().sum())

# ====================================
# CATEGORY DISTRIBUTION
# ====================================

plt.figure(figsize=(12,8))

apps_df['Category'].value_counts().head(10).plot(kind='bar')

plt.title("Top 10 App Categories")
plt.xlabel("Category")
plt.ylabel("Number of Apps")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("outputs/category_distribution.png")

plt.show()

plt.figure(figsize=(8,5))

sns.histplot(apps_df['Rating'], bins=20, kde=True)

plt.title("App Ratings Distribution")

plt.savefig("outputs/ratings_distribution.png")

plt.show()

plt.figure(figsize=(8,5))

sns.histplot(apps_df['Installs'], bins=30)

plt.title("Install Distribution")

plt.savefig("outputs/install_distribution.png")

plt.show()

plt.figure(figsize=(8,5))

sns.histplot(apps_df['Price'], bins=20)

plt.title("Price Distribution")

plt.savefig("outputs/price_distribution.png")

plt.show()

avg_rating = apps_df.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))

avg_rating.plot(kind='bar')

plt.title("Top Categories by Average Rating")

plt.ylabel("Average Rating")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("outputs/average_rating_by_category.png")

plt.show()

plt.figure(figsize=(8,6))

sns.heatmap(
    apps_df[['Rating','Reviews','Size','Installs','Price']].corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("outputs/correlation_heatmap.png")

plt.show()

plt.figure(figsize=(7,5))

sns.countplot(
    x='Sentiment',
    data=reviews_df
)

plt.title("Sentiment Distribution")

plt.savefig("outputs/sentiment_distribution.png")

plt.show()

# ====================================
# TOP FREE APPS
# ====================================

free_apps = apps_df[apps_df['Type'] == 'Free']

top_free = free_apps.nlargest(10, 'Reviews')

plt.figure(figsize=(12,6))

sns.barplot(
    x='Reviews',
    y='App',
    data=top_free
)

plt.title("Top 10 Free Apps by Reviews")

plt.tight_layout()

plt.savefig("outputs/top_free_apps.png")

plt.show()

paid_apps = apps_df[apps_df['Type'] == 'Paid']

top_paid = paid_apps.nlargest(10, 'Reviews')

plt.figure(figsize=(12,6))

sns.barplot(
    x='Reviews',
    y='App',
    data=top_paid
)

plt.title("Top 10 Paid Apps by Reviews")

plt.tight_layout()

plt.savefig("outputs/top_paid_apps.png")

plt.show()

plt.figure(figsize=(10,6))

sns.countplot(
    y='Content Rating',
    data=apps_df,
    order=apps_df['Content Rating'].value_counts().index
)

plt.title("Content Rating Distribution")

plt.tight_layout()

plt.savefig("outputs/content_rating_distribution.png")

plt.show()

plt.figure(figsize=(10,6))

sns.scatterplot(
    x='Reviews',
    y='Rating',
    data=apps_df
)

plt.title("Rating vs Reviews")

plt.tight_layout()

plt.savefig("outputs/rating_vs_reviews.png")

plt.show()


plt.figure(figsize=(10,6))

sns.scatterplot(
    x='Installs',
    y='Rating',
    data=apps_df
)

plt.title("Rating vs Installs")

plt.tight_layout()

plt.savefig("outputs/rating_vs_installs.png")

plt.show()

plt.figure(figsize=(8,6))

sns.boxplot(
    x=apps_df['Size']
)

plt.title("App Size Distribution")

plt.tight_layout()

plt.savefig("outputs/app_size_distribution.png")

plt.show()

# ====================================
# BUSINESS INSIGHTS
# ====================================

print("\n========== BUSINESS INSIGHTS ==========\n")

print("1. Family and Game categories dominate the Google Play Store.")

print("\n2. Most applications are free, while paid apps represent only a small percentage.")

print("\n3. Highly rated apps tend to receive more installs and reviews.")

print("\n4. Positive sentiment dominates user reviews, indicating overall customer satisfaction.")

print("\n5. Price has little impact on app ratings and popularity.")

print("\n6. Apps with larger user engagement generally achieve better visibility.")

print("\n7. Content ratings are mainly targeted towards Everyone and Teen audiences.")

# ====================================
# RECOMMENDATIONS
# ====================================

print("\n========== RECOMMENDATIONS ==========\n")

print("• Focus on user experience to maintain ratings above 4.0.")

print("\n• Encourage positive reviews and user feedback.")

print("\n• Optimize free apps with advertisements or in-app purchases.")

print("\n• Monitor user sentiments to identify improvement areas.")

print("\n• Target popular categories with high market demand.")

print("\n• Regularly update applications to improve engagement and retention.")



