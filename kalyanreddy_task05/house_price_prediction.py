import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

sns.set_style("whitegrid")

# ====================================
# Load Dataset
# ====================================

df = pd.read_csv("house_price_prediction_dataset.csv")

print("Dataset Loaded Successfully!")

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nData Types:")
print(df.dtypes)

print("\nDescriptive Statistics:")
print(df.describe())

plt.figure(figsize=(8,6))

sns.histplot(
    df["price"],
    bins=20
)

plt.title("House Price Distribution")

plt.savefig(
    "outputs/price_distribution.png"
)

plt.close()

plt.figure(figsize=(10,8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.savefig(
    "outputs/correlation_heatmap.png"
)

plt.close()

df = pd.get_dummies(
    df,
    drop_first=True
)

y = df["price"]

X = df.drop(
    "price",
    axis=1
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nMean Absolute Error:",mae)

print("\nMean Squared Error:",mse)

print("\nRoot Mean Squared Error:",rmse)

print("\nR2 Score:",r2)

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title(
    "Actual vs Predicted Price"
)

plt.savefig(
    "outputs/actual_vs_predicted.png"
)

plt.close()

importance = pd.Series(
    model.coef_,
    index=X.columns
)

importance.sort_values(
    ascending=False
).head(10).plot(
    kind="barh",
    figsize=(8,6)
)

plt.title(
    "Top 10 Important Features"
)

plt.savefig(
    "outputs/feature_importance.png"
)

plt.close()

plt.figure(figsize=(8,6))

residuals = y_test - y_pred

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(y=0,color='red')

plt.xlabel("Predicted Price")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.savefig(
    "outputs/residual_plot.png"
)

plt.close()

print("\n========== BUSINESS INSIGHTS ==========")

print("""
1. Area has a strong influence on house prices.

2. Air conditioning and parking facilities positively affect prices.

3. Furnishing status contributes significantly to house valuation.

4. Main road accessibility increases property value.

5. Houses with more bathrooms and stories tend to have higher prices.
""")

print("\n========== RECOMMENDATIONS ==========")

print("""
• Focus on larger houses for premium pricing.

• Include air conditioning and parking facilities to maximize value.

• Furnished houses generally command higher prices.

• Properties near main roads are more attractive to buyers.
""")

