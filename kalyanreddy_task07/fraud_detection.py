import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report)
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import IsolationForest





# Load Dataset
df = pd.read_csv("creditcard.csv")

print("Dataset Loaded Successfully!")

# Shape
print("\nShape:")
print(df.shape)

# Columns
print("\nColumns:")
print(df.columns)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Data Types
print("\nData Types:")
print(df.dtypes)

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# ===================================
# Remove Duplicate Rows
# ===================================

print("\nDuplicate Rows Before Cleaning:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")
print(df.shape)

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

sns.set_style("whitegrid")

# ===================================
# Class Distribution
# ===================================

print("\nClass Distribution:")
print(df["Class"].value_counts())

plt.figure(figsize=(6,5))

sns.countplot(x=df["Class"])

plt.title("Fraud vs Non-Fraud Transactions")

plt.xlabel("Class")
plt.ylabel("Count")

plt.savefig("outputs/class_distribution.png")

plt.close()

plt.figure(figsize=(8,6))

sns.histplot(df["Amount"], bins=50)

plt.title("Transaction Amount Distribution")

plt.savefig("outputs/amount_distribution.png")

plt.close()

plt.figure(figsize=(8,6))

sns.boxplot(x=df["Amount"])

plt.title("Transaction Amount Boxplot")

plt.savefig("outputs/transaction_amount_boxplot.png")

plt.close()

plt.figure(figsize=(16,12))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("outputs/correlation_heatmap.png")

plt.close()

fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

plt.figure(figsize=(7,5))

sns.barplot(
    x=["Normal","Fraud"],
    y=[normal["Amount"].mean(), fraud["Amount"].mean()]
)

plt.ylabel("Average Transaction Amount")

plt.title("Average Amount: Fraud vs Normal")

plt.savefig("outputs/fraud_vs_normal.png")

plt.close()

# ===================================
# Feature Scaling
# ===================================

scaler = StandardScaler()

df["Amount"] = scaler.fit_transform(df[["Amount"]])

# ===================================
# Features and Target
# ===================================

X = df.drop("Class", axis=1)

y = df["Class"]

# ===================================
# Train Test Split
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)

print("Testing Shape:", X_test.shape)

# ===================================
# Logistic Regression
# ===================================

model = LogisticRegression(
    max_iter=5000,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nLogistic Regression Model Trained Successfully!")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nPrecision:")
print(precision_score(y_test, y_pred))

print("\nRecall:")
print(recall_score(y_test, y_pred))

print("\nF1 Score:")
print(f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.savefig("outputs/confusion_matrix.png")

plt.close()

print("\n==============================")
print("Decision Tree")
print("==============================")

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train,y_train)

dt_pred = dt_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test,dt_pred))

print("\nPrecision:")
print(precision_score(y_test,dt_pred))

print("\nRecall:")
print(recall_score(y_test,dt_pred))

print("\nF1 Score:")
print(f1_score(y_test,dt_pred))

print("\n==============================")
print("Random Forest")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train,y_train)

rf_pred = rf_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test,rf_pred))

print("\nPrecision:")
print(precision_score(y_test,rf_pred))

print("\nRecall:")
print(recall_score(y_test,rf_pred))

print("\nF1 Score:")
print(f1_score(y_test,rf_pred))

cm_rf = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "outputs/random_forest_confusion_matrix.png"
)

plt.close()

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

importance.sort_values(
    ascending=False
).head(15).plot(
    kind='barh',
    figsize=(10,7)
)

plt.title(
    "Top 15 Important Features"
)

plt.savefig(
    "outputs/feature_importance.png"
)

plt.close()

rf_prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    rf_prob
)

roc_auc = auc(
    fpr,
    tpr
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.savefig(
    "outputs/roc_curve.png"
)

plt.close()

precision, recall, _ = precision_recall_curve(y_test, rf_prob)

plt.figure(figsize=(8,6))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.savefig("outputs/precision_recall_curve.png")

plt.close()

iso = IsolationForest(
    contamination=0.0017,
    random_state=42
)

iso.fit(X)

anomaly_pred = iso.predict(X)

print("\nIsolation Forest Completed!")

def predict_transaction(data):
    prediction = rf_model.predict(data)

    if prediction[0] == 1:
        print("⚠ Fraud Transaction Detected")
    else:
        print("✓ Legitimate Transaction")

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

importance.sort_values(
    ascending=False
).head(15).plot(
    kind="barh",
    figsize=(10,7)
)

plt.title(
    "Top 15 Important Features"
)

plt.savefig(
    "outputs/feature_importance.png"
)

plt.close()

print("\n========== BUSINESS INSIGHTS ==========")

print("""
1. Fraudulent transactions account for a very small percentage of all transactions.

2. Accuracy alone is misleading due to severe class imbalance.

3. Random Forest achieved the best overall performance.

4. Recall is more important than accuracy because missing fraud transactions is costly.

5. Machine learning models can effectively identify suspicious patterns.
""")

print("\n========== RECOMMENDATIONS ==========")

print("""
• Deploy Random Forest for production systems.

• Continuously monitor transactions in real time.

• Retrain models periodically with new transaction data.

• Combine anomaly detection with supervised learning.

• Use additional features and ensemble models for better fraud detection.
""")













