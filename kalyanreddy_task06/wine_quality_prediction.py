import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc



df = pd.read_csv("WineQT.csv")

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

plt.figure(figsize=(8,5))
sns.countplot(x='quality', data=df)
plt.title("Wine Quality Distribution")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.savefig("outputs/quality_distribution.png")
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation_heatmap.png")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='quality', y='alcohol', data=df)
plt.title("Alcohol vs Quality")
plt.savefig("outputs/alcohol_vs_quality.png")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='quality', y='volatile acidity', data=df)
plt.title("Volatile Acidity vs Quality")
plt.savefig("outputs/acidity_vs_quality.png")
plt.show()

df_pair = df.drop('Id', axis=1)

sns.pairplot(df_pair[['alcohol',
                      'volatile acidity',
                      'citric acid',
                      'sulphates',
                      'quality']])

plt.savefig("outputs/pairplot.png")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['alcohol'], bins=30, kde=True)
plt.title("Alcohol Distribution")
plt.savefig("outputs/alcohol_distribution.png")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df['density'], bins=30, kde=True)
plt.title("Density Distribution")
plt.savefig("outputs/density_distribution.png")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='quality', y='sulphates', data=df)
plt.title("Sulphates vs Quality")
plt.savefig("outputs/sulphates_vs_quality.png")
plt.show()

# Binary Classification
df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

print(df['quality'].value_counts())

X = df.drop(['quality', 'Id'], axis=1)
y = df['quality']



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)


svc_model = SVC()

svc_model.fit(X_train, y_train)

svc_pred = svc_model.predict(X_test)


sgd_model = SGDClassifier(random_state=42)

sgd_model.fit(X_train, y_train)

sgd_pred = sgd_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:")
print(rf_acc)

print(classification_report(y_test, rf_pred))

svc_acc = accuracy_score(y_test, svc_pred)

print("\nSVC Accuracy:")
print(svc_acc)

print(classification_report(y_test, svc_pred))

sgd_acc = accuracy_score(y_test, sgd_pred)

print("\nSGD Accuracy:")
print(sgd_acc)

print(classification_report(y_test, sgd_pred))

cm_rf = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)

plt.figure(figsize=(6,5))
disp.plot(cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.savefig("outputs/confusion_matrix_rf.png")
plt.show()


cm_svc = confusion_matrix(y_test, svc_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_svc)

plt.figure(figsize=(6,5))
disp.plot(cmap='Greens')
plt.title("SVC Confusion Matrix")
plt.savefig("outputs/confusion_matrix_svc.png")
plt.show()

cm_sgd = confusion_matrix(y_test, sgd_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_sgd)

plt.figure(figsize=(6,5))
disp.plot(cmap='Oranges')
plt.title("SGD Confusion Matrix")
plt.savefig("outputs/confusion_matrix_sgd.png")
plt.show()

importance = rf_model.feature_importances_

feature_names = X.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance_df
)

plt.title("Feature Importance - Random Forest")
plt.savefig("outputs/feature_importance_rf.png")
plt.show()


accuracy_df = pd.DataFrame({
    "Model": ["Random Forest", "SVC", "SGD"],
    "Accuracy": [rf_acc, svc_acc, sgd_acc]
})

plt.figure(figsize=(8,5))

sns.barplot(
    x='Model',
    y='Accuracy',
    data=accuracy_df
)

plt.title("Model Accuracy Comparison")
plt.savefig("outputs/model_accuracy_comparison.png")
plt.show()

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

rf_prob = rf_model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, rf_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot([0,1],[0,1],'r--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Random Forest")

plt.legend()

plt.savefig("outputs/roc_curve_rf.png")

plt.show()



