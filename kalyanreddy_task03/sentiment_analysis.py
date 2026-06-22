import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nltk.download('stopwords')
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud





# Load dataset
df = pd.read_csv("Twitter_Data.csv")

print("Dataset Loaded Successfully!\n")

# Shape
print("Shape:")
print(df.shape)

# Columns
print("\nColumns:")
print(df.columns)

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Data types
print("\nData Types:")
print(df.dtypes)

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# ==========================================
# DATA CLEANING
# ==========================================

# Remove missing values
df.dropna(inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Convert category to integer
df["category"] = df["category"].astype(int)

print("\nShape After Cleaning:")
print(df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows After Cleaning:")
print(df.duplicated().sum())

# Sentiment Distribution
print("\nSentiment Distribution:")
print(df["category"].value_counts())

# Convert numeric labels to text labels
df["sentiment"] = df["category"].map({
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
})

print("\nSentiment Counts:")
print(df["sentiment"].value_counts())

plt.figure(figsize=(7,5))

sns.countplot(
    x='sentiment',
    hue='sentiment',
    data=df,
    palette='Set2',
    legend=False
)

plt.title("Sentiment Distribution")
plt.savefig("outputs/sentiment_distribution.png")
plt.close()
print("\nGraph Saved Successfully!")
print("\nStarting Text Preprocessing...")

ps = PorterStemmer()

stop_words = set(stopwords.words('english'))

def preprocess_text(text):

    # lowercase
    text = text.lower()

    # remove numbers and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # tokenize
    words = text.split()

    # remove stopwords + stemming
    words = [
        ps.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

print("\nCleaning Text...")

df["cleaned_text"] = df["clean_text"].apply(preprocess_text)

print("\nFirst 5 Cleaned Texts:\n")

print(
    df[["clean_text","cleaned_text"]].head()
)

positive_text = " ".join(
    df[df["sentiment"]=="Positive"]["cleaned_text"]
)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white'
).generate(positive_text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Word Cloud")
plt.savefig("outputs/wordcloud_positive.png")
plt.close()

negative_text = " ".join(
    df[df["sentiment"]=="Negative"]["cleaned_text"]
)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='black'
).generate(negative_text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Word Cloud")
plt.savefig("outputs/wordcloud_negative.png")
plt.close()

X = df["cleaned_text"]
y = df["category"]

vectorizer = TfidfVectorizer(
    max_features=5000
)

X = vectorizer.fit_transform(X)

print("\nTF-IDF Shape:")
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

print("\n==============================")
print("Multinomial Naive Bayes")
print("==============================")

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test,y_pred_nb))

print("\nPrecision:")
print(precision_score(y_test,y_pred_nb,average='weighted'))

print("\nRecall:")
print(recall_score(y_test,y_pred_nb,average='weighted'))

print("\nF1 Score:")
print(f1_score(y_test,y_pred_nb,average='weighted'))

print("\nClassification Report:")
print(classification_report(y_test,y_pred_nb))

print("\n==============================")
print("Logistic Regression")
print("==============================")

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train,y_train)

y_pred_lr = lr_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test,y_pred_lr))

print("\nPrecision:")
print(precision_score(y_test,y_pred_lr,average='weighted'))

print("\nRecall:")
print(recall_score(y_test,y_pred_lr,average='weighted'))

print("\nF1 Score:")
print(f1_score(y_test,y_pred_lr,average='weighted'))

print("\n==============================")
print("Linear SVM")
print("==============================")

svm_model = LinearSVC()

svm_model.fit(X_train,y_train)

y_pred_svm = svm_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test,y_pred_svm))

print("\nPrecision:")
print(precision_score(y_test,y_pred_svm,average='weighted'))

print("\nRecall:")
print(recall_score(y_test,y_pred_svm,average='weighted'))

print("\nF1 Score:")
print(f1_score(y_test,y_pred_svm,average='weighted'))

print("\n==============================")
print("Random Forest")
print("==============================")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train,y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test,y_pred_rf))

print("\nPrecision:")
print(precision_score(y_test,y_pred_rf,average='weighted'))

print("\nRecall:")
print(recall_score(y_test,y_pred_rf,average='weighted'))

print("\nF1 Score:")
print(f1_score(y_test,y_pred_rf,average='weighted'))

cm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Linear SVM Confusion Matrix")

plt.savefig("outputs/confusion_matrix_svm.png")
plt.close()

models = [
    "Naive Bayes",
    "Logistic Regression",
    "Linear SVM",
    "Random Forest"
]

accuracies = [
    accuracy_score(y_test,y_pred_nb),
    accuracy_score(y_test,y_pred_lr),
    accuracy_score(y_test,y_pred_svm),
    accuracy_score(y_test,y_pred_rf)
]

plt.figure(figsize=(10,6))

plt.bar(models, accuracies)

plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.savefig("outputs/model_comparison.png")
plt.close()

print("\n========== BUSINESS INSIGHTS ==========\n")

print("1. Positive sentiments dominate social media discussions.")

print("\n2. Negative sentiments reveal user dissatisfaction and concerns.")

print("\n3. Linear SVM achieved the best performance among all models.")

print("\n4. NLP techniques effectively classify sentiments from text.")

print("\n5. Automated sentiment analysis can monitor public opinion in real time.")

print("\n========== RECOMMENDATIONS ==========\n")

print("• Use Linear SVM for deployment.")

print("\n• Continuously monitor customer feedback and social media trends.")

print("\n• Focus on negative sentiment tweets for improvement.")

print("\n• Retrain the model periodically with fresh data.")

print("\n• Integrate sentiment analysis into dashboards and analytics systems.")

positive_words = " ".join(
    df[df["sentiment"]=="Positive"]["cleaned_text"]
)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white'
).generate(positive_words)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Word Cloud")
plt.savefig("outputs/wordcloud_positive.png")
plt.close()

negative_words = " ".join(
    df[df["sentiment"]=="Negative"]["cleaned_text"]
)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='black'
).generate(negative_words)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Word Cloud")
plt.savefig("outputs/wordcloud_negative.png")
plt.close()

cm = confusion_matrix(y_test,y_pred_svm)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Linear SVM Confusion Matrix")

plt.savefig("outputs/confusion_matrix_svm.png")
plt.close()

models = [
    "Naive Bayes",
    "Logistic Regression",
    "Linear SVM",
    "Random Forest"
]

accuracies = [
    accuracy_score(y_test,y_pred_nb),
    accuracy_score(y_test,y_pred_lr),
    accuracy_score(y_test,y_pred_svm),
    accuracy_score(y_test,y_pred_rf)
]

plt.figure(figsize=(10,6))

plt.bar(models, accuracies)

plt.ylabel("Accuracy")
plt.title("Model Comparison")

plt.savefig("outputs/model_comparison.png")
plt.close()











