import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("unigram_freq.csv")

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


print("\n==============================")
print("NLP PREPROCESSING")
print("==============================")

# Remove missing values
df.dropna(inplace=True)

# Lowercase words
df["word"] = df["word"].astype(str).str.lower()

# Remove duplicates
df.drop_duplicates(subset=["word"], inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

print("\nShape After Cleaning:")
print(df.shape)

print("\nVocabulary Statistics")
print("---------------------")

print("Total Words:", len(df))
print("Unique Words:", df["word"].nunique())


top_words = df.sort_values(
    by="count",
    ascending=False
)

print("\nTop 20 Most Frequent Words:")
print(top_words.head(20))


# ==============================

# PHASE 2 : VISUALIZATION

# ==============================

print("\n==============================")
print("WORD FREQUENCY ANALYSIS")
print("==============================")

# Top 20 Words

top_20 = df.sort_values(
by="count",
ascending=False
).head(20)

plt.figure(figsize=(12,6))

sns.barplot(
x="count",
y="word",
data=top_20
)

plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.ylabel("Word")

plt.tight_layout()

plt.savefig(
"outputs/top_words_frequency.png"
)

plt.show()

# ==============================

# Word Length Distribution

# ==============================

df["word_length"] = df["word"].apply(len)

plt.figure(figsize=(10,6))

sns.histplot(
df["word_length"],
bins=25,
kde=True
)

plt.title("Word Length Distribution")

plt.xlabel("Word Length")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
"outputs/word_length_distribution.png"
)

plt.show()

# ==============================

# Prefix Distribution

# ==============================

df["prefix"] = df["word"].str[0]

top_prefixes = (
df["prefix"]
.value_counts()
.head(15)
)

plt.figure(figsize=(10,6))

sns.barplot(
x=top_prefixes.index,
y=top_prefixes.values
)

plt.title("Most Common Starting Letters")

plt.xlabel("Prefix")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
"outputs/prefix_distribution.png"
)

plt.show()

# ==============================

# Vocabulary Statistics

# ==============================

stats = {

"Total Words": len(df),

"Unique Words": df["word"].nunique(),

"Average Word Length":
    round(
        df["word_length"].mean(),
        2
    ),

"Maximum Word Length":
    df["word_length"].max(),

"Minimum Word Length":
    df["word_length"].min()

}

print("\nVocabulary Statistics")

for k, v in stats.items():
    print(k, ":", v)


# ==============================

# PHASE 3 : AUTOCOMPLETE ENGINE

# ==============================

print("\n==============================")
print("AUTOCOMPLETE ENGINE")
print("==============================")

def autocomplete(prefix, top_n=10):

    prefix = prefix.lower()

    matches = df[
        df["word"].str.startswith(prefix)
    ]

    matches = matches.sort_values(
        by="count",
        ascending=False
    )

    return matches["word"].head(top_n).tolist()


# -----------------------------

# TEST EXAMPLES

# -----------------------------

test_prefixes = [
"app",
"data",
"mach",
"comp",
"art"
]

for p in test_prefixes:
    suggestions = autocomplete(p)

print("\nInput Prefix:", p)

print("Suggestions:")

for word in suggestions:
    print(word)


# -----------------------------

# AUTOCOMPLETE VISUALIZATION

# -----------------------------

example_data = {
"Prefix": ["app", "data", "mach", "comp", "art"],
"Suggestions": [10, 10, 10, 10, 10]
}

example_df = pd.DataFrame(example_data)

plt.figure(figsize=(8,5))

sns.barplot(
x="Prefix",
y="Suggestions",
data=example_df
)

plt.title("Autocomplete Suggestions")

plt.tight_layout()

plt.savefig(
"outputs/autocomplete_examples.png"
)

plt.show()

# ==============================

# PHASE 4 : AUTOCORRECT ENGINE

# ==============================

from difflib import get_close_matches

print("\n==============================")
print("AUTOCORRECT ENGINE")
print("==============================")

# -----------------------------

# Build Vocabulary

# -----------------------------

vocabulary = (
    df[
        (df["word"].str.len() > 2)
        & (df["count"] > 100000)
    ]["word"]
    .tolist()
)

# -----------------------------

# Autocorrect Function

# -----------------------------

def autocorrect(word, top_n=5):

    word = word.lower()

    suggestions = get_close_matches(
        word,
        vocabulary,
        n=top_n,
        cutoff=0.6
    )

    return suggestions


# -----------------------------

# Test Examples

# -----------------------------

test_words = [
    "aplpe",
    "machne",
    "databse",
    "artifical",
    "computr"
]

for word in test_words:

    print("\nInput:", word)

    suggestions = autocorrect(word)

    print("Suggestions:")

    if suggestions:

        for s in suggestions:
            print(s)

    else:

        print("No suggestions found")


# -----------------------------

# Visualization

# -----------------------------

corrections = {
"aplpe": 1,
"machne": 1,
"databse": 1,
"artifical": 1,
"computr": 1
}

plt.figure(figsize=(8,5))

sns.barplot(
x=list(corrections.keys()),
y=list(corrections.values())
)

plt.title("Autocorrect Examples")

plt.ylabel("Corrected")

plt.tight_layout()

plt.savefig(
"outputs/autocorrect_examples.png"
)

plt.show()




# ==============================

# PHASE 5 : INTERACTIVE MODE

# ==============================

print("\n==============================")
print("INTERACTIVE MODE")
print("==============================")

while True:
    user_input = input(
        "\nEnter a word or prefix (type exit to quit): "
        ).lower()
    if user_input == "exit":
        print("\nProgram Closed Successfully.")
        break
    print("\nAutocomplete Suggestions:")
    auto_results = autocomplete(user_input)
    
    if len(auto_results) == 0:
        print("No autocomplete suggestions found")
        
    else:
        for item in auto_results:
            print(item)
            
    print("\nAutocorrect Suggestions:")
    correct_results = autocorrect(user_input)
    if len(correct_results) == 0:
        print("No autocorrect suggestions found")

    else:
        for item in correct_results:
            print(item)


