import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Load processed CSVs
train_df = pd.read_csv("data/atis_train.csv")
val_df   = pd.read_csv("data/atis_val.csv")
test_df  = pd.read_csv("data/atis_test.csv")

# 2. Extract text + labels
X_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
X_val,   y_val   = val_df["text"].tolist(),   val_df["label"].tolist()
X_test,  y_test  = test_df["text"].tolist(),  test_df["label"].tolist()

# 3. TF–IDF Vectorizer (standard baseline choice)
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1,2),      # unigrams + bigrams improves LR significantly
    min_df=2                # ignore ultra-rare tokens
)

# Fit only on training
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)
X_test_vec  = vectorizer.transform(X_test)

# 4. Logistic Regression classifier
clf = LogisticRegression(
    max_iter=2000,          # needs high iteration count for convergence
    class_weight="balanced",# because ATIS intents are imbalanced
    n_jobs=-1
)

clf.fit(X_train_vec, y_train)

# 5. Evaluate
print("Validation Performance")
val_pred = clf.predict(X_val_vec)
print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred))

print("\nTest Performance")
test_pred = clf.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# 6. Confusion Matrix (for your report)
print("\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, test_pred))
