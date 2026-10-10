import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/reviews.csv")

print("Dataset loaded successfully!")
print("Total reviews:", len(df))


# --------------------------------------------------
# 2. Input and Output
# --------------------------------------------------

X = df["review"]
y = df["sentiment"]


# --------------------------------------------------
# 3. Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))


# --------------------------------------------------
# 4. TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF conversion completed.")


# --------------------------------------------------
# 5. Train Model
# --------------------------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed.")


# --------------------------------------------------
# 6. Predictions
# --------------------------------------------------

y_pred = model.predict(
    X_test_tfidf
)


# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred
)

matrix = confusion_matrix(
    y_test,
    y_pred
)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(matrix)


# --------------------------------------------------
# 8. Save Model
# --------------------------------------------------

joblib.dump(
    model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


# --------------------------------------------------
# 9. Save Evaluation Results
# --------------------------------------------------

evaluation = {
    "accuracy": accuracy,
    "classification_report": report,
    "confusion_matrix": matrix
}

joblib.dump(
    evaluation,
    "models/evaluation.pkl"
)


print("\n==============================")
print("MODEL FILES SAVED")
print("==============================")

print("sentiment_model.pkl")
print("tfidf_vectorizer.pkl")
print("evaluation.pkl")