import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.metrics import classification_report

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Online Review Rating Analyzer",
    page_icon="⭐",
    layout="wide"
)

st.title("⭐ Online Review Rating Pattern Analyzer")
st.write(
    "Analyze online reviews, ratings, and customer sentiment using Machine Learning."
)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv("data/reviews.csv")


# --------------------------------------------------
# Load ML Model
# --------------------------------------------------

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
evaluation = joblib.load("models/evaluation.pkl")


# --------------------------------------------------
# Review Dataset
# --------------------------------------------------

st.subheader("📋 Review Dataset")

st.dataframe(
    df,
    use_container_width=True
)


# --------------------------------------------------
# Overall Statistics
# --------------------------------------------------

total_reviews = len(df)

average_rating = df["rating"].mean()

positive_reviews = len(
    df[df["sentiment"] == "positive"]
)

neutral_reviews = len(
    df[df["sentiment"] == "neutral"]
)

negative_reviews = len(
    df[df["sentiment"] == "negative"]
)


st.subheader("📊 Overall Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Reviews",
    total_reviews
)

col2.metric(
    "Average Rating",
    f"{average_rating:.2f} ⭐"
)

col3.metric(
    "Positive Reviews",
    positive_reviews
)

col4.metric(
    "Negative Reviews",
    negative_reviews
)


# --------------------------------------------------
# Rating Distribution
# --------------------------------------------------

st.subheader("⭐ Rating Distribution")

rating_counts = (
    df["rating"]
    .value_counts()
    .sort_index()
)

rating_chart = px.bar(
    x=rating_counts.index,
    y=rating_counts.values,
    labels={
        "x": "Rating",
        "y": "Number of Reviews"
    },
    title="Distribution of Ratings"
)

st.plotly_chart(
    rating_chart,
    use_container_width=True
)


# --------------------------------------------------
# Sentiment Distribution
# --------------------------------------------------

st.subheader("😊 Sentiment Distribution")

sentiment_counts = df["sentiment"].value_counts()

sentiment_chart = px.pie(
    values=sentiment_counts.values,
    names=sentiment_counts.index,
    title="Customer Sentiment"
)

st.plotly_chart(
    sentiment_chart,
    use_container_width=True
)


# --------------------------------------------------
# Rating vs Sentiment
# --------------------------------------------------

st.subheader("🔍 Rating vs Sentiment")

rating_sentiment = pd.crosstab(
    df["rating"],
    df["sentiment"]
)

st.dataframe(
    rating_sentiment,
    use_container_width=True
)

comparison_chart = px.bar(
    rating_sentiment,
    barmode="group",
    title="Relationship Between Rating and Sentiment"
)

st.plotly_chart(
    comparison_chart,
    use_container_width=True
)


# --------------------------------------------------
# Category Analysis
# --------------------------------------------------

st.subheader("📦 Category Analysis")

category_rating = (
    df.groupby("category")["rating"]
    .mean()
    .reset_index()
)

category_chart = px.bar(
    category_rating,
    x="category",
    y="rating",
    title="Average Rating by Category",
    labels={
        "rating": "Average Rating",
        "category": "Category"
    }
)

st.plotly_chart(
    category_chart,
    use_container_width=True
)


# --------------------------------------------------
# ML Prediction
# --------------------------------------------------

# --------------------------------------------------
# Machine Learning Model Performance
# --------------------------------------------------

st.subheader("🤖 Machine Learning Model Performance")

accuracy = evaluation["accuracy"]

col1, col2 = st.columns(2)

col1.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Model",
    "TF-IDF + Logistic Regression"
)

st.write("### Classification Report")

report_text = evaluation["classification_report"]

st.code(
    report_text,
    language="text"
)

st.write("### Confusion Matrix")

confusion_matrix_data = evaluation["confusion_matrix"]

confusion_df = pd.DataFrame(
    confusion_matrix_data,
    index=["Negative", "Neutral", "Positive"],
    columns=["Negative", "Neutral", "Positive"]
)

st.dataframe(
    confusion_df,
    use_container_width=True
)

st.subheader("🤖 Test a New Review")

new_review = st.text_area(
    "Enter a review below:",
    placeholder="Example: The product is excellent and I really love it!"
)


if st.button("Analyze Review"):

    if new_review.strip():

        # Convert review into TF-IDF
        review_tfidf = vectorizer.transform(
            [new_review]
        )

        # Predict sentiment
        prediction = model.predict(
            review_tfidf
        )[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(
            review_tfidf
        )[0]

        # Get confidence
        confidence = probabilities.max() * 100

        # Display result
        st.success(
            f"Predicted Sentiment: {prediction.upper()}"
        )

        st.info(
            f"Prediction Confidence: {confidence:.2f}%"
        )

        # Show probability for each sentiment
        st.write("### Sentiment Probabilities")

        probability_df = pd.DataFrame({
            "Sentiment": model.classes_,
            "Probability": probabilities * 100
        })

        probability_chart = px.bar(
            probability_df,
            x="Sentiment",
            y="Probability",
            title="Sentiment Prediction Probability",
            labels={
                "Probability": "Probability (%)"
            }
        )

        st.plotly_chart(
            probability_chart,
            use_container_width=True

        )

    else:

        st.warning(
            "Please enter a review first."
        )