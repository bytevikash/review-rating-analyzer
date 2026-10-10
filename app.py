import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ReviewAI | Review Analytics",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROFESSIONAL UI THEME
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0b0f19;
        color: #e8edf7;
    }

    .block-container {
        max-width: 1500px;
        padding: 1.7rem 2rem 1.2rem;
    }

    [data-testid="stSidebar"] {
        background: #101624;
        border-right: 1px solid #222c3d;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    [data-testid="stSidebar"] hr,
    hr {
        border-color: #273044 !important;
    }

    h1, h2, h3 {
        color: #f4f7fc !important;
        letter-spacing: -0.4px;
    }

    h1 { font-size: 1.8rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.35rem !important; font-weight: 650 !important; }
    h3 { font-size: 1.05rem !important; font-weight: 600 !important; }

    p, label {
        color: #c1cbda;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #aab6c9;
        font-size: 0.88rem;
    }

    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 5px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 8px 10px;
        border-radius: 8px;
        border: 1px solid transparent;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #1c2739;
    }

    [data-testid="stMetric"] {
        background: #131b2a;
        border: 1px solid #263248;
        border-radius: 12px;
        padding: 14px 16px;
        min-height: 100px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.12);
    }

    [data-testid="stMetricLabel"] {
        color: #9eacc1 !important;
        font-size: 0.82rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #f5f8ff !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }

    .stTextInput input,
    .stTextArea textarea,
    [data-baseweb="select"] > div {
        background: #131b2a;
        border-color: #303c52;
        border-radius: 8px;
        color: #edf2fa;
    }

    .stTextArea textarea {
        min-height: 145px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 8px;
        min-height: 40px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: #5579e8;
        color: white;
        border: 1px solid #5579e8;
    }

    .stButton > button[kind="primary"]:hover {
        background: #4165d1;
        border-color: #4165d1;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #273248;
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 9px;
    }

    .dashboard-eyebrow {
        color: #88a5ff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .dashboard-title {
        font-size: 30px;
        line-height: 1.25;
        font-weight: 750;
        letter-spacing: -0.8px;
        color: #f4f7fc;
    }

    .dashboard-subtitle {
        font-size: 14px;
        color: #94a2b8;
        margin-top: 7px;
        margin-bottom: 20px;
    }

    .sidebar-brand {
        font-size: 25px;
        font-weight: 750;
        color: #f5f7ff;
    }

    .sidebar-caption {
        font-size: 11px;
        color: #8492aa;
        letter-spacing: 1px;
        margin-top: 5px;
    }

    .sidebar-status {
        margin-top: 24px;
        padding: 12px;
        background: #151e2e;
        border: 1px solid #28354b;
        border-radius: 9px;
        font-size: 12px;
        color: #a6e3bd;
    }

    .section-note {
        color: #94a2b8;
        font-size: 13px;
        margin-top: -6px;
        margin-bottom: 18px;
    }

    .stPlotlyChart {
        border: 1px solid #222d40;
        border-radius: 10px;
        padding: 4px;
        background: #101624;
    }

    .dashboard-footer {
        color: #78869d;
        font-size: 12px;
        text-align: center;
        padding: 12px 0 4px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1.1rem 1rem;
        }

        .dashboard-title {
            font-size: 25px;
        }

        [data-testid="stMetric"] {
            padding: 10px;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOAD DATA AND MODELS
# =========================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("data/reviews.csv")


@st.cache_resource
def load_models():
    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    evaluation = joblib.load("models/evaluation.pkl")
    return model, vectorizer, evaluation


try:
    df = load_dataset()
    model, vectorizer, evaluation = load_models()
except FileNotFoundError as exc:
    st.error(f"Required file not found: {exc.filename}")
    st.info(
        "Check that data/reviews.csv and these model files exist: "
        "models/sentiment_model.pkl, models/tfidf_vectorizer.pkl, "
        "models/evaluation.pkl."
    )
    st.stop()
except Exception as exc:
    st.error(f"Could not load the dataset or model: {exc}")
    st.stop()

# =========================================================
# VALIDATE AND PREPARE DATA
# =========================================================

required_columns = {
    "review_id", "review", "rating", "sentiment", "date", "category"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        "Dataset is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

df = df.copy()
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["sentiment"] = (
    df["sentiment"].fillna("unknown").astype(str).str.lower().str.strip()
)
df["category"] = df["category"].fillna("Unknown").astype(str)
df["review"] = df["review"].fillna("").astype(str)

total_reviews = len(df)
average_rating = df["rating"].mean()
positive_reviews = int((df["sentiment"] == "positive").sum())
neutral_reviews = int((df["sentiment"] == "neutral").sum())
negative_reviews = int((df["sentiment"] == "negative").sum())

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral": "#f59e0b",
    "negative": "#ef4444",
    "unknown": "#94a3b8",
}

PLOT_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#dce5f3"},
    "margin": {"t": 55, "b": 35, "l": 35, "r": 20},
}

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:4px 0 16px;">
            <div class="sidebar-brand">
                <span style="color:#83a0ff;">R</span>eviewAI
            </div>
            <div class="sidebar-caption">
                CUSTOMER INSIGHTS PLATFORM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Analyze Review",
            "Dashboard",
            "Review Dataset",
            "Rating Analysis",
            "Category Analysis",
            "Model Performance",
        ],
        index=0,
        label_visibility="collapsed",
        key="navigation",
    )

    st.divider()
    st.caption("MODEL INFORMATION")
    st.markdown("**TF-IDF + Logistic Regression**")
    st.caption(f"{total_reviews:,} reviews loaded")

    st.divider()

    if st.button("↻  Refresh Dataset", use_container_width=True):
        load_dataset.clear()
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-status">
            ● &nbsp; Dataset loaded
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# SHARED HEADER AND KPI CARDS
# =========================================================

st.markdown(
    """
    <div class="dashboard-eyebrow">CUSTOMER INTELLIGENCE</div>
    <div class="dashboard-title">Review Analytics</div>
    <div class="dashboard-subtitle">
        Monitor customer feedback, analyze sentiment, and understand product ratings.
    </div>
    """,
    unsafe_allow_html=True,
)

metric1, metric2, metric3, metric4 = st.columns(4, gap="medium")

with metric1:
    st.metric("Total Reviews", f"{total_reviews:,}")

with metric2:
    st.metric(
        "Average Rating",
        f"{average_rating:.2f} / 5" if pd.notna(average_rating) else "N/A",
    )

with metric3:
    st.metric("Positive Reviews", f"{positive_reviews:,}")

with metric4:
    st.metric("Negative Reviews", f"{negative_reviews:,}")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.divider()

# =========================================================
# PAGE 1: ANALYZE REVIEW
# =========================================================

if page == "Analyze Review":
    st.subheader("Analyze a Customer Review")
    st.markdown(
        '<div class="section-note">'
        "Enter a product review to predict its sentiment using your trained model."
        "</div>",
        unsafe_allow_html=True,
    )

    input_col, overview_col = st.columns([1.1, 0.9], gap="large")

    with input_col:
        with st.form("review_analysis_form"):
            new_review = st.text_area(
                "Review text",
                placeholder=(
                    "Example: The product quality is excellent and "
                    "delivery was fast. I really love it!"
                ),
                height=165,
            )
            analyze_clicked = st.form_submit_button(
                "Analyze Review",
                type="primary",
                use_container_width=True,
            )

        if analyze_clicked:
            if not new_review.strip():
                st.warning("Please enter a review before analyzing.")
            else:
                try:
                    review_tfidf = vectorizer.transform([new_review])
                    prediction = str(model.predict(review_tfidf)[0]).lower()

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(review_tfidf)[0]
                        probability_map = {
                            str(label).lower(): float(probability * 100)
                            for label, probability in zip(
                                model.classes_, probabilities
                            )
                        }
                        confidence = max(probability_map.values())
                    else:
                        probability_map = {}
                        confidence = None

                    st.session_state["last_prediction"] = {
                        "review": new_review.strip(),
                        "sentiment": prediction,
                        "confidence": confidence,
                        "probabilities": probability_map,
                    }
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    with overview_col:
        st.markdown("### Sentiment Overview")

        sentiment_data = pd.DataFrame(
            {
                "Sentiment": ["Positive", "Neutral", "Negative"],
                "Reviews": [
                    positive_reviews,
                    neutral_reviews,
                    negative_reviews,
                ],
            }
        )

        if sentiment_data["Reviews"].sum() > 0:
            sentiment_chart = px.pie(
                sentiment_data,
                names="Sentiment",
                values="Reviews",
                hole=0.62,
                color="Sentiment",
                color_discrete_map={
                    "Positive": SENTIMENT_COLORS["positive"],
                    "Neutral": SENTIMENT_COLORS["neutral"],
                    "Negative": SENTIMENT_COLORS["negative"],
                },
            )
            sentiment_chart.update_layout(
                **PLOT_LAYOUT,
                height=285,
                showlegend=True,
                legend={"orientation": "h", "y": -0.12},
            )
            st.plotly_chart(
                sentiment_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No sentiment records are available.")

    result = st.session_state.get("last_prediction")

    if result:
        st.divider()
        st.subheader("Prediction Result")

        result_col, probability_col = st.columns([0.85, 1.15], gap="large")

        with result_col:
            predicted_sentiment = result["sentiment"]

            if predicted_sentiment == "positive":
                st.success(f"Predicted sentiment: {predicted_sentiment.upper()}")
            elif predicted_sentiment == "negative":
                st.error(f"Predicted sentiment: {predicted_sentiment.upper()}")
            else:
                st.info(f"Predicted sentiment: {predicted_sentiment.upper()}")

            if result["confidence"] is not None:
                st.metric("Prediction Confidence", f"{result['confidence']:.2f}%")

            st.caption("Review analyzed")
            st.write(result["review"])

        with probability_col:
            probability_map = result["probabilities"]

            if probability_map:
                probability_df = pd.DataFrame(
                    {
                        "Sentiment": list(probability_map.keys()),
                        "Probability": list(probability_map.values()),
                    }
                )

                probability_chart = px.bar(
                    probability_df,
                    x="Sentiment",
                    y="Probability",
                    color="Sentiment",
                    text=probability_df["Probability"].map(
                        lambda value: f"{value:.1f}%"
                    ),
                    color_discrete_map=SENTIMENT_COLORS,
                    title="Prediction Probability",
                )
                probability_chart.update_layout(
                    **PLOT_LAYOUT,
                    height=310,
                    showlegend=False,
                    yaxis_title="Probability (%)",
                    yaxis_range=[0, 100],
                )
                st.plotly_chart(
                    probability_chart,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            else:
                st.info("This model does not provide probability estimates.")

# =========================================================
# PAGE 2: DASHBOARD
# =========================================================

elif page == "Dashboard":
    st.subheader("Analytics Overview")
    st.markdown(
        '<div class="section-note">'
        "A high-level view of customer ratings and feedback."
        "</div>",
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        rating_counts = (
            df["rating"].dropna().value_counts().sort_index()
        )

        if not rating_counts.empty:
            rating_chart = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={"x": "Star Rating", "y": "Number of Reviews"},
                title="Rating Distribution",
                text=rating_counts.values,
                color_discrete_sequence=["#7da8ff"],
            )
            rating_chart.update_layout(**PLOT_LAYOUT, height=340)
            st.plotly_chart(
                rating_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No numeric ratings are available.")

    with chart_col2:
        sentiment_counts = df["sentiment"].value_counts()

        if not sentiment_counts.empty:
            sentiment_chart = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                hole=0.58,
                title="Customer Sentiment",
                color=sentiment_counts.index,
                color_discrete_map=SENTIMENT_COLORS,
            )
            sentiment_chart.update_layout(**PLOT_LAYOUT, height=340)
            st.plotly_chart(
                sentiment_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No sentiment labels are available.")

    st.subheader("Recent Reviews")

    recent_reviews = df.copy()
    recent_reviews["date"] = pd.to_datetime(
        recent_reviews["date"], errors="coerce"
    )
    recent_reviews = recent_reviews.sort_values(
        "date", ascending=False, na_position="last"
    )

    st.dataframe(
        recent_reviews.drop(columns=["date"], errors="ignore")
        .assign(Date=recent_reviews["date"].dt.strftime("%Y-%m-%d").values)
        .head(8),
        use_container_width=True,
        hide_index=True,
        height=330,
    )

# =========================================================
# PAGE 3: REVIEW DATASET
# =========================================================

elif page == "Review Dataset":
    st.subheader("Review Dataset")
    st.markdown(
        '<div class="section-note">'
        f"Search and filter the available {total_reviews:,} customer reviews."
        "</div>",
        unsafe_allow_html=True,
    )

    search_text = st.text_input(
        "Search reviews",
        placeholder="Search review text, category, ID or sentiment...",
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    sentiment_options = ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
    category_options = ["All"] + sorted(df["category"].dropna().unique().tolist())
    rating_options = ["All"] + sorted(df["rating"].dropna().unique().tolist())

    with filter_col1:
        selected_sentiment = st.selectbox("Sentiment", sentiment_options)

    with filter_col2:
        selected_category = st.selectbox("Category", category_options)

    with filter_col3:
        selected_rating = st.selectbox("Rating", rating_options)

    filtered_df = df.copy()

    if search_text.strip():
        search_mask = (
            filtered_df.astype(str)
            .apply(
                lambda column: column.str.contains(
                    search_text, case=False, na=False, regex=False
                )
            )
            .any(axis=1)
        )
        filtered_df = filtered_df[search_mask]

    if selected_sentiment != "All":
        filtered_df = filtered_df[
            filtered_df["sentiment"] == selected_sentiment
        ]

    if selected_category != "All":
        filtered_df = filtered_df[
            filtered_df["category"] == selected_category
        ]

    if selected_rating != "All":
        filtered_df = filtered_df[
            filtered_df["rating"] == selected_rating
        ]

    st.write(f"**{len(filtered_df):,} matching reviews**")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        height=410,
    )

    st.download_button(
        "Download filtered CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_reviews.csv",
        mime="text/csv",
    )

# =========================================================
# PAGE 4: RATING ANALYSIS
# =========================================================

elif page == "Rating Analysis":
    st.subheader("Rating Analysis")
    st.markdown(
        '<div class="section-note">'
        "Understand rating patterns and how they relate to sentiment."
        "</div>",
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        rating_counts = df["rating"].dropna().value_counts().sort_index()

        if not rating_counts.empty:
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={"x": "Rating", "y": "Number of Reviews"},
                title="Distribution of Ratings",
                text=rating_counts.values,
                color_discrete_sequence=["#7da8ff"],
            )
            fig.update_layout(**PLOT_LAYOUT, height=360)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No rating data is available.")

    with chart_col2:
        rating_sentiment = pd.crosstab(df["rating"], df["sentiment"])

        if not rating_sentiment.empty:
            fig2 = px.bar(
                rating_sentiment,
                barmode="group",
                title="Rating vs Sentiment",
                labels={
                    "value": "Number of Reviews",
                    "rating": "Rating",
                    "variable": "Sentiment",
                },
                color_discrete_map=SENTIMENT_COLORS,
            )
            fig2.update_layout(**PLOT_LAYOUT, height=360)
            st.plotly_chart(
                fig2,
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("No rating and sentiment combinations are available.")

    st.subheader("Rating and Sentiment Summary")
    st.dataframe(
        pd.crosstab(df["rating"], df["sentiment"]),
        use_container_width=True,
    )

# =========================================================
# PAGE 5: CATEGORY ANALYSIS
# =========================================================

elif page == "Category Analysis":
    st.subheader("Category Analysis")
    st.markdown(
        '<div class="section-note">'
        "Compare customer ratings and review volumes across categories."
        "</div>",
        unsafe_allow_html=True,
    )

    category_summary = (
        df.groupby("category", dropna=False)
        .agg(
            Average_Rating=("rating", "mean"),
            Total_Reviews=("review_id", "count"),
        )
        .reset_index()
        .sort_values("Average_Rating", ascending=False, na_position="last")
    )

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        fig = px.bar(
            category_summary,
            x="category",
            y="Average_Rating",
            title="Average Rating by Category",
            labels={
                "category": "Category",
                "Average_Rating": "Average Rating",
            },
            text=category_summary["Average_Rating"].round(2),
            color_discrete_sequence=["#7da8ff"],
        )
        fig.update_layout(
            **PLOT_LAYOUT,
            height=360,
            yaxis_range=[0, 5],
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with chart_col2:
        fig2 = px.bar(
            category_summary,
            x="category",
            y="Total_Reviews",
            title="Reviews by Category",
            labels={
                "category": "Category",
                "Total_Reviews": "Total Reviews",
            },
            text="Total_Reviews",
            color_discrete_sequence=["#6fc7b7"],
        )
        fig2.update_layout(**PLOT_LAYOUT, height=360)
        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.subheader("Category Summary")
    st.dataframe(
        category_summary.round({"Average_Rating": 2}),
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# PAGE 6: MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":
    st.subheader("Machine Learning Model Performance")
    st.markdown(
        '<div class="section-note">'
        "Evaluation results saved with the trained sentiment classifier."
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        accuracy = float(evaluation["accuracy"])
        accuracy_col, model_col = st.columns(2, gap="medium")

        accuracy_col.metric("Model Accuracy", f"{accuracy * 100:.2f}%")
        model_col.metric("Model", "TF-IDF + Logistic Regression")

        st.caption(
            "Accuracy refers to the saved evaluation dataset; "
            "it does not guarantee the same performance on new reviews."
        )

        st.subheader("Classification Report")
        st.code(
            evaluation["classification_report"],
            language="text",
        )

        st.subheader("Confusion Matrix")

        matrix = evaluation["confusion_matrix"]
        labels = ["Negative", "Neutral", "Positive"]

        confusion_df = pd.DataFrame(
            matrix,
            index=labels,
            columns=labels,
        )

        matrix_col, heatmap_col = st.columns([0.9, 1.1], gap="large")

        with matrix_col:
            st.dataframe(
                confusion_df,
                use_container_width=True,
            )

        with heatmap_col:
            fig = px.imshow(
                confusion_df,
                text_auto=True,
                labels={
                    "x": "Predicted Sentiment",
                    "y": "Actual Sentiment",
                    "color": "Review Count",
                },
                x=labels,
                y=labels,
                title="Confusion Matrix Heatmap",
                color_continuous_scale="Blues",
            )
            fig.update_layout(**PLOT_LAYOUT, height=350)
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    except (KeyError, TypeError, ValueError) as exc:
        st.error(f"Could not display model evaluation results: {exc}")

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div class="dashboard-footer">
        ReviewAI &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Plotly
        &nbsp;·&nbsp; Scikit-learn
    </div>
    """,
    unsafe_allow_html=True,
)
