from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Potato Price Intelligence",
    page_icon="🥔",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .subtitle {
        color: #6b7280;
        margin-top: 0.2rem;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. DATASET FINDER
#    Works whether CSV is in repository root or data/
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

EXPECTED_FILE_NAMES = [
    "Potato_Price_TimeSeries_Weather_Dataset.csv",
    "Potato_Price_TimeSeries_Weather_Dataset.CSV",
]


def find_dataset():
    """Find the project CSV automatically."""

    # A. Repository root
    for name in EXPECTED_FILE_NAMES:
        path = BASE_DIR / name
        if path.is_file():
            return path

    # B. data/ folder
    for name in EXPECTED_FILE_NAMES:
        path = BASE_DIR / "data" / name
        if path.is_file():
            return path

    # C. Any subfolder
    for path in BASE_DIR.rglob("*.csv"):
        if path.name.lower() == EXPECTED_FILE_NAMES[0].lower():
            return path

    return None


# ============================================================
# 3. DATA LOADING
# ============================================================
REQUIRED_COLUMNS = [
    "State",
    "District",
    "Market",
    "Commodity",
    "Variety",
    "Grade",
    "Arrival_Date",
    "Year",
    "Month",
    "Week",
    "Season",
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_percent",
    "Min_Price_Rs_per_Quintal",
    "Max_Price_Rs_per_Quintal",
    "Modal_Price_Rs_per_Quintal",
]


@st.cache_data
def load_dataframe_from_bytes(file_bytes):
    """Read an uploaded CSV from bytes."""
    from io import BytesIO

    df = pd.read_csv(BytesIO(file_bytes))
    return clean_dataframe(df)


def clean_dataframe(df):
    """Clean and validate the potato dataset."""

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        raise ValueError(
            "The uploaded CSV does not match the expected project dataset. "
            f"Missing columns: {', '.join(missing)}"
        )

    df = df.copy()

    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        errors="coerce",
    )

    numeric_columns = [
        "Year",
        "Month",
        "Week",
        "Rainfall_mm",
        "Temperature_C",
        "Humidity_percent",
        "Min_Price_Rs_per_Quintal",
        "Max_Price_Rs_per_Quintal",
        "Modal_Price_Rs_per_Quintal",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_columns = [
        "State",
        "District",
        "Market",
        "Commodity",
        "Variety",
        "Grade",
        "Season",
    ]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df = df.dropna(
        subset=[
            "Arrival_Date",
            "Modal_Price_Rs_per_Quintal",
        ]
    )

    df = df.sort_values("Arrival_Date").reset_index(drop=True)

    return df


DATA_PATH = find_dataset()

if DATA_PATH is not None:
    try:
        df = clean_dataframe(pd.read_csv(DATA_PATH))
        dataset_source = str(DATA_PATH.relative_to(BASE_DIR))
    except Exception as exc:
        st.error(f"Dataset could not be read: {exc}")
        st.stop()
else:
    st.warning(
        "⚠️ The project CSV was not found in the GitHub repository. "
        "Upload the CSV below, or add "
        "`Potato_Price_TimeSeries_Weather_Dataset.csv` to the repository."
    )

    uploaded_file = st.file_uploader(
        "Upload Potato_Price_TimeSeries_Weather_Dataset.csv",
        type=["csv"],
    )

    if uploaded_file is None:
        st.info(
            "Expected file name: "
            "`Potato_Price_TimeSeries_Weather_Dataset.csv`"
        )
        st.stop()

    try:
        df = load_dataframe_from_bytes(uploaded_file.getvalue())
        dataset_source = "Uploaded CSV"
    except Exception as exc:
        st.error(str(exc))
        st.stop()


# ============================================================
# 4. WEEKLY TIME-SERIES FEATURES
# ============================================================
@st.cache_data
def build_weekly_data(data):
    weekly = (
        data.set_index("Arrival_Date")
        .resample("W-FRI")
        .agg(
            price=("Modal_Price_Rs_per_Quintal", "mean"),
            rainfall=("Rainfall_mm", "mean"),
            temperature=("Temperature_C", "mean"),
            humidity=("Humidity_percent", "mean"),
            records=("Modal_Price_Rs_per_Quintal", "size"),
        )
        .reset_index()
    )

    weekly["lag1"] = weekly["price"].shift(1)
    weekly["lag2"] = weekly["price"].shift(2)
    weekly["rolling4"] = weekly["price"].shift(1).rolling(4).mean()

    season_lookup = (
        data.groupby(
            data["Arrival_Date"].dt.to_period("W-FRI")
        )["Season"]
        .agg(
            lambda x: (
                x.mode().iloc[0]
                if not x.mode().empty
                else x.iloc[0]
            )
        )
        .reset_index()
    )

    season_lookup["Arrival_Date"] = (
        season_lookup["Arrival_Date"]
        .dt.end_time
        .dt.normalize()
    )

    weekly = weekly.merge(
        season_lookup,
        on="Arrival_Date",
        how="left",
    )

    weekly = weekly.dropna(
        subset=[
            "lag1",
            "lag2",
            "rolling4",
            "price",
        ]
    ).reset_index(drop=True)

    return weekly


weekly = build_weekly_data(df)

FEATURES = [
    "lag1",
    "lag2",
    "rolling4",
    "rainfall",
    "temperature",
    "humidity",
]


# ============================================================
# 5. MODEL EVALUATION
# ============================================================
def rmse(y_true, y_pred):
    """Version-safe RMSE calculation."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@st.cache_data
def evaluate_models(weekly_df):
    if len(weekly_df) < 10:
        raise ValueError(
            "Not enough weekly records for train/test evaluation."
        )

    # Last 8 weeks are held out as test data.
    n_test = min(8, max(2, len(weekly_df) // 6))

    train = weekly_df.iloc[:-n_test].copy()
    test = weekly_df.iloc[-n_test:].copy()

    X_train = train[FEATURES]
    y_train = train["price"]

    X_test = test[FEATURES]
    y_test = test["price"]

    models = {
        "Linear Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "Ridge Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42,
            n_jobs=-1,
        ),
    }

    result_rows = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        result_rows.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(
                    y_test,
                    prediction,
                ),
                "RMSE": rmse(
                    y_test,
                    prediction,
                ),
                "R2": r2_score(
                    y_test,
                    prediction,
                ),
            }
        )

        fitted_models[name] = model

    return (
        pd.DataFrame(result_rows),
        fitted_models,
        train,
        test,
    )


try:
    model_results, fitted_models, train_df, test_df = evaluate_models(
        weekly
    )
except Exception as exc:
    st.error(f"Model evaluation failed: {exc}")
    st.stop()


# ============================================================
# 6. RANDOM FOREST HYPERPARAMETER EVALUATION
# ============================================================
@st.cache_data
def rf_hyperparameter_sweep(weekly_df):
    n_test = min(8, max(2, len(weekly_df) // 6))

    train = weekly_df.iloc[:-n_test]
    test = weekly_df.iloc[-n_test:]

    X_train = train[FEATURES]
    y_train = train["price"]

    X_test = test[FEATURES]
    y_test = test["price"]

    rows = []

    for n_estimators in [100, 200, 300]:
        for max_depth in [5, 8, 12]:

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )

            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            rows.append(
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "MAE": mean_absolute_error(
                        y_test,
                        prediction,
                    ),
                    "RMSE": rmse(
                        y_test,
                        prediction,
                    ),
                    "R2": r2_score(
                        y_test,
                        prediction,
                    ),
                }
            )

    return pd.DataFrame(rows)


hp_results = rf_hyperparameter_sweep(weekly)

# For demonstration, select the configuration with highest R².
# Ties are resolved using lower RMSE and lower MAE.
best_row = hp_results.sort_values(
    ["R2", "RMSE", "MAE"],
    ascending=[False, True, True],
).iloc[0]


@st.cache_resource
def train_best_random_forest(
    n_estimators,
    max_depth,
    weekly_df,
):
    n_test = min(8, max(2, len(weekly_df) // 6))

    train = weekly_df.iloc[:-n_test]

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        train[FEATURES],
        train["price"],
    )

    return model


best_rf = train_best_random_forest(
    int(best_row["n_estimators"]),
    int(best_row["max_depth"]),
    weekly,
)


# ============================================================
# 7. LIGHTWEIGHT RAG
# ============================================================
def build_rag_documents(data):
    """Create text documents from dataset statistics."""

    documents = []

    latest_date = data["Arrival_Date"].max()
    latest_rows = data[
        data["Arrival_Date"] == latest_date
    ]

    documents.append(
        f"Latest dataset date: {latest_date.date()}. "
        f"Average modal potato price on this date: "
        f"Rs {latest_rows['Modal_Price_Rs_per_Quintal'].mean():.2f} "
        f"per quintal."
    )

    # State documents
    for state in sorted(
        data["State"].dropna().unique()
    ):
        state_df = data[
            data["State"].str.lower()
            == state.lower()
        ]

        latest_state_date = state_df[
            "Arrival_Date"
        ].max()

        latest_state_rows = state_df[
            state_df["Arrival_Date"]
            == latest_state_date
        ]

        documents.append(
            f"State {state}: latest available date "
            f"{latest_state_date.date()}, latest average "
            f"modal price Rs "
            f"{latest_state_rows['Modal_Price_Rs_per_Quintal'].mean():.2f} "
            f"per quintal, overall average Rs "
            f"{state_df['Modal_Price_Rs_per_Quintal'].mean():.2f}."
        )

        # State + season documents
        for season in sorted(
            state_df["Season"].dropna().unique()
        ):
            season_df = state_df[
                state_df["Season"].str.lower()
                == season.lower()
            ]

            documents.append(
                f"State {state} during {season}: "
                f"average modal potato price is Rs "
                f"{season_df['Modal_Price_Rs_per_Quintal'].mean():.2f} "
                f"per quintal."
            )

    # National season documents
    for season in sorted(
        data["Season"].dropna().unique()
    ):
        season_df = data[
            data["Season"].str.lower()
            == season.lower()
        ]

        documents.append(
            f"National {season}: average modal potato price "
            f"is Rs "
            f"{season_df['Modal_Price_Rs_per_Quintal'].mean():.2f} "
            f"per quintal."
        )

    return documents


@st.cache_resource
def build_rag_index(documents):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(documents)

    return vectorizer, matrix


rag_documents = build_rag_documents(df)

rag_vectorizer, rag_matrix = build_rag_index(
    rag_documents
)


def retrieve_documents(
    query,
    top_k=5,
):
    query_vector = rag_vectorizer.transform(
        [query]
    )

    scores = (
        rag_matrix @ query_vector.T
    ).toarray().ravel()

    indexes = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in indexes:
        if scores[index] > 0:
            results.append(
                (
                    rag_documents[index],
                    float(scores[index]),
                )
            )

    return results


def detect_state(text):
    lower_text = text.lower()

    states = sorted(
        df["State"]
        .dropna()
        .unique(),
        key=len,
        reverse=True,
    )

    for state in states:
        if state.lower() in lower_text:
            return state

    return None


def detect_season(text):
    lower_text = text.lower()

    seasons = sorted(
        df["Season"]
        .dropna()
        .unique(),
        key=len,
        reverse=True,
    )

    for season in seasons:
        if season.lower() in lower_text:
            return season

    return None


def rag_answer(
    question,
    state_context=None,
    season_context=None,
):
    """Simple dataset-grounded conversational RAG answer."""

    state = (
        detect_state(question)
        or state_context
    )

    season = (
        detect_season(question)
        or season_context
    )

    question_lower = question.lower()

    filtered = df.copy()

    if state:
        filtered = filtered[
            filtered["State"].str.lower()
            == state.lower()
        ]

    if season:
        filtered = filtered[
            filtered["Season"].str.lower()
            == season.lower()
        ]

    if filtered.empty:
        return (
            "I could not find matching records for "
            "that state/season combination. "
            "Try another state or season."
        ), state, season

    latest_date = filtered[
        "Arrival_Date"
    ].max()

    latest_rows = filtered[
        filtered["Arrival_Date"]
        == latest_date
    ]

    # Current/latest question
    if any(
        word in question_lower
        for word in [
            "current",
            "latest",
            "today",
            "now",
        ]
    ):
        value = latest_rows[
            "Modal_Price_Rs_per_Quintal"
        ].mean()

        location_text = (
            f" in {state}"
            if state
            else ""
        )

        season_text = (
            f" during {season}"
            if season
            else ""
        )

        answer = (
            f"The latest available modal potato "
            f"price in the dataset{location_text}"
            f"{season_text} is "
            f"₹{value:,.2f} per quintal "
            f"on {latest_date.date()}."
        )

    # Highest
    elif (
        "highest" in question_lower
        or "top" in question_lower
    ):
        grouped = (
            filtered.groupby("State")[
                "Modal_Price_Rs_per_Quintal"
            ]
            .mean()
            .sort_values(
                ascending=False
            )
            .head(5)
        )

        lines = [
            f"- {name}: ₹{value:,.2f}"
            for name, value in grouped.items()
        ]

        answer = (
            "Top states by average modal price:\n\n"
            + "\n".join(lines)
        )

    # Lowest
    elif "lowest" in question_lower:
        grouped = (
            filtered.groupby("State")[
                "Modal_Price_Rs_per_Quintal"
            ]
            .mean()
            .sort_values()
            .head(5)
        )

        lines = [
            f"- {name}: ₹{value:,.2f}"
            for name, value in grouped.items()
        ]

        answer = (
            "Lowest states by average modal price:\n\n"
            + "\n".join(lines)
        )

    # Trend
    elif any(
        word in question_lower
        for word in [
            "trend",
            "rising",
            "falling",
        ]
    ):
        weekly_local = (
            filtered.set_index("Arrival_Date")[
                "Modal_Price_Rs_per_Quintal"
            ]
            .resample("W-FRI")
            .mean()
            .dropna()
        )

        if len(weekly_local) >= 8:
            recent = weekly_local.tail(4).mean()
            previous = (
                weekly_local.iloc[-8:-4].mean()
            )

            change_pct = (
                (recent - previous)
                / previous
                * 100
                if previous != 0
                else 0
            )

            if recent > previous:
                direction = "rising"
            elif recent < previous:
                direction = "falling"
            else:
                direction = "stable"

            answer = (
                f"The recent price trend is "
                f"**{direction}**. "
                f"The latest 4-week average is "
                f"₹{recent:,.2f}, compared with "
                f"₹{previous:,.2f} in the previous "
                f"4-week period "
                f"({change_pct:+.2f}%)."
            )
        else:
            answer = (
                "There are not enough observations "
                "to calculate a short-term trend."
            )

    # Average / general question
    else:
        value = (
            filtered[
                "Modal_Price_Rs_per_Quintal"
            ].mean()
        )

        location_text = (
            f" in {state}"
            if state
            else ""
        )

        season_text = (
            f" during {season}"
            if season
            else ""
        )

        answer = (
            f"The average modal potato price"
            f"{location_text}{season_text} "
            f"is ₹{value:,.2f} per quintal."
        )

    # Add retrieved RAG evidence
    retrieved = retrieve_documents(
        question,
        top_k=3,
    )

    if retrieved:
        answer += (
            "\n\n**Retrieved dataset context:**\n"
            f"{retrieved[0][0]}"
        )

    return answer, state, season


# ============================================================
# 8. HEADER
# ============================================================
st.markdown(
    '<div class="main-title">🥔 Potato Price Intelligence</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Time-series ML • seasonal & weather features • "
    "hyperparameter evaluation • RAG chat"
    "</div>",
    unsafe_allow_html=True,
)

st.caption(
    f"Dataset source: `{dataset_source}`"
)

st.warning(
    "Academic project note: this dashboard demonstrates "
    "time-series feature engineering, regression, "
    "hyperparameter evaluation and retrieval-augmented "
    "question answering using the supplied dataset. "
    "Predictions should not be treated as live market advice."
)


# ============================================================
# 9. TABS
# ============================================================
tab_dashboard, tab_predict, tab_chat, tab_data = st.tabs(
    [
        "📊 Dashboard",
        "🔮 Prediction",
        "💬 RAG Chat",
        "📋 Data & Prompts",
    ]
)


# ============================================================
# 10. DASHBOARD
# ============================================================
with tab_dashboard:

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Records",
        f"{len(df):,}",
    )

    c2.metric(
        "States",
        df["State"].nunique(),
    )

    c3.metric(
        "Markets",
        df["Market"].nunique(),
    )

    c4.metric(
        "Average Modal Price",
        f"₹{df['Modal_Price_Rs_per_Quintal'].mean():,.0f}",
    )

    c5.metric(
        "Latest Date",
        str(df["Arrival_Date"].max().date()),
    )

    st.subheader("Model comparison")

    display_results = model_results.copy()

    st.dataframe(
        display_results.style.format(
            {
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}",
                "R2": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    fig_models = go.Figure()

    fig_models.add_trace(
        go.Bar(
            name="MAE",
            x=model_results["Model"],
            y=model_results["MAE"],
        )
    )

    fig_models.add_trace(
        go.Bar(
            name="RMSE",
            x=model_results["Model"],
            y=model_results["RMSE"],
        )
    )

    fig_models.update_layout(
        barmode="group",
        title="Model Error Comparison",
        yaxis_title="Error",
    )

    st.plotly_chart(
        fig_models,
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        season_stats = (
            df.groupby("Season")[
                "Modal_Price_Rs_per_Quintal"
            ]
            .mean()
            .reset_index()
        )

        fig = px.bar(
            season_stats,
            x="Season",
            y="Modal_Price_Rs_per_Quintal",
            title="Average Modal Price by Season",
            labels={
                "Modal_Price_Rs_per_Quintal":
                    "₹ / quintal"
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    with col2:
        trend = weekly[
            [
                "Arrival_Date",
                "price",
            ]
        ].copy()

        fig = px.line(
            trend,
            x="Arrival_Date",
            y="price",
            markers=True,
            title="Weekly National Modal Price Trend",
            labels={
                "Arrival_Date": "Date",
                "price": "₹ / quintal",
            },
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    state_stats = (
        df.groupby("State")[
            "Modal_Price_Rs_per_Quintal"
        ]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        state_stats,
        x="Modal_Price_Rs_per_Quintal",
        y="State",
        orientation="h",
        title="Top 10 States by Average Modal Price",
        labels={
            "Modal_Price_Rs_per_Quintal":
                "₹ / quintal"
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader(
        "Random Forest Hyperparameter Evaluation"
    )

    st.dataframe(
        hp_results.style.format(
            {
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}",
                "R2": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"Best displayed Random Forest configuration "
        f"by held-out R²: "
        f"n_estimators={int(best_row['n_estimators'])}, "
        f"max_depth={int(best_row['max_depth'])}. "
        f"R²={best_row['R2']:.4f}, "
        f"RMSE=₹{best_row['RMSE']:.2f}, "
        f"MAE=₹{best_row['MAE']:.2f}."
    )


# ============================================================
# 11. PREDICTION
# ============================================================
with tab_predict:

    st.subheader(
        "🔮 Next-Week Potato Price Prediction"
    )

    latest_week = weekly.iloc[-1]

    next_date = (
        latest_week["Arrival_Date"]
        + pd.Timedelta(days=7)
    )

    season_options = sorted(
        df["Season"].dropna().unique()
    )

    default_season = str(
        latest_week["Season"]
    )

    season = st.selectbox(
        "Expected season for next week",
        season_options,
        index=(
            season_options.index(
                default_season
            )
            if default_season
            in season_options
            else 0
        ),
    )

    rainfall = st.number_input(
        "Expected rainfall (mm)",
        min_value=0.0,
        value=float(
            latest_week["rainfall"]
        ),
    )

    temperature = st.number_input(
        "Expected temperature (°C)",
        value=float(
            latest_week["temperature"]
        ),
    )

    humidity = st.number_input(
        "Expected humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(
            latest_week["humidity"]
        ),
    )

    future_features = pd.DataFrame(
        [
            {
                "lag1": latest_week["price"],
                "lag2": weekly.iloc[-2]["price"],
                "rolling4": weekly.tail(4)["price"].mean(),
                "rainfall": rainfall,
                "temperature": temperature,
                "humidity": humidity,
            }
        ]
    )

    prediction = float(
        best_rf.predict(
            future_features[FEATURES]
        )[0]
    )

    st.metric(
        f"Predicted Modal Price for {next_date.date()}",
        f"₹{prediction:,.2f} / quintal",
    )

    st.caption(
        "Prediction uses lagged price, rolling mean and "
        "weather features with the displayed best "
        "Random Forest configuration."
    )

    prediction_chart = go.Figure()

    history = weekly.tail(12)

    prediction_chart.add_trace(
        go.Scatter(
            x=history["Arrival_Date"],
            y=history["price"],
            mode="lines+markers",
            name="Historical",
        )
    )

    prediction_chart.add_trace(
        go.Scatter(
            x=[next_date],
            y=[prediction],
            mode="markers",
            marker={"size": 13},
            name="Predicted",
        )
    )

    prediction_chart.update_layout(
        title="Recent Price History + Next-Week Prediction",
        xaxis_title="Date",
        yaxis_title="₹ / quintal",
    )

    st.plotly_chart(
        prediction_chart,
        use_container_width=True,
    )


# ============================================================
# 12. RAG CHAT
# ============================================================
with tab_chat:

    st.subheader(
        "💬 Potato Price RAG Chat"
    )

    st.write(
        "Ask questions about the prices, states, seasons "
        "or recent trends in the dataset."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! 👋 Ask me about potato prices. "
                    "For example: "
                    "'What is the latest potato price in Tamil Nadu?'"
                ),
            }
        ]

    if "state_context" not in st.session_state:
        st.session_state.state_context = None

    if "season_context" not in st.session_state:
        st.session_state.season_context = None

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

    prompt = st.chat_input(
        "Ask about potato prices..."
    )

    if prompt:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        answer, detected_state, detected_season = (
            rag_answer(
                prompt,
                st.session_state.state_context,
                st.session_state.season_context,
            )
        )

        st.session_state.state_context = (
            detected_state
        )

        st.session_state.season_context = (
            detected_season
        )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.rerun()

    st.caption(
        "RAG workflow: dataset → data-derived documents → "
        "TF-IDF retrieval → conversation context → answer."
    )


# ============================================================
# 13. DATA + PROMPTS
# ============================================================
with tab_data:

    st.subheader("Dataset Information")

    st.write(
        f"Rows: **{len(df):,}**"
    )

    st.write(
        f"Columns: **{len(df.columns)}**"
    )

    st.write(
        f"Date range: **{df['Arrival_Date'].min().date()} "
        f"to {df['Arrival_Date'].max().date()}**"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader(
        "Natural-Language Prompts Used"
    )

    prompts = [
        "What is the current potato price in Tamil Nadu?",
        "What is the average potato price in Punjab during monsoon?",
        "How much does potato cost during winter?",
        "Which states have the highest potato prices?",
        "Is the potato price trend rising or falling?",
        "What is the price in Uttar Pradesh in post-monsoon season?",
    ]

    for number, prompt_text in enumerate(
        prompts,
        start=1,
    ):
        st.markdown(
            f"**{number}.** {prompt_text}"
        )

    st.subheader(
        "Complete Project Workflow"
    )

    st.code(
        """
Agricultural Potato Dataset
        ↓
Data Validation & Preprocessing
        ↓
Weekly Time-Series Aggregation
        ↓
Lag 1 + Lag 2 + 4-Week Rolling Mean
        ↓
Weather Features
(Rainfall + Temperature + Humidity)
        ↓
Regression Models
(Linear + Ridge + Random Forest)
        ↓
MAE + RMSE + R²
        ↓
Random Forest Hyperparameter Evaluation
        ↓
Price Prediction
        ↓
TF-IDF RAG Retrieval
        ↓
Conversation Context
        ↓
Streamlit Dashboard + Chatbot
        """,
        language="text",
    )

    st.subheader(
        "Model Features"
    )

    st.write(
        FEATURES
    )

    st.subheader(
        "Dataset Source Used by App"
    )

    st.code(
        dataset_source
    )
