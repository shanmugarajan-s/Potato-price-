
import re
from pathlib import Path

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

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Potato Price Intelligence",
    page_icon="🥔",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0;}
    .subtitle {color: #667; margin-top: 0.2rem; margin-bottom: 1.2rem;}
    .metric-card {padding: 0.3rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "Potato_Price_TimeSeries_Weather_Dataset.csv"

if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "Potato_Price_TimeSeries_Weather_Dataset.csv"

# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")
    df = df.dropna(subset=["Arrival_Date", "Modal_Price_Rs_per_Quintal"]).copy()
    df = df.sort_values("Arrival_Date")

    numeric_cols = [
        "Rainfall_mm",
        "Temperature_C",
        "Humidity_percent",
        "Min_Price_Rs_per_Quintal",
        "Max_Price_Rs_per_Quintal",
        "Modal_Price_Rs_per_Quintal",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.dropna(subset=["Modal_Price_Rs_per_Quintal"])


df = load_data()

# ------------------------------------------------------------
# National weekly time series + model features
# ------------------------------------------------------------
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

    # Derive season from the original observations for each weekly date.
    season_lookup = (
        data.groupby(data["Arrival_Date"].dt.to_period("W-FRI"))["Season"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        .reset_index()
    )
    season_lookup["Arrival_Date"] = season_lookup["Arrival_Date"].dt.end_time.dt.normalize()
    weekly = weekly.merge(season_lookup, on="Arrival_Date", how="left")

    return weekly.dropna(subset=["lag1", "lag2", "rolling4"]).reset_index(drop=True)


weekly = build_weekly_data(df)

FEATURES = ["lag1", "lag2", "rolling4", "rainfall", "temperature", "humidity"]

# ------------------------------------------------------------
# Model evaluation
# ------------------------------------------------------------
@st.cache_data
def evaluate_models(weekly_df):
    n_test = min(8, max(2, len(weekly_df) // 6))
    train = weekly_df.iloc[:-n_test].copy()
    test = weekly_df.iloc[-n_test:].copy()

    X_train, y_train = train[FEATURES], train["price"]
    X_test, y_test = test[FEATURES], test["price"]

    models = {
        "Linear Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0))
        ]),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": mean_squared_error(y_test, pred, squared=False),
            "R2": r2_score(y_test, pred),
        })
        fitted[name] = model

    results = pd.DataFrame(rows)
    return results, fitted, train, test


model_results, fitted_models, train_df, test_df = evaluate_models(weekly)

# ------------------------------------------------------------
# Random Forest hyperparameter sweep
# ------------------------------------------------------------
@st.cache_data
def rf_sweep(weekly_df):
    n_test = min(8, max(2, len(weekly_df) // 6))
    train = weekly_df.iloc[:-n_test]
    test = weekly_df.iloc[-n_test:]

    X_train, y_train = train[FEATURES], train["price"]
    X_test, y_test = test[FEATURES], test["price"]

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
            pred = model.predict(X_test)
            rows.append({
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": mean_squared_error(y_test, pred, squared=False),
                "R2": r2_score(y_test, pred),
            })
    return pd.DataFrame(rows)


hp_results = rf_sweep(weekly)

# Select best configuration by highest R² for display.
best_row = hp_results.sort_values(
    ["R2", "RMSE", "MAE"], ascending=[False, True, True]
).iloc[0]

@st.cache_resource
def train_best_rf(params, weekly_df):
    n_test = min(8, max(2, len(weekly_df) // 6))
    train = weekly_df.iloc[:-n_test]
    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train["price"])
    return model


best_rf = train_best_rf(
    {"n_estimators": best_row["n_estimators"], "max_depth": best_row["max_depth"]},
    weekly,
)

# ------------------------------------------------------------
# Lightweight RAG
# ------------------------------------------------------------
def build_rag_documents(data):
    docs = []

    latest_date = data["Arrival_Date"].max()
    latest = data[data["Arrival_Date"] == latest_date]
    docs.append(
        f"Current dataset date is {latest_date.date()}. "
        f"National average modal potato price on that date is "
        f"Rs {latest['Modal_Price_Rs_per_Quintal'].mean():.2f} per quintal."
    )

    for state in sorted(data["State"].dropna().unique()):
        s = data[data["State"] == state]
        latest_state_date = s["Arrival_Date"].max()
        latest_state = s[s["Arrival_Date"] == latest_state_date]
        docs.append(
            f"State {state}: latest available date is {latest_state_date.date()}, "
            f"latest modal price average is Rs "
            f"{latest_state['Modal_Price_Rs_per_Quintal'].mean():.2f} per quintal, "
            f"overall average is Rs {s['Modal_Price_Rs_per_Quintal'].mean():.2f}."
        )

        for season in sorted(s["Season"].dropna().unique()):
            ss = s[s["Season"] == season]
            docs.append(
                f"State {state} during {season}: average modal potato price is "
                f"Rs {ss['Modal_Price_Rs_per_Quintal'].mean():.2f} per quintal."
            )

    for season in sorted(data["Season"].dropna().unique()):
        ss = data[data["Season"] == season]
        docs.append(
            f"National {season}: average modal potato price is "
            f"Rs {ss['Modal_Price_Rs_per_Quintal'].mean():.2f} per quintal."
        )

    return docs


@st.cache_resource
def build_retriever(documents):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents)
    return vectorizer, matrix


rag_documents = build_rag_documents(df)
rag_vectorizer, rag_matrix = build_retriever(rag_documents)


def retrieve_documents(query, top_k=5):
    q = rag_vectorizer.transform([query])
    scores = (rag_matrix @ q.T).toarray().ravel()
    idx = np.argsort(scores)[::-1][:top_k]
    return [(rag_documents[i], float(scores[i])) for i in idx if scores[i] > 0]


def detect_state(text):
    lower = text.lower()
    for state in sorted(df["State"].dropna().unique(), key=len, reverse=True):
        if state.lower() in lower:
            return state
    return None


def detect_season(text):
    lower = text.lower()
    for season in sorted(df["Season"].dropna().unique(), key=len, reverse=True):
        if season.lower() in lower:
            return season
    return None


def rag_answer(question, state_context=None, season_context=None):
    state = detect_state(question) or state_context
    season = detect_season(question) or season_context

    q_lower = question.lower()
    filtered = df.copy()

    if state:
        filtered = filtered[filtered["State"].str.lower() == state.lower()]
    if season:
        filtered = filtered[filtered["Season"].str.lower() == season.lower()]

    if filtered.empty:
        return (
            "I could not find matching records for that state/season combination. "
            "Try another state or season."
        ), state, season

    latest_date = filtered["Arrival_Date"].max()
    latest_rows = filtered[filtered["Arrival_Date"] == latest_date]

    if any(k in q_lower for k in ["current", "latest", "today", "now"]):
        value = latest_rows["Modal_Price_Rs_per_Quintal"].mean()
        answer = (
            f"The latest available {('for ' + state) if state else 'national'} "
            f"modal potato price in the dataset"
            f"{(' during ' + season) if season else ''} is "
            f"₹{value:,.2f} per quintal on {latest_date.date()}."
        )
    elif "highest" in q_lower or "top" in q_lower:
        grouped = (
            filtered.groupby("State")["Modal_Price_Rs_per_Quintal"]
            .mean().sort_values(ascending=False).head(5)
        )
        lines = [f"{k}: ₹{v:,.2f}" for k, v in grouped.items()]
        answer = "Top states by average modal price:\n\n" + "\n".join(lines)
    elif "lowest" in q_lower:
        grouped = (
            filtered.groupby("State")["Modal_Price_Rs_per_Quintal"]
            .mean().sort_values().head(5)
        )
        lines = [f"{k}: ₹{v:,.2f}" for k, v in grouped.items()]
        answer = "Lowest states by average modal price:\n\n" + "\n".join(lines)
    elif "trend" in q_lower or "rising" in q_lower or "falling" in q_lower:
        weekly_local = (
            filtered.set_index("Arrival_Date")["Modal_Price_Rs_per_Quintal"]
            .resample("W-FRI").mean().dropna()
        )
        if len(weekly_local) >= 4:
            recent = weekly_local.tail(4).mean()
            previous = weekly_local.iloc[-8:-4].mean() if len(weekly_local) >= 8 else weekly_local.head(4).mean()
            direction = "rising" if recent > previous else "falling"
            pct = ((recent - previous) / previous * 100) if previous else 0
            answer = f"The recent price trend is **{direction}**, about {pct:+.2f}% comparing the latest 4-week average with the previous 4-week period."
        else:
            answer = "There are not enough observations to calculate a reliable short-term trend."
    else:
        value = filtered["Modal_Price_Rs_per_Quintal"].mean()
        label = "average modal potato price"
        if season:
            label += f" during {season}"
        answer = f"The {label}{(' in ' + state) if state else ''} is ₹{value:,.2f} per quintal."

    retrieved = retrieve_documents(question, top_k=3)
    if retrieved:
        answer += "\n\n**RAG context retrieved:** " + retrieved[0][0]

    return answer, state, season


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.markdown('<div class="main-title">🥔 Potato Price Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Time-series ML • seasonal & weather features • hyperparameter evaluation • lightweight RAG chat</div>',
    unsafe_allow_html=True,
)

st.warning(
    "Academic dataset note: the supplied 52-week history and weather extension are documented "
    "simulation anchored to a real market-price snapshot. The reported metrics demonstrate the "
    "methodology on this academic dataset, not production forecasting accuracy."
)

tab_dashboard, tab_predict, tab_chat, tab_data = st.tabs(
    ["📊 Dashboard", "🔮 Prediction", "💬 RAG Chat", "📋 Data & Prompts"]
)

with tab_dashboard:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{len(df):,}")
    c2.metric("States", df["State"].nunique())
    c3.metric("Markets", df["Market"].nunique())
    c4.metric("Avg Modal Price", f"₹{df['Modal_Price_Rs_per_Quintal'].mean():,.0f}")
    c5.metric("Latest Date", str(df["Arrival_Date"].max().date()))

    st.subheader("Model comparison")
    st.dataframe(
        model_results.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    fig_models = go.Figure()
    fig_models.add_trace(go.Bar(name="MAE", x=model_results["Model"], y=model_results["MAE"]))
    fig_models.add_trace(go.Bar(name="RMSE", x=model_results["Model"], y=model_results["RMSE"]))
    fig_models.update_layout(barmode="group", title="MAE vs RMSE")
    st.plotly_chart(fig_models, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        season_stats = (
            df.groupby("Season")["Modal_Price_Rs_per_Quintal"]
            .mean().reset_index()
        )
        fig = px.bar(
            season_stats, x="Season", y="Modal_Price_Rs_per_Quintal",
            title="Average modal price by season",
            labels={"Modal_Price_Rs_per_Quintal": "₹ / quintal"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        trend = weekly[["Arrival_Date", "price"]].copy()
        fig = px.line(
            trend, x="Arrival_Date", y="price",
            markers=True,
            title="Weekly national modal price trend",
            labels={"Arrival_Date": "Date", "price": "₹ / quintal"},
        )
        st.plotly_chart(fig, use_container_width=True)

    state_stats = (
        df.groupby("State")["Modal_Price_Rs_per_Quintal"]
        .mean().sort_values(ascending=False).head(10).reset_index()
    )
    fig = px.bar(
        state_stats, x="Modal_Price_Rs_per_Quintal", y="State",
        orientation="h",
        title="Top 10 states by average modal price",
        labels={"Modal_Price_Rs_per_Quintal": "₹ / quintal"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Random Forest hyperparameter sweep")
    st.dataframe(
        hp_results.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"Displayed best RF configuration by held-out R²: "
        f"n_estimators={int(best_row['n_estimators'])}, "
        f"max_depth={int(best_row['max_depth'])}. "
        f"Test R²={best_row['R2']:.4f}, RMSE=₹{best_row['RMSE']:.2f}, "
        f"MAE=₹{best_row['MAE']:.2f}."
    )

with tab_predict:
    st.subheader("Next-week national potato price prediction")

    latest = weekly.iloc[-1]
    next_date = latest["Arrival_Date"] + pd.Timedelta(days=7)

    season_options = sorted(df["Season"].dropna().unique())
    default_season = str(latest["Season"]) if pd.notna(latest["Season"]) else season_options[0]
    season = st.selectbox(
        "Expected season for next week",
        season_options,
        index=season_options.index(default_season) if default_season in season_options else 0,
    )

    rainfall = st.number_input("Expected rainfall (mm)", value=float(latest["rainfall"]), min_value=0.0)
    temperature = st.number_input("Expected temperature (°C)", value=float(latest["temperature"]))
    humidity = st.number_input("Expected humidity (%)", value=float(latest["humidity"]), min_value=0.0, max_value=100.0)

    X_future = pd.DataFrame([{
        "lag1": latest["price"],
        "lag2": weekly.iloc[-2]["price"],
        "rolling4": weekly.tail(4)["price"].mean(),
        "rainfall": rainfall,
        "temperature": temperature,
        "humidity": humidity,
    }])

    prediction = float(best_rf.predict(X_future[FEATURES])[0])

    st.metric(f"Predicted modal price for {next_date.date()}", f"₹{prediction:,.2f} / quintal")
    st.caption(
        "The prediction uses the best displayed Random Forest configuration and the latest "
        "price-history/weather features. It is an academic demonstration, not a production forecast."
    )

    pred_fig = go.Figure()
    hist = weekly.tail(12)
    pred_fig.add_trace(go.Scatter(
        x=hist["Arrival_Date"], y=hist["price"],
        mode="lines+markers", name="Historical"
    ))
    pred_fig.add_trace(go.Scatter(
        x=[next_date], y=[prediction],
        mode="markers", marker=dict(size=12),
        name="Predicted"
    ))
    pred_fig.update_layout(
        title="Recent history + next-week prediction",
        xaxis_title="Date", yaxis_title="₹ / quintal"
    )
    st.plotly_chart(pred_fig, use_container_width=True)

with tab_chat:
    st.subheader("💬 Ask about potato prices")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! Ask me about potato prices, states, seasons or trends. "
                    "Example: 'What is the current potato price in Tamil Nadu?'"
                ),
            }
        ]
    if "state_context" not in st.session_state:
        st.session_state.state_context = None
    if "season_context" not in st.session_state:
        st.session_state.season_context = None

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("Ask about potato prices...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        answer, state_ctx, season_ctx = rag_answer(
            prompt,
            st.session_state.state_context,
            st.session_state.season_context,
        )
        st.session_state.state_context = state_ctx
        st.session_state.season_context = season_ctx
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()

    st.caption(
        "RAG pipeline: dataset → data-derived documents → TF-IDF retrieval → "
        "conversation context → natural-language response. No external LLM API is required."
    )

with tab_data:
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    st.subheader("Assignment prompts used")
    prompts = [
        "What is the current potato price in Tamil Nadu?",
        "What is the average potato price in Punjab during monsoon?",
        "How much does potato cost during winter?",
        "Which states have the highest potato prices?",
        "Is the potato price trend rising or falling?",
        "What is the price in Uttar Pradesh in post-monsoon season?",
    ]
    for i, p in enumerate(prompts, 1):
        st.markdown(f"**{i}.** {p}")

    st.subheader("Project workflow")
    st.code(
        """Prepared dataset
        ↓
Data validation + preprocessing
        ↓
Weekly time-series features
        ↓
Lag1 + Lag2 + rolling mean + weather
        ↓
Linear / Ridge / Random Forest
        ↓
MAE + RMSE + R²
        ↓
Random Forest hyperparameter sweep
        ↓
TF-IDF RAG retrieval + conversation context
        ↓
Streamlit dashboard + chat""",
        language="text",
    )
