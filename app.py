import streamlit as st
import pandas as pd
import altair as alt
from datetime import timedelta
from predict_utils import predict_all, explain_prediction

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="centered")

CATEGORY_COLORS = {
    "Good": "#00e400",
    "Moderate": "#f4d03f",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}
CATEGORY_ICONS = {
    "Good": "🟢",
    "Moderate": "🟡",
    "Unhealthy for Sensitive Groups": "🟠",
    "Unhealthy": "🔴",
    "Very Unhealthy": "🟣",
    "Hazardous": "🟤",
}

st.title("🌫️ Karachi AQI Predictor")
st.caption("3-day average Air Quality Index forecast for Karachi, Pakistan · Powered by machine learning")

if st.button("🔄 Refresh prediction"):
    st.cache_data.clear()


@st.cache_data(ttl=1800)
def get_prediction():
    return predict_all()


with st.spinner("Fetching live data and running forecast..."):
    try:
        result = get_prediction()
    except Exception as e:
        st.error("Unable to fetch the latest AQI prediction. Please try refreshing in a moment.")
        st.exception(e)
        st.stop()

pkt_time = result['now_time'] + timedelta(hours=5)
st.caption(
    f"Last updated: {result['now_time'].strftime('%d %b %Y, %I:%M %p')} UTC "
    f"({pkt_time.strftime('%I:%M %p')} PKT)"
)

# ---- Current AQI ----
# ---- Current AQI ----
st.subheader("Current Air Quality")
current_color = CATEGORY_COLORS[result["current_category"]]
current_icon = CATEGORY_ICONS[result["current_category"]]
st.markdown(
    f"""
    <div style="background-color:{current_color}22; border:1px solid {current_color};
                border-radius:10px; padding:20px; text-align:center; margin-bottom:10px;">
        <div style="font-size:2.5em; font-weight:700;">{current_icon} {result['current_aqi']}</div>
        <div style="font-size:1.1em;">{result['current_category']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Hazardous AQI alert ----
all_aqi_values = [result["current_aqi"]] + [
    result["predictions"][t]["aqi"] for t in ["target_day1", "target_day2", "target_day3"]
]
if max(all_aqi_values) > 150:
    st.error(
        "⚠️ **Health Alert**: Average AQI is expected to reach Unhealthy levels or worse "
        "in the next 3 days. Sensitive groups should limit outdoor exposure."
    )
elif max(all_aqi_values) > 100:
    st.warning(
        "⚠️ Average air quality may reach levels affecting sensitive groups "
        "(children, elderly, those with respiratory conditions) in the next 3 days."
    )

st.divider()

# ---- 3-day forecast cards ----
st.subheader("3-Day Average AQI Forecast")

today = result["now_time"].date()
day_dates = [today + timedelta(days=i) for i in [1, 2, 3]]
day_titles = ["Tomorrow", day_dates[1].strftime("%A"), day_dates[2].strftime("%A")]
target_keys = ["target_day1", "target_day2", "target_day3"]

cols = st.columns(3)
for col, title, date, key in zip(cols, day_titles, day_dates, target_keys):
    pred = result["predictions"][key]
    color = CATEGORY_COLORS[pred["category"]]
    icon = CATEGORY_ICONS[pred["category"]]
    with col:
        st.markdown(
            f"""
            <div style="background-color:{color}22; border:1px solid {color};
                        border-radius:10px; padding:16px; text-align:center;">
                <div style="font-weight:600; font-size:0.95em;">{title}</div>
                <div style="font-size:0.8em; color:gray; margin-bottom:8px;">{date.strftime('%b %d, %Y')}</div>
                <div style="font-size:1.8em; font-weight:700;">{icon} {pred['aqi']}</div>
                <div style="font-size:0.85em; margin-bottom:6px;">{pred['category']}</div>
                <div style="font-size:0.75em; color:gray;">Expected: {pred['range_low']}–{pred['range_high']}</div>
                <div style="font-size:0.75em; color:gray;">± RMSE {pred['rmse']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
st.markdown("#### Why these predictions?")
st.caption("Based on the linear (Ridge) component of the ensemble model")
explain_tabs = st.tabs(day_titles)
for tab, key in zip(explain_tabs, target_keys):
    with tab:
        with st.spinner("Computing explanation..."):
            explanation = explain_prediction(key, result["feature_row"])
        col_up, col_down = st.columns(2)
        with col_up:
            st.markdown("**⬆️ Increases predicted AQI**")
            for name, val in explanation["increase"]:
                st.markdown(f"- {name} (+{val})")
        with col_down:
            st.markdown("**⬇️ Decreases predicted AQI**")
            for name, val in explanation["decrease"]:
                st.markdown(f"- {name} ({val})")
st.divider()

# ---- Trend chart (Altair, correct chronological order, tighter y-axis) ----
st.subheader("Forecast Trend")

period_labels = ["Today"] + [d.strftime("%a %d") for d in day_dates]
chart_df = pd.DataFrame({
    "Period": pd.Categorical(period_labels, categories=period_labels, ordered=True),
    "Average AQI": all_aqi_values,
})

y_min = max(0, min(all_aqi_values) - 20)
y_max = max(all_aqi_values) + 20

chart = (
    alt.Chart(chart_df)
    .mark_line(point=True, strokeWidth=3, color="#f4a300")
    .encode(
        x=alt.X("Period", sort=period_labels, title=None),
        y=alt.Y("Average AQI", scale=alt.Scale(domain=[y_min, y_max]), title="AQI"),
        tooltip=["Period", "Average AQI"],
    )
    .properties(height=280)
)
st.altair_chart(chart, use_container_width=True)

st.divider()

# ---- AQI scale reference ----
with st.expander("AQI Scale Reference"):
    st.markdown("""
    | AQI | Category | Meaning |
    |---|---|---|
    | 0–50 | 🟢 Good | Air quality is satisfactory |
    | 51–100 | 🟡 Moderate | Acceptable; sensitive groups should watch heavy exertion |
    | 101–150 | 🟠 Unhealthy for Sensitive Groups | Sensitive groups may be affected |
    | 151–200 | 🔴 Unhealthy | Everyone may begin experiencing effects |
    | 201–300 | 🟣 Very Unhealthy | Health alert — everyone may be seriously affected |
    | 301+ | 🟤 Hazardous | Emergency conditions |
    """)

# ---- About section ----
with st.expander("ℹ️ About this project"):
    st.markdown("""
    - **Historical data**: Open-Meteo (Air Quality + Weather Archive APIs), Jan 2023 – present
    - **Live data**: Open-Meteo (current + forecast APIs)
    - **Model**: Ensemble (Ridge + SVR + CatBoost average) — selected after comparing 6+ model families
    - **Features**: 92 engineered features (lags, rolling mean/std/min/max, cyclical calendar encoding, current + forecasted weather, current pollutants)
    - **Target**: Calendar-day average AQI (Day 1 / Day 2 / Day 3)
    - **Feature Store**: Hopsworks (features and targets)
    - **Model Registry**: Hopsworks (versioned models)
    - **Dashboard**: Streamlit
    """)

st.divider()
st.caption("Model: Ridge + SVR + CatBoost ensemble · Trained on 3.5 years of Karachi historical data · Features and targets stored in Hopsworks Feature Store · Models versioned in Hopsworks Model Registry")