import streamlit as st
import pandas as pd
import altair as alt
from datetime import timedelta
from predict_utils import predict_all, explain_prediction

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: #f5f6f8; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 880px; }

h3, h4 { font-weight: 700 !important; color: #0f172a; }
[data-testid="stCaptionContainer"] { color: #64748b; }

.stButton > button {
    border-radius: 8px; border: 1px solid #e2e8f0; background: white;
    color: #334155; font-weight: 600; padding: 0.4rem 1rem;
    transition: all 0.15s ease;
}
.stButton > button:hover { border-color: #3b82f6; color: #3b82f6; }

.streamlit-expanderHeader { font-weight: 600; border-radius: 10px; background: white; }
[data-testid="stExpander"] { border: 1px solid #e2e8f0; border-radius: 12px; background: white; }

hr { margin: 1.6em 0; border-color: #e2e8f0; }

.stTabs [data-baseweb="tab-list"] { gap: 4px; background: #eef1f5; padding: 4px; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 600; color: #64748b; }
.stTabs [aria-selected="true"] { background: white !important; color: #0f172a !important; }

.pulse-dot {
    display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#22c55e; margin-right:6px; box-shadow:0 0 0 rgba(34,197,94,0.5);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(34,197,94,0.5); }
    70% { box-shadow: 0 0 0 8px rgba(34,197,94,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
}

.header-badge {
    display:inline-flex; align-items:center; justify-content:center;
    width:44px; height:44px; border-radius:12px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    font-size:1.4em; margin-right:12px;
}
</style>
""", unsafe_allow_html=True)

CATEGORY_COLORS = {
    "Good": "#22c55e", "Moderate": "#eab308",
    "Unhealthy for Sensitive Groups": "#f97316", "Unhealthy": "#ef4444",
    "Very Unhealthy": "#a855f7", "Hazardous": "#881337",
}
CATEGORY_ICONS = {
    "Good": "🟢", "Moderate": "🟡", "Unhealthy for Sensitive Groups": "🟠",
    "Unhealthy": "🔴", "Very Unhealthy": "🟣", "Hazardous": "🟤",
}

# ---- Header ----
st.markdown(
    """
    <div style="display:flex; align-items:center; margin-bottom:2px;">
        <div class="header-badge">🌫️</div>
        <div>
            <div style="font-size:1.5em; font-weight:800; color:#0f172a; line-height:1.1;">Karachi AQI Predictor</div>
            <div style="font-size:0.85em; color:#64748b;">3-day Air Quality forecast · Machine learning powered</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

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
st.markdown(
    f"""<span class="pulse-dot"></span><span style="color:#64748b; font-size:0.85em;">
    Last updated {result['now_time'].strftime('%d %b, %I:%M %p')} UTC
    ({pkt_time.strftime('%I:%M %p')} PKT)</span>""",
    unsafe_allow_html=True,
)
st.write("")

# ---- Current AQI: circular gauge ----
current_color = CATEGORY_COLORS[result["current_category"]]
current_aqi = result["current_aqi"]
gauge_pct = min(current_aqi / 300, 1.0)  # scale gauge to 0-300 AQI range
circumference = 326  # for r=52 arc (approx 3/4 circle)
offset = circumference * (1 - gauge_pct)

st.markdown(
    f"""
    <div style="background:white; border:1px solid #e2e8f0; border-radius:16px; padding:24px 28px;
                margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.04);
                display:flex; align-items:center; gap:24px;">
        <svg width="100" height="100" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="52" fill="none" stroke="#eef1f5" stroke-width="10"/>
            <circle cx="60" cy="60" r="52" fill="none" stroke="{current_color}" stroke-width="10"
                    stroke-linecap="round" stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}" transform="rotate(-90 60 60)"/>
            <text x="60" y="56" text-anchor="middle" font-size="26" font-weight="800" fill="#0f172a" font-family="Inter">{current_aqi:.0f}</text>
            <text x="60" y="76" text-anchor="middle" font-size="11" fill="#94a3b8" font-family="Inter">AQI</text>
        </svg>
        <div>
            <div style="font-size:0.75em; color:#94a3b8; font-weight:700; letter-spacing:0.5px; text-transform:uppercase;">Current Air Quality</div>
            <div style="font-size:1.3em; font-weight:700; color:{current_color}; margin-top:2px;">{result['current_category']}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Hazard alert ----
all_aqi_values = [result["current_aqi"]] + [
    result["predictions"][t]["aqi"] for t in ["target_day1", "target_day2", "target_day3"]
]
if max(all_aqi_values) > 150:
    st.error("⚠️ **Health Alert**: Average AQI is expected to reach Unhealthy levels or worse in the next 3 days. Sensitive groups should limit outdoor exposure.")
elif max(all_aqi_values) > 100:
    st.warning("⚠️ Average air quality may reach levels affecting sensitive groups (children, elderly, those with respiratory conditions) in the next 3 days.")

st.divider()

# ---- Forecast cards ----
st.markdown("#### 3-Day Average AQI Forecast")

today = result["now_time"].date()
day_dates = [today + timedelta(days=i) for i in [1, 2, 3]]
day_titles = ["Tomorrow", day_dates[1].strftime("%A"), day_dates[2].strftime("%A")]
target_keys = ["target_day1", "target_day2", "target_day3"]

cols = st.columns(3)
for i, (col, title, date, key) in enumerate(zip(cols, day_titles, day_dates, target_keys)):
    pred = result["predictions"][key]
    color = CATEGORY_COLORS[pred["category"]]
    icon = CATEGORY_ICONS[pred["category"]]
    bar_pct = min(max((pred["aqi"] - pred["range_low"]) / max(pred["range_high"] - pred["range_low"], 1), 0), 1) * 100
    with col:
        st.markdown(
            f"""
            <div style="background:white; border:1px solid #e2e8f0; border-radius:14px;
                        padding:16px 12px; text-align:center; height:100%;
                        box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                <div style="font-size:0.7em; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.4px;">Day {i+1}</div>
                <div style="font-weight:700; font-size:0.95em; color:#0f172a; margin-top:2px;">{title}</div>
                <div style="font-size:0.68em; color:#94a3b8; margin-bottom:10px;">{date.strftime('%b %d')}</div>
                <div style="font-size:1.9em; margin-bottom:0px;">{icon}</div>
                <div style="font-size:1.9em; font-weight:800; color:#0f172a;">{pred['aqi']}</div>
                <div style="font-size:0.8em; font-weight:600; color:{color}; margin:2px 0 10px;">{pred['category']}</div>
                <div style="height:5px; background:#eef1f5; border-radius:3px; overflow:hidden; margin-bottom:6px;">
                    <div style="height:100%; width:{bar_pct}%; background:{color}; border-radius:3px;"></div>
                </div>
                <div style="font-size:0.65em; color:#94a3b8;">
                    {pred['range_low']}–{pred['range_high']} · ±{pred['rmse']} RMSE
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.write("")
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

# ---- Trend chart ----
st.markdown("#### Forecast Trend")
period_labels = ["Today"] + [d.strftime("%a %d") for d in day_dates]
chart_df = pd.DataFrame({
    "Period": pd.Categorical(period_labels, categories=period_labels, ordered=True),
    "Average AQI": all_aqi_values,
})
y_min = max(0, min(all_aqi_values) - 20)
y_max = max(all_aqi_values) + 20

chart = (
    alt.Chart(chart_df)
    .mark_line(point=alt.OverlayMarkDef(size=80, filled=True), strokeWidth=3, color="#3b82f6")
    .encode(
        x=alt.X("Period", sort=period_labels, title=None),
        y=alt.Y("Average AQI", scale=alt.Scale(domain=[y_min, y_max]), title="AQI"),
        tooltip=["Period", "Average AQI"],
    )
    .properties(height=260)
    .configure_view(strokeWidth=0)
    .configure_axis(gridColor="#eef1f5", domainColor="#e2e8f0")
)
st.altair_chart(chart, use_container_width=True)

st.divider()

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

with st.expander("ℹ️ About this project"):
    st.markdown("""
    - **Historical data**: Open-Meteo (Air Quality + Weather Archive APIs), Jan 2023 – present
    - **Live data**: Open-Meteo (current + forecast APIs)
    - **Model**: Ensemble (Ridge + SVR + CatBoost average) — selected after comparing 7 model families
    - **Features**: 92 engineered features (lags, rolling mean/std/min/max, cyclical calendar encoding, current + forecasted weather, current pollutants)
    - **Target**: Calendar-day average AQI (Day 1 / Day 2 / Day 3)
    - **Feature Store**: Hopsworks (features and targets)
    - **Model Registry**: Hopsworks (versioned models)
    - **Dashboard**: Streamlit
    """)

st.divider()
st.caption("Model: Ridge + SVR + CatBoost ensemble · Trained on 3.5 years of Karachi historical data · Features and targets stored in Hopsworks Feature Store · Models versioned in Hopsworks Model Registry")