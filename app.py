import json
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

from weather_api import (
    search_city,
    get_weather_data,
    prepare_prediction_features
)
from utils.advisory import (
    get_apparent_temperature,
    calculate_storm_threat_score,
    get_fog_risk
)
from utils.tracker import (
    log_prediction,
    get_logs_dataframe,
    get_tracker_metrics,
    update_ground_truth,
    verify_pending_records
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CloudCue | Atmospheric ML Forecast",
    page_icon="🌤️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "datasets"

# ============================================================
# DARK THEME STYLING
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426, #102B46);
    color: #EAF4FF;
}

header[data-testid="stHeader"] {
    background-color: #071426 !important;
}

[data-testid="stToolbar"] {
    background-color: transparent !important;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

section[data-testid="stSidebar"] {
    background-color: #091A2D;
    border-right: 1px solid #244563;
}

section[data-testid="stSidebar"] * {
    color: #EAF4FF;
}

h1, h2, h3, h4 {
    color: #FFFFFF !important;
}

p, label, .stMarkdown {
    color: #D5E6F7;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #65C7FF;
    margin-bottom: 0;
}

.tagline {
    font-size: 16px;
    color: #A5DFFF;
    margin-top: 0;
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #163653, #102A43);
    padding: 16px;
    border-radius: 14px;
    border: 1px solid #28618A;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.18);
}

div[data-testid="stMetric"] label {
    color: #A9D9F5 !important;
}

div[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

.stTextInput input {
    background-color: #163653;
    color: white;
    border: 1px solid #347BA8;
}

.stButton button {
    background: linear-gradient(90deg, #087ECA, #20B7D8);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}

.stButton button:hover {
    background: linear-gradient(90deg, #20B7D8, #087ECA);
    color: white;
    border: none;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #28618A;
    border-radius: 10px;
}

hr {
    border-color: #31516E;
}

a {
    color: #65C7FF !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(16, 42, 67, 0.65);
    border: 1px solid #28618A;
    border-radius: 16px;
    padding: 16px;
}

.badge-low {
    background-color: #1b4d3e;
    color: #70f3c2;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
}

.badge-med {
    background-color: #1d456a;
    color: #7bd4ff;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
}

.badge-high {
    background-color: #633c07;
    color: #ffc061;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
}

.badge-severe {
    background-color: #611818;
    color: #ff8e8e;
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
}

/* Fix Alert box styling for clear readability */
div[data-testid="stAlert"] {
    background-color: #132d46 !important;
    border: 1px solid #28618a !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
}

div[data-testid="stAlert"] * {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TRAINED MODELS & METADATA
# ============================================================

@st.cache_resource
def load_models():
    temp_model = joblib.load(MODEL_DIR / "best_temperature_model.pkl")
    hum_model = joblib.load(MODEL_DIR / "best_humidity_model.pkl")
    rain_model = joblib.load(MODEL_DIR / "best_rainfall_model.pkl")

    t3_p = MODEL_DIR / "best_temperature_3h_model.pkl"
    t6_p = MODEL_DIR / "best_temperature_6h_model.pkl"
    r3_p = MODEL_DIR / "best_rainfall_3h_model.pkl"
    r6_p = MODEL_DIR / "best_rainfall_6h_model.pkl"

    horizons = {
        "t3": joblib.load(t3_p) if t3_p.exists() else None,
        "t6": joblib.load(t6_p) if t6_p.exists() else None,
        "r3": joblib.load(r3_p) if r3_p.exists() else None,
        "r6": joblib.load(r6_p) if r6_p.exists() else None,
    }

    metadata = {}
    meta_path = MODEL_DIR / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return temp_model, hum_model, rain_model, horizons, metadata


# ============================================================
# CITY SEARCH
# ============================================================

def find_city(city_name):
    cities = search_city(city_name)
    if not cities:
        return []

    unique_cities = []
    seen = set()
    for city in cities:
        coord = (round(city["latitude"], 4), round(city["longitude"], 4))
        if coord not in seen:
            seen.add(coord)
            unique_cities.append(city)
    return unique_cities


# ============================================================
# LIVE FORECAST & INFERENCE
# ============================================================

def get_forecast(city):
    current, hourly_df, timezone = get_weather_data(city["latitude"], city["longitude"])
    features, latest_row = prepare_prediction_features(hourly_df, current["time"])

    temp_model, hum_model, rain_model, horizons, metadata = load_models()

    pred_temp = float(temp_model.predict(features)[0])
    pred_hum = float(hum_model.predict(features)[0])
    pred_rain = int(rain_model.predict(features)[0])

    classes = list(rain_model.classes_)
    rain_idx = classes.index(1) if 1 in classes else 0
    rain_prob = float(rain_model.predict_proba(features)[0][rain_idx] * 100)

    # 95% Confidence Intervals (+/- 1.96 * RMSE)
    ci = metadata.get("confidence_intervals", {})
    temp_rmse = ci.get("temperature_rmse", 0.57)
    hum_rmse = ci.get("humidity_rmse", 3.32)

    pred_t3 = float(horizons["t3"].predict(features)[0]) if horizons["t3"] else None
    pred_t6 = float(horizons["t6"].predict(features)[0]) if horizons["t6"] else None
    pred_r3 = int(horizons["r3"].predict(features)[0]) if horizons["r3"] else None
    pred_r6 = int(horizons["r6"].predict(features)[0]) if horizons["r6"] else None

    dew_point = float(latest_row.get("dew_point", current["temperature_2m"] - ((100 - current["relative_humidity_2m"]) / 5)))
    dp_spread = current["temperature_2m"] - dew_point
    pressure_trend_3h = float(latest_row.get("pressure_tendency_3h", 0.0))

    # Calculate Advisories
    apparent_temp, thermal_cat = get_apparent_temperature(
        current["temperature_2m"], current["relative_humidity_2m"], current["wind_speed_10m"]
    )
    storm_score, storm_level = calculate_storm_threat_score(
        dp_spread, pressure_trend_3h, current["relative_humidity_2m"], rain_prob
    )
    fog_active, fog_desc = get_fog_risk(
        current["temperature_2m"], dp_spread, current["wind_speed_10m"]
    )

    # Automatically log to SQLite tracker
    try:
        log_prediction(
            city_name=city["name"],
            latitude=city["latitude"],
            longitude=city["longitude"],
            predicted_temperature=pred_temp,
            predicted_humidity=pred_hum,
            predicted_rain=pred_rain,
            rain_probability=rain_prob,
            observation_time_str=current["time"]
        )
    except Exception:
        pass

    return {
        "current": current,
        "hourly": hourly_df,
        "timezone": timezone,
        "temperature": pred_temp,
        "temp_ci": (pred_temp - 1.96 * temp_rmse, pred_temp + 1.96 * temp_rmse),
        "temp_margin": 1.96 * temp_rmse,
        "humidity": pred_hum,
        "hum_ci": (max(0.0, pred_hum - 1.96 * hum_rmse), min(100.0, pred_hum + 1.96 * hum_rmse)),
        "hum_margin": 1.96 * hum_rmse,
        "rain": pred_rain,
        "rain_probability": rain_prob,
        "dew_point": dew_point,
        "dew_point_spread": dp_spread,
        "pressure_trend_3h": pressure_trend_3h,
        "apparent_temperature": apparent_temp,
        "thermal_category": thermal_cat,
        "storm_threat_score": storm_score,
        "storm_threat_level": storm_level,
        "fog_description": fog_desc,
        "pred_t3": pred_t3,
        "pred_t6": pred_t6,
        "pred_r3": pred_r3,
        "pred_r6": pred_r6,
        "features": features
    }


# ============================================================
# PLOTLY DARK THEME HELPER
# ============================================================

def apply_dark_chart_layout(chart, height=380):
    chart.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAF4FF", size=12),
        margin=dict(l=20, r=20, t=35, b=45),
        hoverlabel=dict(bgcolor="#163653", font_color="white"),
        xaxis=dict(showgrid=False, tickangle=-30, automargin=True),
        yaxis=dict(showgrid=True, gridcolor="#29445E", zerolinecolor="#29445E")
    )
    return chart


# ============================================================
# HEADER
# ============================================================

st.markdown('<p class="main-title">🌤️ CloudCue</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Atmospheric Intelligence & Multi-Horizon Machine Learning</p>', unsafe_allow_html=True)
st.divider()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🌦️ CloudCue")
st.sidebar.caption("Intelligent Weather Forecasting")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Live Weather",
        "ML Predictions",
        "Historical Analysis",
        "Model Performance",
        "Forecast Tracker (MLOps)",
        "About"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("🔍 Location Search")

# Initialize default selected city if not present
if "selected_city" not in st.session_state:
    st.session_state.selected_city = {
        "name": "Guntur",
        "country": "India",
        "admin1": "Andhra Pradesh",
        "display_name": "Guntur, Andhra Pradesh, India",
        "latitude": 16.3067,
        "longitude": 80.4365
    }

# Search form allows both clicking button AND pressing Enter on keyboard
with st.sidebar.form(key="city_search_form", clear_on_submit=False):
    city_input = st.text_input(
        "Enter city name:",
        value="",
        placeholder="e.g. Vijayawada, Hyderabad, London..."
    )
    search_submitted = st.form_submit_button("🔍 Get Weather", use_container_width=True)

if search_submitted:
    if city_input.strip():
        with st.spinner(f"Resolving location for '{city_input.strip()}'..."):
            matching_cities = find_city(city_input.strip())
        if matching_cities:
            st.session_state.selected_city = matching_cities[0]
            st.sidebar.success(f"Loaded: {matching_cities[0].get('display_name', matching_cities[0]['name'])}")
            st.rerun()
        else:
            st.sidebar.error(f"Could not find '{city_input.strip()}'. Check spelling or try a nearby city.")
    else:
        st.sidebar.warning("Please enter a city name first.")

# Quick Pick buttons for common locations
st.sidebar.caption("⚡ Quick Locations:")
q_cols = st.sidebar.columns(3)
if q_cols[0].button("Vijayawada", use_container_width=True, key="quick_vijayawada"):
    st.session_state.selected_city = {
        "name": "Vijayawada", "country": "India", "admin1": "Andhra Pradesh",
        "display_name": "Vijayawada, Andhra Pradesh, India",
        "latitude": 16.5075, "longitude": 80.6466
    }
    st.rerun()

if q_cols[1].button("Guntur", use_container_width=True, key="quick_guntur"):
    st.session_state.selected_city = {
        "name": "Guntur", "country": "India", "admin1": "Andhra Pradesh",
        "display_name": "Guntur, Andhra Pradesh, India",
        "latitude": 16.3067, "longitude": 80.4365
    }
    st.rerun()

if q_cols[2].button("Hyderabad", use_container_width=True, key="quick_hyderabad"):
    st.session_state.selected_city = {
        "name": "Hyderabad", "country": "India", "admin1": "Telangana",
        "display_name": "Hyderabad, Telangana, India",
        "latitude": 17.3850, "longitude": 78.4867
    }
    st.rerun()

selected_city = st.session_state.selected_city
disp_location = selected_city.get("display_name") or f"{selected_city['name']}, {selected_city.get('country', '')}"

st.sidebar.markdown(
    f"""
    <div style="background: linear-gradient(145deg, #163653, #102A43); color: #65C7FF; padding: 12px; border-radius: 10px; border: 1px solid #28618A; margin-top: 10px;">
        📍 <b>Active Location:</b><br><span style="color:#ffffff; font-weight:600;">{disp_location}</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# WEATHER DATA FETCH
# ============================================================

weather_data = None
if page in ["Dashboard", "Live Weather", "ML Predictions"]:
    try:
        with st.spinner(f"Querying live atmospheric data for {selected_city['name']}..."):
            weather_data = get_forecast(selected_city)
    except Exception as error:
        st.error(f"Unable to load live weather data: {error}")
        st.info("Check your internet connection and verify that model files are available in `models/`.")


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if page == "Dashboard":
    st.subheader(f"📍 Weather Dashboard — {disp_location}")

    if weather_data:
        current = weather_data["current"]
        st.caption(f"Observation time: {current['time']} ({weather_data['timezone']})")

        # Environmental & Thermal Advisory Banner
        with st.container(border=True):
            st.markdown("### 🛡️ Environmental & Biometeorology Advisory")
            adv_c1, adv_c2, adv_c3 = st.columns(3)
            with adv_c1:
                st.write(f"🌡️ **Thermal Feel:** `{weather_data['apparent_temperature']:.1f} °C`")
                st.caption(weather_data['thermal_category'])
            with adv_c2:
                s_score = weather_data['storm_threat_score']
                st.write(f"⚡ **Storm Threat Index:** `{s_score}/100`")
                st.caption(weather_data['storm_threat_level'])
            with adv_c3:
                st.write(f"🌫️ **Visibility Condition:**")
                st.caption(weather_data['fog_description'])

        st.markdown("<br>", unsafe_allow_html=True)

        # Live Physical Observations
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("🌡️ Temp", f"{current['temperature_2m']} °C")
        col2.metric("💧 Humidity", f"{current['relative_humidity_2m']} %")
        col3.metric("🌫️ Dew Point", f"{weather_data['dew_point']:.1f} °C")
        trend = weather_data['pressure_trend_3h']
        trend_str = f"{trend:+.1f} hPa/3h"
        col4.metric("⏱️ Pressure", f"{current['surface_pressure']} hPa", delta=trend_str)
        col5.metric("💨 Wind", f"{current['wind_speed_10m']} km/h")
        col6.metric("🌧️ Rain (now)", f"{current['precipitation']} mm")

        st.divider()

        # ML Forecast Cards
        st.subheader("🤖 AI Next-Hour Predictions (with 95% Confidence Intervals)")

        pred1, pred2, pred3 = st.columns(3)

        margin_t = weather_data['temp_margin']
        pred1.metric(
            "Next-Hour Temperature",
            f"{weather_data['temperature']:.2f} °C",
            delta=f"±{margin_t:.2f} °C (95% CI)"
        )

        margin_h = weather_data['hum_margin']
        pred2.metric(
            "Next-Hour Humidity",
            f"{weather_data['humidity']:.2f} %",
            delta=f"±{margin_h:.1f} % (95% CI)"
        )

        prob = weather_data["rain_probability"]
        if prob < 25:
            badge_class = "badge-low"
            risk_label = "Low Rain Risk"
        elif prob < 50:
            badge_class = "badge-med"
            risk_label = "Moderate Risk"
        elif prob < 75:
            badge_class = "badge-high"
            risk_label = "High Rain Risk"
        else:
            badge_class = "badge-severe"
            risk_label = "Severe / Storm Risk"

        with pred3:
            st.metric(
                "Rainfall Status",
                "Rain Expected" if weather_data["rain"] == 1 else "No Rain",
                delta=f"{prob:.1f}% Estimated Probability"
            )

        st.markdown(
            f'Risk Assessment: <span class="{badge_class}">{risk_label} ({prob:.1f}%)</span>',
            unsafe_allow_html=True
        )

        # Multi-Horizon Projections
        if weather_data["pred_t3"] is not None and weather_data["pred_t6"] is not None:
            st.markdown("#### ⏱️ Multi-Horizon AI Forecast Outlook")
            h_col1, h_col2, h_col3 = st.columns(3)
            with h_col1:
                with st.container(border=True):
                    st.markdown("**Next 1 Hour (+1h)**")
                    st.write(f"🌡️ Temp: **{weather_data['temperature']:.2f} °C**")
                    st.write(f"🌧️ Rain: **{'Rain' if weather_data['rain'] == 1 else 'No Rain'}**")
            with h_col2:
                with st.container(border=True):
                    st.markdown("**Short-Term (+3h)**")
                    st.write(f"🌡️ Temp: **{weather_data['pred_t3']:.2f} °C**")
                    st.write(f"🌧️ Rain: **{'Rain' if weather_data['pred_r3'] == 1 else 'No Rain'}**")
            with h_col3:
                with st.container(border=True):
                    st.markdown("**Medium-Term (+6h)**")
                    st.write(f"🌡️ Temp: **{weather_data['pred_t6']:.2f} °C**")
                    st.write(f"🌧️ Rain: **{'Rain' if weather_data['pred_r6'] == 1 else 'No Rain'}**")

        # 24-Hour Plotly Forecast Charts
        st.divider()
        st.subheader("📅 24-Hour Open-Meteo Hourly Projections")

        hourly_df = weather_data["hourly"].copy()
        hourly_df["time"] = pd.to_datetime(hourly_df["time"])
        obs_time = pd.to_datetime(current["time"])
        next_24h = hourly_df[hourly_df["time"] > obs_time].head(24).copy()

        if not next_24h.empty:
            next_24h["display_time"] = next_24h["time"].dt.strftime("%d %b, %H:%M")

            with st.container(border=True):
                st.markdown("### 🌡️ Temperature Trajectory")
                t_chart = px.line(
                    next_24h, x="display_time", y="temperature_2m",
                    markers=True, labels={"display_time": "Time", "temperature_2m": "Temperature (°C)"}
                )
                t_chart.update_traces(line=dict(color="#20C5E8", width=3), marker=dict(size=7, color="#20C5E8"))
                t_chart = apply_dark_chart_layout(t_chart)
                st.plotly_chart(t_chart, use_container_width=True, key="dashboard_temp_chart")

            with st.container(border=True):
                st.markdown("### 🌧️ Expected Precipitation")
                r_chart = px.bar(
                    next_24h, x="display_time", y="precipitation",
                    labels={"display_time": "Time", "precipitation": "Precipitation (mm)"}
                )
                r_chart.update_traces(marker_color="#299CDB", marker_line_color="#65C7FF", marker_line_width=1)
                r_chart = apply_dark_chart_layout(r_chart)
                st.plotly_chart(r_chart, use_container_width=True, key="dashboard_rain_chart")


# ============================================================
# PAGE 2: LIVE WEATHER
# ============================================================

elif page == "Live Weather":
    st.subheader(f"🌍 Live Atmospheric Observations: {disp_location}")

    if weather_data:
        current = weather_data["current"]
        st.caption(f"Timezone: {weather_data['timezone']} | Observation: {current['time']}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌡️ Temperature", f"{current['temperature_2m']} °C")
        col2.metric("💧 Relative Humidity", f"{current['relative_humidity_2m']} %")
        col3.metric("🌫️ Dew Point", f"{weather_data['dew_point']:.2f} °C")
        col4.metric("💨 Wind Speed", f"{current['wind_speed_10m']} km/h")

        col5, col6 = st.columns(2)
        col5.metric("⏱️ Surface Pressure", f"{current['surface_pressure']} hPa", delta=f"{weather_data['pressure_trend_3h']:+.2f} hPa/3h")
        col6.metric("🌧️ Precipitation", f"{current['precipitation']} mm")

        st.divider()
        st.subheader("Recent 24-Hour Temperature Profile")
        hourly = weather_data["hourly"].copy()
        hourly["time"] = pd.to_datetime(hourly["time"])
        recent_hours = hourly.tail(24)

        live_chart = px.line(
            recent_hours, x="time", y="temperature_2m", markers=True,
            labels={"time": "Time", "temperature_2m": "Temperature (°C)"}
        )
        live_chart.update_traces(line=dict(color="#20C5E8", width=3))
        live_chart = apply_dark_chart_layout(live_chart)
        st.plotly_chart(live_chart, use_container_width=True, key="live_weather_temperature")

        st.subheader("Hourly Sensor Log")
        st.dataframe(recent_hours, use_container_width=True)


# ============================================================
# PAGE 3: ML PREDICTIONS & DIAGNOSTICS
# ============================================================

elif page == "ML Predictions":
    st.subheader("🤖 Machine Learning Diagnostics & Predictions")

    if weather_data:
        st.write("Detailed breakdown of machine learning predictions with meteorological diagnostic indicators.")

        col1, col2 = st.columns(2)
        low_t, high_t = weather_data["temp_ci"]
        low_h, high_h = weather_data["hum_ci"]

        col1.metric(
            "Next-Hour Temperature",
            f"{weather_data['temperature']:.2f} °C",
            f"95% CI: [{low_t:.2f} °C, {high_t:.2f} °C]"
        )
        col2.metric(
            "Next-Hour Humidity",
            f"{weather_data['humidity']:.2f} %",
            f"95% CI: [{low_h:.1f} %, {high_h:.1f} %]"
        )

        st.divider()
        st.subheader("Rainfall Probability & Atmospheric Instability")

        if weather_data["rain"] == 1:
            st.warning("⚠️ High Probability: The model predicts rain within the next hour.")
        else:
            st.success("✅ Dry Conditions: The model predicts no rainfall for the next hour.")

        prob_val = min(max(weather_data["rain_probability"] / 100, 0.0), 1.0)
        st.progress(prob_val)
        st.write(f"**Rainfall Probability Estimate:** {weather_data['rain_probability']:.2f}%")

        # Meteorological Diagnostics
        with st.container(border=True):
            st.markdown("### 🔬 Physical Meteorological Indicators")
            dp_spread = weather_data["current"]["temperature_2m"] - weather_data["dew_point"]
            p_trend = weather_data["pressure_trend_3h"]

            d1, d2 = st.columns(2)
            d1.write(f"• **Dew Point Spread:** `{dp_spread:.2f} °C`")
            d1.caption("Smaller spreads (< 2.5°C) indicate high moisture saturation, favorable for clouds and precipitation.")

            d2.write(f"• **3-Hour Barometric Tendency:** `{p_trend:+.2f} hPa`")
            d2.caption("Falling pressure (> -1.5 hPa) signals approaching storm fronts or cyclonic depressions.")

        # Multi-horizon table
        if weather_data["pred_t3"] is not None:
            st.divider()
            st.subheader("⏱️ Multi-Horizon Forecast Comparison")
            horizons_df = pd.DataFrame([
                {"Horizon": "Next 1 Hour (+1h)", "Predicted Temp (°C)": round(weather_data["temperature"], 2), "Rain Forecast": "Rain" if weather_data["rain"] == 1 else "No Rain"},
                {"Horizon": "Short-Term (+3h)", "Predicted Temp (°C)": round(weather_data["pred_t3"], 2), "Rain Forecast": "Rain" if weather_data["pred_r3"] == 1 else "No Rain"},
                {"Horizon": "Medium-Term (+6h)", "Predicted Temp (°C)": round(weather_data["pred_t6"], 2), "Rain Forecast": "Rain" if weather_data["pred_r6"] == 1 else "No Rain"}
            ])
            st.dataframe(horizons_df, use_container_width=True)


# ============================================================
# PAGE 4: HISTORICAL ANALYSIS
# ============================================================

elif page == "Historical Analysis":
    st.subheader("📊 Historical Weather Data & Seasonality")

    historical_path = DATA_DIR / "weather_data.csv"
    if historical_path.exists():
        history = pd.read_csv(historical_path)
        history["time"] = pd.to_datetime(history["time"])

        st.write(f"Historical records available: **{len(history):,} hours** (1 Full Year)")

        metric_options = {
            "Temperature (°C)": "temperature_2m",
            "Humidity (%)": "relative_humidity_2m",
            "Precipitation (mm)": "precipitation",
            "Wind Speed (km/h)": "wind_speed_10m",
            "Surface Pressure (hPa)": "surface_pressure"
        }

        col_a, col_b = st.columns(2)
        with col_a:
            selected_metric_label = st.selectbox("Choose weather parameter", list(metric_options.keys()))
            selected_metric = metric_options[selected_metric_label]

        with col_b:
            history_period = st.selectbox("Time Window", ["Last 7 days", "Last 30 days", "All available data"])

        if history_period == "Last 7 days":
            chart_history = history.tail(24 * 7)
        elif history_period == "Last 30 days":
            chart_history = history.tail(24 * 30)
        else:
            chart_history = history

        with st.container(border=True):
            st.markdown(f"### {selected_metric_label} Trend")
            history_chart = px.line(chart_history, x="time", y=selected_metric, labels={"time": "Date", selected_metric: selected_metric_label})
            history_chart.update_traces(line=dict(color="#20C5E8", width=2))
            history_chart = apply_dark_chart_layout(history_chart, height=420)
            st.plotly_chart(history_chart, use_container_width=True, key="historical_chart")

        st.subheader("Historical Records Preview")
        st.dataframe(history.tail(25), use_container_width=True)
    else:
        st.error("Historical dataset not found.")


# ============================================================
# PAGE 5: MODEL PERFORMANCE & FEATURE IMPORTANCES
# ============================================================

elif page == "Model Performance":
    st.subheader("📈 Machine Learning Performance & Explainability")

    eval_path = DATA_DIR / "model_evaluation.csv"
    fi_path = DATA_DIR / "feature_importance.csv"

    if eval_path.exists():
        eval_df = pd.read_csv(eval_path)

        st.markdown("### Model Benchmark Comparison")
        st.dataframe(eval_df, use_container_width=True)

        targets = eval_df["Target"].unique().tolist()
        selected_target = st.selectbox("Filter by Prediction Target", targets)

        target_df = eval_df[eval_df["Target"] == selected_target]
        num_cols = target_df.select_dtypes(include="number").dropna(how="all", axis=1).columns.tolist()

        if num_cols:
            metric_to_plot = st.selectbox("Select Evaluation Metric", num_cols)
            with st.container(border=True):
                st.markdown(f"### {selected_target} — {metric_to_plot} Comparison")
                fig = px.bar(
                    target_df, x="Model", y=metric_to_plot, color="Model",
                    labels={"Model": "Algorithm", metric_to_plot: metric_to_plot}
                )
                fig = apply_dark_chart_layout(fig, height=380)
                st.plotly_chart(fig, use_container_width=True, key="eval_metric_plot")

    # Feature Importance Visualization
    if fi_path.exists():
        st.divider()
        st.subheader("🔍 Feature Importance & Model Drivers")
        st.caption("Derived from tree-based gain and split metrics, showing what physical variables most influence model decisions.")

        fi_df = pd.read_csv(fi_path)
        fi_targets = [c.replace("_Importance_%", "") for c in fi_df.columns if c.endswith("_Importance_%")]

        if fi_targets:
            chosen_fi_target = st.selectbox("Choose Target for Feature Attribution", fi_targets)
            fi_col = f"{chosen_fi_target}_Importance_%"

            top_fi = fi_df[["Feature", fi_col]].sort_values(by=fi_col, ascending=True).tail(12)

            with st.container(border=True):
                st.markdown(f"### Top 12 Predictive Features for {chosen_fi_target.capitalize()}")
                fi_chart = px.bar(
                    top_fi, x=fi_col, y="Feature", orientation="h",
                    labels={fi_col: "Attribution (%)", "Feature": "Engineered Variable"},
                    color=fi_col, color_continuous_scale="Blues"
                )
                fi_chart = apply_dark_chart_layout(fi_chart, height=420)
                st.plotly_chart(fi_chart, use_container_width=True, key="feature_importance_plot")


# ============================================================
# PAGE 6: FORECAST TRACKER & MLOPS
# ============================================================

elif page == "Forecast Tracker (MLOps)":
    st.subheader("📡 Live Prediction Tracker & Model Drift (MLOps)")
    st.caption("Tracks predictions live in SQLite, logging real inference and comparing against verified ground-truth.")

    metrics = get_tracker_metrics()

    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    t_col1.metric("Total Predictions Logged", f"{metrics['total_logged']}")
    t_col2.metric("Verified Ground Truths", f"{metrics['total_verified']}")

    temp_mae_str = f"{metrics['temp_mae']:.2f} °C" if metrics['temp_mae'] is not None else "Pending"
    t_col3.metric("Live Real-World Temp MAE", temp_mae_str)

    rain_acc_str = f"{metrics['rain_accuracy']:.1f} %" if metrics['rain_accuracy'] is not None else "Pending"
    t_col4.metric("Live Rain Accuracy", rain_acc_str)

    st.divider()
    m_head_col, m_btn_col = st.columns([3, 1])
    with m_head_col:
        st.markdown("### 📋 Recent Live Inference Records")
    with m_btn_col:
        if st.button("🔄 Verify Past Ground Truth", use_container_width=True):
            with st.spinner("Checking past records against Open-Meteo observations..."):
                v_count, p_count = verify_pending_records()
            if v_count > 0:
                st.success(f"Verified {v_count} past record(s)!")
                st.rerun()
            else:
                st.info(f"{p_count} forecast(s) still pending (target hour in future).")

    logs_df = get_logs_dataframe(limit=50)

    if not logs_df.empty:
        display_cols = [
            "id", "created_at", "target_time", "city_name",
            "predicted_temperature", "predicted_humidity",
            "predicted_rain", "rain_probability",
            "actual_temperature", "actual_humidity", "verified"
        ]
        available_cols = [c for c in display_cols if c in logs_df.columns]
        st.dataframe(logs_df[available_cols], use_container_width=True)

        # Plot verified residuals if any exist
        verified_df = logs_df[logs_df["verified"] == 1]
        if not verified_df.empty and "actual_temperature" in verified_df.columns:
            st.markdown("#### 🎯 Predicted vs Actual Temperature (Residual Scatter)")
            sc_fig = px.scatter(
                verified_df, x="predicted_temperature", y="actual_temperature",
                color="city_name", hover_data=["target_time"],
                labels={"predicted_temperature": "Predicted (°C)", "actual_temperature": "Actual Ground Truth (°C)"}
            )
            sc_fig = apply_dark_chart_layout(sc_fig, height=400)
            st.plotly_chart(sc_fig, use_container_width=True, key="tracker_scatter")
    else:
        st.info("No prediction records logged yet. Query a city on the Dashboard or CLI to start tracking!")


# ============================================================
# PAGE 7: ABOUT
# ============================================================

elif page == "About":
    st.subheader("About CloudCue Phase 3")
    st.markdown(r"""
    **CloudCue — Atmospheric Intelligence, Multi-Horizon Machine Learning & MLOps**

    CloudCue combines live meteorological telemetry from Open-Meteo with machine learning algorithms to deliver real-time, physics-informed forecasts.

    ### 🚀 Advanced Capabilities
    - **Meteorological Physical Modeling**: Incorporates dew point calculation, dew point spread, and barometric pressure tendency ($\Delta P / 3\text{h}$).
    - **Biometeorology & Safety Advisories**: Real-time Steadman Heat Index, Wind Chill, Storm/Cloudburst Threat Index, and Fog Risk detection.
    - **Cyclical Time Transforms**: $\sin / \cos$ diurnal and annual harmonic encodings.
    - **Multi-Horizon Projections**: Multi-target forecasting for $+1\text{h}$, $+3\text{h}$, and $+6\text{h}$ horizons.
    - **Uncertainty Quantification**: 95% error margins ($\pm 1.96 \times \text{RMSE}$) calculated from test residuals.
    - **Live MLOps Tracking**: SQLite database logger recording live inference and tracking real-world model error.
    - **Top Models Deployed**:
      - Temperature: **XGBoost Regressor** ($R^2 = 0.973, \text{RMSE} = 0.57^\circ\text{C}$)
      - Humidity: **XGBoost Regressor** ($R^2 = 0.970, \text{RMSE} = 3.32\%$)
      - Rainfall: **Random Forest Classifier** ($\text{Accuracy} = 94.5\%, F_1 = 0.759$)

    ### 🛠️ Architecture Stack
    - Python 3.14
    - Scikit-Learn & XGBoost
    - SQLite3 & Joblib
    - Streamlit & Plotly Express
    - Open-Meteo REST API
    """)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("CloudCue Phase 3 | Powered by Open-Meteo, Scikit-learn, XGBoost, SQLite & Streamlit")
