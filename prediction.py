import json
import joblib
import pandas as pd
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
from utils.tracker import log_prediction


MODEL_DIR = Path("models")


def load_models():
    temperature_model = joblib.load(MODEL_DIR / "best_temperature_model.pkl")
    humidity_model = joblib.load(MODEL_DIR / "best_humidity_model.pkl")
    rainfall_model = joblib.load(MODEL_DIR / "best_rainfall_model.pkl")

    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    # Multi-horizon models
    t3_path = MODEL_DIR / "best_temperature_3h_model.pkl"
    t6_path = MODEL_DIR / "best_temperature_6h_model.pkl"
    r3_path = MODEL_DIR / "best_rainfall_3h_model.pkl"
    r6_path = MODEL_DIR / "best_rainfall_6h_model.pkl"

    horizon_models = {
        "t3": joblib.load(t3_path) if t3_path.exists() else None,
        "t6": joblib.load(t6_path) if t6_path.exists() else None,
        "r3": joblib.load(r3_path) if r3_path.exists() else None,
        "r6": joblib.load(r6_path) if r6_path.exists() else None,
    }

    print("All trained models loaded successfully!")
    return temperature_model, humidity_model, rainfall_model, metadata, horizon_models


def predict_live_weather(latitude: float, longitude: float, city_name: str = "Target City"):
    temperature_model, humidity_model, rainfall_model, metadata, horizon_models = load_models()

    current, hourly_df, timezone = get_weather_data(latitude, longitude)
    X, latest = prepare_prediction_features(hourly_df, current["time"])

    # 1-Hour Horizon Predictions
    predicted_temperature = float(temperature_model.predict(X)[0])
    predicted_humidity = float(humidity_model.predict(X)[0])
    rainfall_prediction = int(rainfall_model.predict(X)[0])

    classes = list(rainfall_model.classes_)
    rain_index = classes.index(1) if 1 in classes else 0
    rainfall_probability = float(rainfall_model.predict_proba(X)[0][rain_index])

    # Error bounds (95% confidence intervals from test residuals: +/- 1.96 * RMSE)
    ci = metadata.get("confidence_intervals", {})
    temp_rmse = ci.get("temperature_rmse", 0.57)
    hum_rmse = ci.get("humidity_rmse", 3.32)

    temp_lower = predicted_temperature - 1.96 * temp_rmse
    temp_upper = predicted_temperature + 1.96 * temp_rmse
    hum_lower = max(0.0, predicted_humidity - 1.96 * hum_rmse)
    hum_upper = min(100.0, predicted_humidity + 1.96 * hum_rmse)

    # Multi-horizon outputs
    pred_t3 = float(horizon_models["t3"].predict(X)[0]) if horizon_models["t3"] else None
    pred_t6 = float(horizon_models["t6"].predict(X)[0]) if horizon_models["t6"] else None
    pred_r3 = int(horizon_models["r3"].predict(X)[0]) if horizon_models["r3"] else None
    pred_r6 = int(horizon_models["r6"].predict(X)[0]) if horizon_models["r6"] else None

    # Atmospheric diagnostic indicators
    dew_point = float(latest.get("dew_point", current["temperature_2m"] - ((100 - current["relative_humidity_2m"]) / 5)))
    dp_spread = current["temperature_2m"] - dew_point
    pressure_trend_3h = float(latest.get("pressure_tendency_3h", 0.0))

    # Biometeorological & Severe Weather Advisories
    apparent_temp, thermal_cat = get_apparent_temperature(
        current["temperature_2m"], current["relative_humidity_2m"], current["wind_speed_10m"]
    )
    storm_score, storm_level = calculate_storm_threat_score(
        dp_spread, pressure_trend_3h, current["relative_humidity_2m"], rainfall_probability * 100
    )
    fog_active, fog_desc = get_fog_risk(
        current["temperature_2m"], dp_spread, current["wind_speed_10m"]
    )

    # Log to SQLite tracking database for continuous verification
    try:
        log_id = log_prediction(
            city_name=city_name,
            latitude=latitude,
            longitude=longitude,
            predicted_temperature=predicted_temperature,
            predicted_humidity=predicted_humidity,
            predicted_rain=rainfall_prediction,
            rain_probability=rainfall_probability * 100,
            observation_time_str=current["time"]
        )
    except Exception as e:
        log_id = None

    print("\n" + "=" * 60)
    print(f"      CLOUDCUE - ATMOSPHERIC ML FORECAST: {city_name.upper()}")
    print("=" * 60)

    print("\n[LIVE OBSERVATIONS]")
    print(f"Observation time: {current['time']} ({timezone})")
    print(f"Temperature:      {current['temperature_2m']} °C  (Feels Like: {apparent_temp:.1f} °C — {thermal_cat})")
    print(f"Humidity:         {current['relative_humidity_2m']} %")
    print(f"Dew Point:        {dew_point:.2f} °C  (Spread: {dp_spread:.2f} °C)")
    print(f"Surface Pressure: {current['surface_pressure']} hPa (3h trend: {pressure_trend_3h:+.1f} hPa)")
    print(f"Wind Speed:       {current['wind_speed_10m']} km/h")
    print(f"Precipitation:    {current['precipitation']} mm")

    print("\n[NEXT-HOUR (t+1h) ML FORECAST]")
    print(f"Temperature:      {predicted_temperature:.2f} °C  (95% CI: [{temp_lower:.1f} °C, {temp_upper:.1f} °C])")
    print(f"Humidity:         {predicted_humidity:.2f} %   (95% CI: [{hum_lower:.1f} %, {hum_upper:.1f} %])")
    print(f"Rainfall State:   {'Rain Expected' if rainfall_prediction == 1 else 'No Rain'}")
    print(f"Rain Probability: {rainfall_probability * 100:.2f}%")

    if pred_t3 is not None and pred_t6 is not None:
        print("\n[MULTI-HORIZON PROJECTIONS]")
        print(f"  +3h Horizon -> Temp: {pred_t3:.2f} °C | Rain: {'Yes' if pred_r3 == 1 else 'No'}")
        print(f"  +6h Horizon -> Temp: {pred_t6:.2f} °C | Rain: {'Yes' if pred_r6 == 1 else 'No'}")

    print("\n[ENVIRONMENTAL & SEVERE WEATHER ADVISORY]")
    print(f"  • Storm Threat Index: {storm_score}/100 ({storm_level})")
    print(f"  • Thermal Comfort:    {thermal_cat}")
    print(f"  • Visibility/Fog:     {fog_desc}")
    if log_id:
        print(f"  • Forecast Logged:    ID #{log_id} recorded in forecast_tracker.db")

    print("\n" + "=" * 60)

    return {
        "city_name": city_name,
        "current_temperature": current["temperature_2m"],
        "apparent_temperature": apparent_temp,
        "thermal_category": thermal_cat,
        "current_humidity": current["relative_humidity_2m"],
        "dew_point": float(dew_point),
        "dew_point_spread": float(dp_spread),
        "pressure_trend_3h": float(pressure_trend_3h),
        "storm_threat_score": storm_score,
        "storm_threat_level": storm_level,
        "fog_description": fog_desc,
        "predicted_temperature": predicted_temperature,
        "temperature_ci": (temp_lower, temp_upper),
        "predicted_humidity": predicted_humidity,
        "humidity_ci": (hum_lower, hum_upper),
        "rainfall_prediction": rainfall_prediction,
        "rainfall_probability": rainfall_probability,
        "predicted_t3": pred_t3,
        "predicted_t6": pred_t6,
        "predicted_r3": pred_r3,
        "predicted_r6": pred_r6,
        "log_id": log_id
    }


def main():
    city_name = input("Enter city name: ").strip()

    if not city_name:
        print("Please enter a city name.")
        return

    cities = search_city(city_name)
    if not cities:
        print("No matching cities found.")
        return

    unique_cities = []
    seen = set()
    for city in cities:
        key = (round(city["latitude"], 4), round(city["longitude"], 4))
        if key not in seen:
            seen.add(key)
            unique_cities.append(city)

    print("\nAvailable cities:")
    for index, city in enumerate(unique_cities):
        print(f"{index + 1}. {city['name']}, {city['country']}")

    try:
        choice = int(input("\nSelect city number: "))
        if choice < 1 or choice > len(unique_cities):
            print("Invalid city selection.")
            return

        selected_city = unique_cities[choice - 1]
        print(f"\nGenerating forecast for {selected_city['name']}...")
        predict_live_weather(selected_city["latitude"], selected_city["longitude"], selected_city["name"])
        print("\nCloudCue prediction completed successfully!")

    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print("Error:", error)
    except Exception as error:
        print("Unexpected error:", error)


if __name__ == "__main__":
    main()
