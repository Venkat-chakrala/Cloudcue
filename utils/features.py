import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    # Base Current Observations
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",

    # Cyclical & Temporal Features
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week",

    # Meteorological Physical Features
    "dew_point",
    "dew_point_spread",
    "pressure_tendency_1h",
    "pressure_tendency_3h",
    "temperature_delta_1h",
    "temperature_delta_3h",
    "humidity_delta_1h",
    "humidity_delta_3h",

    # Lagged Historical Observations
    "temperature_lag_1",
    "temperature_lag_3",
    "temperature_lag_6",
    "temperature_lag_24",
    "humidity_lag_1",
    "humidity_lag_3",
    "humidity_lag_24",
    "pressure_lag_1",
    "pressure_lag_3",
    "wind_speed_lag_1",
    "wind_speed_lag_3",
    "precipitation_lag_1",
    "precipitation_sum_3h"
]


def calculate_meteorological_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes domain-specific meteorological, cyclical, and lagged features.
    Can be used for both training data preparation and live streaming inference.
    """
    df = df.copy()

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    df = df.sort_values("time").reset_index(drop=True)

    # 1. Cyclical Time Features
    hour = df["time"].dt.hour
    month = df["time"].dt.month
    df["day_of_week"] = df["time"].dt.day_of_week

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    # 2. Meteorological Derived Features
    # Dew point approximation: T - ((100 - RH) / 5)
    df["dew_point"] = df["temperature_2m"] - ((100.0 - df["relative_humidity_2m"]) / 5.0)
    df["dew_point_spread"] = df["temperature_2m"] - df["dew_point"]

    # Atmospheric Pressure Tendency (Barometric Trend)
    df["pressure_tendency_1h"] = df["surface_pressure"] - df["surface_pressure"].shift(1)
    df["pressure_tendency_3h"] = df["surface_pressure"] - df["surface_pressure"].shift(3)

    # Temperature & Humidity Rate of Change
    df["temperature_delta_1h"] = df["temperature_2m"] - df["temperature_2m"].shift(1)
    df["temperature_delta_3h"] = df["temperature_2m"] - df["temperature_2m"].shift(3)
    df["humidity_delta_1h"] = df["relative_humidity_2m"] - df["relative_humidity_2m"].shift(1)
    df["humidity_delta_3h"] = df["relative_humidity_2m"] - df["relative_humidity_2m"].shift(3)

    # 3. Lagged Weather Observations
    df["temperature_lag_1"] = df["temperature_2m"].shift(1)
    df["temperature_lag_3"] = df["temperature_2m"].shift(3)
    df["temperature_lag_6"] = df["temperature_2m"].shift(6)
    df["temperature_lag_24"] = df["temperature_2m"].shift(24)

    df["humidity_lag_1"] = df["relative_humidity_2m"].shift(1)
    df["humidity_lag_3"] = df["relative_humidity_2m"].shift(3)
    df["humidity_lag_24"] = df["relative_humidity_2m"].shift(24)

    df["pressure_lag_1"] = df["surface_pressure"].shift(1)
    df["pressure_lag_3"] = df["surface_pressure"].shift(3)

    df["wind_speed_lag_1"] = df["wind_speed_10m"].shift(1)
    df["wind_speed_lag_3"] = df["wind_speed_10m"].shift(3)

    df["precipitation_lag_1"] = df["precipitation"].shift(1)
    df["precipitation_sum_3h"] = (
        df["precipitation"].shift(1) +
        df["precipitation"].shift(2) +
        df["precipitation"].shift(3)
    )

    return df

