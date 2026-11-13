import pandas as pd
from pathlib import Path
from utils.features import calculate_meteorological_features, FEATURE_COLUMNS


def preprocess_weather_data():
    input_path = Path("datasets/weather_data.csv")

    if not input_path.exists():
        raise FileNotFoundError(
            "weather_data.csv not found. Run data_collection.py first."
        )

    df = pd.read_csv(input_path)
    print("Original dataset shape:", df.shape)

    # 1. Datetime conversion and sorting
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # 2. Convert weather columns to numeric and handle missing values
    weather_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "surface_pressure"
    ]

    for column in weather_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Linear interpolation for small gaps
    df[weather_columns] = df[weather_columns].interpolate(method="linear", limit=3)
    df = df.dropna(subset=weather_columns).copy()

    # 3. Calculate meteorological, cyclical, and lagged features
    df = calculate_meteorological_features(df)

    # 4. Create prediction targets
    # Horizon 1: Next Hour (t+1)
    df["target_temperature"] = df["temperature_2m"].shift(-1)
    df["target_humidity"] = df["relative_humidity_2m"].shift(-1)
    df["target_rainfall"] = (df["precipitation"].shift(-1) > 0).astype(int)

    # Horizon 2: Short-term 3 Hours (t+3)
    df["target_temperature_3h"] = df["temperature_2m"].shift(-3)
    df["target_humidity_3h"] = df["relative_humidity_2m"].shift(-3)
    df["target_rainfall_3h"] = (df["precipitation"].shift(-3) > 0).astype(int)

    # Horizon 3: Medium-term 6 Hours (t+6)
    df["target_temperature_6h"] = df["temperature_2m"].shift(-6)
    df["target_humidity_6h"] = df["relative_humidity_2m"].shift(-6)
    df["target_rainfall_6h"] = (df["precipitation"].shift(-6) > 0).astype(int)

    # 5. Clean NaNs from shift boundaries
    df = df.dropna().reset_index(drop=True)

    # 6. Save processed dataset
    output_path = Path("datasets/processed_weather_data.csv")
    df.to_csv(output_path, index=False)

    print("\nPreprocessing completed successfully!")
    print(f"Processed dataset shape: {df.shape}")
    print(f"Engineered features count: {len(FEATURE_COLUMNS)}")
    print(f"Saved to: {output_path}")

    return df


if __name__ == "__main__":
    preprocess_weather_data()
