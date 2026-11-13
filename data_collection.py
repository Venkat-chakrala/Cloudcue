import time
import requests
import pandas as pd
from pathlib import Path


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Diverse reference climate locations
REFERENCE_CLIMATES = [
    {"city": "Guntur", "country": "India", "latitude": 16.3067, "longitude": 80.4365, "climate": "Tropical Wet & Dry"},
    {"city": "Delhi", "country": "India", "latitude": 28.6139, "longitude": 77.2090, "climate": "Subtropical Semi-Arid"},
    {"city": "London", "country": "UK", "latitude": 51.5074, "longitude": -0.1278, "climate": "Temperate Oceanic"},
    {"city": "Tokyo", "country": "Japan", "latitude": 35.6762, "longitude": 139.6503, "climate": "Humid Subtropical"},
    {"city": "Phoenix", "country": "USA", "latitude": 33.4484, "longitude": -112.0740, "climate": "Hot Desert (Arid)"}
]


def collect_single_city(
    latitude: float,
    longitude: float,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    city_name: str = "Guntur"
) -> pd.DataFrame:
    """
    Pulls hourly weather data for a single coordinate location from Open-Meteo Archive API.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure"
        ],
        "timezone": "auto"
    }

    try:
        response = requests.get(ARCHIVE_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        hourly_data = data["hourly"]
        df = pd.DataFrame(hourly_data)

        # Append location metadata
        df["city"] = city_name
        df["latitude"] = latitude
        df["longitude"] = longitude
        elevation = data.get("elevation", 0)
        df["elevation"] = elevation

        return df

    except requests.RequestException as error:
        print(f"API request failed for {city_name}: {error}")
    except (KeyError, ValueError) as error:
        print(f"Data processing failed for {city_name}: {error}")

    return None


def collect_multi_city_dataset(
    cities: list = REFERENCE_CLIMATES,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    save_path: str = "datasets/multi_city_weather_data.csv"
) -> pd.DataFrame:
    """
    Collects a multi-climate training dataset across diverse global regions.
    """
    all_frames = []

    print(f"Collecting multi-city weather data across {len(cities)} regions...")

    for item in cities:
        city_name = item["city"]
        lat = item["latitude"]
        lon = item["longitude"]
        print(f"Fetching data for {city_name} ({item.get('climate', '')})...")

        df = collect_single_city(lat, lon, start_date, end_date, city_name)
        if df is not None and not df.empty:
            all_frames.append(df)
            print(f"  -> Collected {len(df)} hourly records for {city_name}.")
        else:
            print(f"  -> Failed to retrieve data for {city_name}.")

        time.sleep(1.0)  # Rate limiting courtesy

    if not all_frames:
        print("No weather data could be collected.")
        return None

    combined_df = pd.concat(all_frames, ignore_index=True)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)

    print("\nMulti-city collection complete!")
    print(f"Total combined records: {len(combined_df):,}")
    print(f"Saved to: {output_path}")

    return combined_df


if __name__ == "__main__":
    # Default: collect/refresh standard historical data
    print("Collecting standard reference dataset (Guntur)...")
    df = collect_single_city(
        latitude=16.3067,
        longitude=80.4365,
        start_date="2025-01-01",
        end_date="2025-12-31",
        city_name="Guntur"
    )

    if df is not None:
        out = Path("datasets/weather_data.csv")
        df.to_csv(out, index=False)
        print(f"Saved {len(df)} records to {out}")
