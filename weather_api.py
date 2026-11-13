
import difflib
import requests
import pandas as pd
from utils.features import calculate_meteorological_features, FEATURE_COLUMNS


# -----------------------------------------
# 1. API CONFIGURATION & CITY ALIASES
# -----------------------------------------

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Common spelling variants, phonetic forms, and regional nicknames
CITY_ALIASES = {
    # Andhra Pradesh & Telangana
    "vijaywada": "Vijayawada",
    "vijayawada": "Vijayawada",
    "bezawada": "Vijayawada",
    "guntur": "Guntur",
    "vizag": "Visakhapatnam",
    "vishakapatnam": "Visakhapatnam",
    "visakhapatnam": "Visakhapatnam",
    "visakhapatnam city": "Visakhapatnam",
    "hyd": "Hyderabad",
    "hyderabad": "Hyderabad",
    "secunderabad": "Secunderabad",
    "tirupati": "Tirupati",
    "tirupathi": "Tirupati",
    "rajahmundry": "Rajahmundry",
    "rajamahendravaram": "Rajahmundry",
    "kakinada": "Kakinada",
    "nellore": "Nellore",
    "kurnool": "Kurnool",
    "kadapa": "Kadapa",
    "cudappah": "Kadapa",
    "anantapur": "Anantapur",
    "ananthapur": "Anantapur",
    "warangal": "Warangal",
    "karimnagar": "Karimnagar",
    "nizamabad": "Nizamabad",
    "khammam": "Khammam",
    "chittoor": "Chittoor",
    "ongole": "Ongole",
    "eluru": "Eluru",
    "machilipatnam": "Machilipatnam",

    # Major Indian Cities
    "bangalore": "Bengaluru",
    "banglore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "chennai": "Chennai",
    "chenai": "Chennai",
    "madras": "Chennai",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "delhi": "New Delhi",
    "new delhi": "New Delhi",
    "dilli": "New Delhi",
    "kolkata": "Kolkata",
    "calcutta": "Kolkata",
    "pune": "Pune",
    "poona": "Pune",
    "ahmedabad": "Ahmedabad",
    "jaipur": "Jaipur",
    "lucknow": "Lucknow",
    "kanpur": "Kanpur",
    "nagpur": "Nagpur",
    "indore": "Indore",
    "bhopal": "Bhopal",
    "patna": "Patna",
    "vadodara": "Vadodara",
    "baroda": "Vadodara",
    "coimbatore": "Coimbatore",
    "madurai": "Madurai",
    "kochi": "Kochi",
    "cochin": "Kochi",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "calicut": "Kozhikode",
    "kozhikode": "Kozhikode",
    "mangalore": "Mangalore",
    "mangaluru": "Mangalore",
    "mysore": "Mysuru",
    "mysuru": "Mysuru",
    "chandigarh": "Chandigarh",
    "amritsar": "Amritsar",
    "shimla": "Shimla",
    "dehradun": "Dehradun",
    "varanasi": "Varanasi",
    "banaras": "Varanasi",
    "agra": "Agra",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "noida": "Noida",
    "ghaziabad": "Ghaziabad",
    "goa": "Panaji",
    "panaji": "Panaji",

    # Major Global Cities
    "nyc": "New York",
    "new york": "New York",
    "sf": "San Francisco",
    "san francisco": "San Francisco",
    "la": "Los Angeles",
    "los angeles": "Los Angeles",
    "chicago": "Chicago",
    "london": "London",
    "paris": "Paris",
    "tokyo": "Tokyo",
    "dubai": "Dubai",
    "singapore": "Singapore",
    "sydney": "Sydney",
    "melbourne": "Melbourne",
    "toronto": "Toronto",
    "berlin": "Berlin",
    "rome": "Rome"
}


# -----------------------------------------
# 2. SEARCH FOR A CITY WITH SMART FUZZY RESOLUTION
# -----------------------------------------

def search_city(city_name):
    query = city_name.strip()
    if not query:
        return None

    normalized = query.lower()

    # Step 1: Direct alias lookup
    target_query = CITY_ALIASES.get(normalized, query)

    # Helper to call Open-Meteo
    def query_open_meteo(name_to_search):
        params = {
            "name": name_to_search,
            "count": 5,
            "language": "en",
            "format": "json"
        }
        try:
            resp = requests.get(GEOCODING_URL, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results") or []
        except Exception as error:
            print(f"Geocoding request failed for {name_to_search}: {error}")
            return []

    results = query_open_meteo(target_query)

    # Step 2: If no results, try fuzzy match against known aliases
    if not results:
        fuzzy_matches = difflib.get_close_matches(normalized, list(CITY_ALIASES.keys()), n=1, cutoff=0.70)
        if fuzzy_matches:
            fallback_target = CITY_ALIASES[fuzzy_matches[0]]
            results = query_open_meteo(fallback_target)

    # Step 3: If still no results and original query differed from target, try original
    if not results and target_query != query:
        results = query_open_meteo(query)

    if not results:
        print(f"City '{city_name}' not found!")
        return None

    cities = []
    for city in results:
        admin_region = city.get("admin1", "")
        country = city.get("country", "Unknown")
        display_label = f"{city['name']}"
        if admin_region:
            display_label += f", {admin_region}"
        display_label += f", {country}"

        cities.append({
            "name": city["name"],
            "country": country,
            "admin1": admin_region,
            "display_name": display_label,
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "timezone": city.get("timezone", "UTC")
        })

    return cities


# -----------------------------------------
# 3. GET LIVE WEATHER AND HOURLY DATA
# -----------------------------------------

def get_weather_data(latitude, longitude):

    params = {
        "latitude": latitude,
        "longitude": longitude,

        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure",
            "weather_code"
        ],

        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "surface_pressure"
        ],

        "past_days": 2,
        "forecast_days": 1,
        "timezone": "auto"
    }

    try:
        response = requests.get(
            WEATHER_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        current_weather = data.get("current")

        if not current_weather:
            raise ValueError("Current weather data unavailable.")

        hourly_data = data.get("hourly", {})

        df = pd.DataFrame(hourly_data)

        if df.empty:
            raise ValueError("Hourly weather data unavailable.")

        df["time"] = pd.to_datetime(df["time"])

        df = df.sort_values("time").reset_index(drop=True)

        return current_weather, df, data.get("timezone", "UTC")

    except requests.RequestException as error:
        raise RuntimeError(f"Weather API request failed: {error}")

    except (KeyError, ValueError) as error:
        raise RuntimeError(f"Weather data processing failed: {error}")


# -----------------------------------------
# 4. PREPARE INPUT FOR ML PREDICTION
# -----------------------------------------

def prepare_prediction_features(hourly_df, current_time):

    df = hourly_df.copy()

    weather_columns = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "surface_pressure"
    ]

    # Keep only observations available at or before current time
    current_time = pd.to_datetime(current_time)
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["time"] <= current_time].copy()
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) < 26:
        raise ValueError(
            "Not enough recent hourly observations for ML prediction."
        )

    # Convert weather values to numeric
    for column in weather_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Handle missing values
    df[weather_columns] = df[weather_columns].interpolate(
        method="linear",
        limit=3
    )

    # Use complete observations
    df = df.dropna(subset=weather_columns).reset_index(drop=True)

    if len(df) < 26:
        raise ValueError(
            "Not enough complete weather observations."
        )

    # Calculate exact meteorological, cyclical, and lagged features
    df_features = calculate_meteorological_features(df)
    latest = df_features.iloc[-1]

    # Return exactly matching feature columns as DataFrame
    features_df = pd.DataFrame([latest[FEATURE_COLUMNS].to_dict()], columns=FEATURE_COLUMNS)

    return features_df, latest


# -----------------------------------------
# 5. TEST THE WEATHER API
# -----------------------------------------

if __name__ == "__main__":

    city_name = input("Enter city name: ")

    cities = search_city(city_name)

    if cities:

        print("\nAvailable cities:")

        for index, city in enumerate(cities):
            print(
                f"{index + 1}. {city['name']}, "
                f"{city['country']}"
            )

        try:
            choice = int(input("\nSelect city number: "))

            if choice < 1 or choice > len(cities):
                raise ValueError("Invalid city selection.")

            selected_city = cities[choice - 1]

            current, hourly_df, timezone = get_weather_data(
                selected_city["latitude"],
                selected_city["longitude"]
            )

            print("\n========== CLOUDCUE LIVE WEATHER ==========")

            print(
                f"City: {selected_city['name']}, "
                f"{selected_city['country']}"
            )

            print("Timezone:", timezone)
            print("Observation time:", current["time"])

            print(
                "Temperature:",
                current["temperature_2m"], "°C"
            )

            print(
                "Humidity:",
                current["relative_humidity_2m"], "%"
            )

            print(
                "Wind speed:",
                current["wind_speed_10m"], "km/h"
            )

            print(
                "Pressure:",
                current["surface_pressure"], "hPa"
            )

            print(
                "Precipitation:",
                current["precipitation"], "mm"
            )

            print("\nHourly records collected:", len(hourly_df))

            features, latest = prepare_prediction_features(
                hourly_df,
                current["time"]
            )

            print("\nML input features prepared successfully!")
            print(features.to_string(index=False))

        except (ValueError, IndexError) as error:
            print("Input error:", error)

        except RuntimeError as error:
            print("API error:", error)

        except Exception as error:
            print("Unexpected error:", error)
