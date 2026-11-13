import math


def calculate_heat_index(temperature_c: float, humidity_pct: float) -> float:
    """
    Computes the Heat Index (Feels Like temperature in Celsius)
    using the NOAA Steadman / Rothfusz regression equation.
    Applicable when temperature >= 26.7 C (80 F) and relative humidity >= 40%.
    """
    # If cool, heat index is essentially equal to ambient temperature
    if temperature_c < 26.7 or humidity_pct < 35:
        return temperature_c

    # Convert to Fahrenheit for standard NOAA constants
    T = (temperature_c * 9.0 / 5.0) + 32.0
    R = humidity_pct

    # Simple Steadman equation first
    hi = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (R * 0.094))

    # If >= 80 F, use full Rothfusz regression
    if hi >= 80.0:
        hi = (
            -42.379
            + 2.04901523 * T
            + 10.14333127 * R
            - 0.22475541 * T * R
            - 0.00683783 * (T ** 2)
            - 0.05481717 * (R ** 2)
            + 0.00122874 * (T ** 2) * R
            + 0.00085282 * T * (R ** 2)
            - 0.00000199 * (T ** 2) * (R ** 2)
        )

        # Adjustment for low humidity
        if R < 13.0 and 80.0 <= T <= 112.0:
            adj = ((13.0 - R) / 4.0) * math.sqrt((17.0 - abs(T - 95.0)) / 17.0)
            hi -= adj
        # Adjustment for high humidity
        elif R > 85.0 and 80.0 <= T <= 87.0:
            adj = ((R - 85.0) / 10.0) * ((87.0 - T) / 5.0)
            hi += adj

    # Convert back to Celsius
    hi_c = (hi - 32.0) * 5.0 / 9.0
    return round(hi_c, 1)


def calculate_wind_chill(temperature_c: float, wind_speed_kmh: float) -> float:
    """
    Computes Wind Chill Temperature in Celsius.
    Applicable when temperature <= 10 C and wind speed > 4.8 km/h.
    """
    if temperature_c > 10.0 or wind_speed_kmh <= 4.8:
        return temperature_c

    wc = (
        13.12
        + 0.6215 * temperature_c
        - 11.37 * (wind_speed_kmh ** 0.16)
        + 0.3965 * temperature_c * (wind_speed_kmh ** 0.16)
    )
    return round(wc, 1)


def get_apparent_temperature(temperature_c: float, humidity_pct: float, wind_speed_kmh: float) -> tuple[float, str]:
    """
    Determines effective 'Feels Like' temperature and thermal condition descriptor.
    """
    if temperature_c >= 26.7:
        feels_like = calculate_heat_index(temperature_c, humidity_pct)
        if feels_like >= 41.0:
            category = "Danger (Heat Cramps/Stroke Likely)"
        elif feels_like >= 32.0:
            category = "Extreme Caution (Heat Exhaustion Possible)"
        elif feels_like >= 27.0:
            category = "Caution (Fatigue with Prolonged Exposure)"
        else:
            category = "Normal Warm"
    elif temperature_c <= 10.0 and wind_speed_kmh > 4.8:
        feels_like = calculate_wind_chill(temperature_c, wind_speed_kmh)
        category = "Cold / Wind Chill Factor Active"
    else:
        feels_like = temperature_c
        category = "Comfortable / Neutral"

    return feels_like, category


def calculate_storm_threat_score(
    dew_point_spread: float,
    pressure_trend_3h: float,
    humidity_pct: float,
    rain_probability: float
) -> tuple[int, str]:
    """
    Computes a composite Atmospheric Storm / Convective Threat Score (0 to 100).
    Evaluates moisture saturation, barometric gradient, and ML probability.
    """
    score = 0

    # 1. Barometric Pressure Drop (up to 35 points)
    if pressure_trend_3h <= -3.0:
        score += 35  # Extreme pressure drop -> rapid storm/cyclone development
    elif pressure_trend_3h <= -1.5:
        score += 25  # Significant trough
    elif pressure_trend_3h <= -0.5:
        score += 12  # Mild drop
    elif pressure_trend_3h > 1.0:
        score -= 10  # Rising pressure -> clearing weather

    # 2. Moisture Saturation & Dew Point Depression (up to 30 points)
    if dew_point_spread <= 1.0:
        score += 30  # Saturated air
    elif dew_point_spread <= 2.5:
        score += 20
    elif dew_point_spread <= 4.0:
        score += 10

    # 3. Ambient Humidity (up to 15 points)
    if humidity_pct >= 90:
        score += 15
    elif humidity_pct >= 75:
        score += 10

    # 4. ML Model Prior Probability (up to 20 points)
    score += int((rain_probability / 100.0) * 20)

    # Bound between 0 and 100
    score = max(0, min(100, score))

    if score >= 75:
        level = "Severe Storm Threat / Cloudburst Likely"
    elif score >= 50:
        level = "Elevated Storm Risk"
    elif score >= 25:
        level = "Moderate Atmospheric Instability"
    else:
        level = "Stable Atmosphere"

    return score, level


def get_fog_risk(temperature_c: float, dew_point_spread: float, wind_speed_kmh: float) -> tuple[bool, str]:
    """
    Detects Radiation or Advection Fog potential.
    Occurs when air is cooled to near dew point (spread < 1.5 C) with calm to light winds (< 8 km/h).
    """
    if dew_point_spread <= 1.5 and wind_speed_kmh <= 8.0:
        return True, "Dense Fog / Low Visibility Warning (Saturated air + light wind)"
    elif dew_point_spread <= 2.5 and wind_speed_kmh <= 12.0:
        return False, "Mist / Light Haze Possible"
    return False, "Clear Visibility"

