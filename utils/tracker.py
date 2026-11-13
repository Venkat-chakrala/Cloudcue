import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DB_PATH = Path("datasets/forecast_tracker.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS forecast_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                target_time TEXT NOT NULL,
                city_name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                predicted_temperature REAL NOT NULL,
                predicted_humidity REAL NOT NULL,
                predicted_rain INTEGER NOT NULL,
                rain_probability REAL NOT NULL,
                actual_temperature REAL,
                actual_humidity REAL,
                actual_rain INTEGER,
                temp_error REAL,
                humidity_error REAL,
                rain_correct INTEGER,
                verified INTEGER DEFAULT 0
            )
        """)
        conn.commit()


# Initialize on import
init_db()


def log_prediction(
    city_name: str,
    latitude: float,
    longitude: float,
    predicted_temperature: float,
    predicted_humidity: float,
    predicted_rain: int,
    rain_probability: float,
    observation_time_str: str = None
) -> int:
    """
    Records a live inference prediction made for next-hour target time.
    """
    now = datetime.now()
    created_at = now.isoformat()

    # If observation time given, target is 1 hour after observation; else 1 hour from now
    if observation_time_str:
        try:
            obs_dt = pd.to_datetime(observation_time_str)
            target_dt = obs_dt + timedelta(hours=1)
            target_time = target_dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            target_time = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
    else:
        target_time = (now + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO forecast_logs (
                created_at, target_time, city_name, latitude, longitude,
                predicted_temperature, predicted_humidity, predicted_rain,
                rain_probability, verified
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            created_at,
            target_time,
            city_name,
            latitude,
            longitude,
            round(predicted_temperature, 2),
            round(predicted_humidity, 2),
            predicted_rain,
            round(rain_probability, 2)
        ))
        conn.commit()
        return cursor.lastrowid


def update_ground_truth(
    log_id: int,
    actual_temperature: float,
    actual_humidity: float,
    actual_rain: int
):
    """
    Updates a log record with the ground truth observation and evaluates error.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT predicted_temperature, predicted_humidity, predicted_rain FROM forecast_logs WHERE id = ?", (log_id,))
        row = cursor.fetchone()
        if not row:
            return

        pred_t = row["predicted_temperature"]
        pred_h = row["predicted_humidity"]
        pred_r = row["predicted_rain"]

        temp_error = abs(actual_temperature - pred_t)
        hum_error = abs(actual_humidity - pred_h)
        rain_correct = 1 if (actual_rain == pred_r) else 0

        cursor.execute("""
            UPDATE forecast_logs SET
                actual_temperature = ?,
                actual_humidity = ?,
                actual_rain = ?,
                temp_error = ?,
                humidity_error = ?,
                rain_correct = ?,
                verified = 1
            WHERE id = ?
        """, (
            round(actual_temperature, 2),
            round(actual_humidity, 2),
            actual_rain,
            round(temp_error, 2),
            round(hum_error, 2),
            rain_correct,
            log_id
        ))
        conn.commit()


def get_logs_dataframe(limit: int = 100) -> pd.DataFrame:
    """Returns recent prediction logs as a DataFrame."""
    with get_connection() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM forecast_logs ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,)
        )
    return df


def get_tracker_metrics() -> dict:
    """
    Calculates live aggregate verification metrics across verified historical forecasts.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM forecast_logs")
        total_logged = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM forecast_logs WHERE verified = 1")
        total_verified = cursor.fetchone()[0]

        if total_verified == 0:
            return {
                "total_logged": total_logged,
                "total_verified": 0,
                "temp_mae": None,
                "hum_mae": None,
                "rain_accuracy": None
            }

        cursor.execute("""
            SELECT
                AVG(temp_error) as temp_mae,
                AVG(humidity_error) as hum_mae,
                AVG(rain_correct) * 100 as rain_acc
            FROM forecast_logs
            WHERE verified = 1
        """)
        row = cursor.fetchone()

        return {
            "total_logged": total_logged,
            "total_verified": total_verified,
            "temp_mae": round(row["temp_mae"], 2) if row["temp_mae"] is not None else None,
            "hum_mae": round(row["hum_mae"], 2) if row["hum_mae"] is not None else None,
            "rain_accuracy": round(row["rain_acc"], 1) if row["rain_acc"] is not None else None
        }


def verify_pending_records() -> tuple[int, int]:
    """
    Checks unverified records where target_time has passed, queries Open-Meteo for actuals,
    and updates ground truth metrics.
    Returns (verified_count, remaining_pending_count).
    """
    import requests
    now = datetime.now()
    verified_count = 0
    pending_count = 0

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, target_time, latitude, longitude FROM forecast_logs WHERE verified = 0")
        unverified = cursor.fetchall()

    for row in unverified:
        log_id = row["id"]
        target_str = row["target_time"]
        lat = row["latitude"]
        lon = row["longitude"]

        try:
            target_dt = pd.to_datetime(target_str)
            # If target is still in the future, it cannot be verified yet
            if target_dt > now:
                pending_count += 1
                continue

            date_str = target_dt.strftime("%Y-%m-%d")
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"],
                "start_date": date_str,
                "end_date": date_str,
                "timezone": "auto"
            }
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json().get("hourly", {})
                times = data.get("time", [])
                temps = data.get("temperature_2m", [])
                hums = data.get("relative_humidity_2m", [])
                precs = data.get("precipitation", [])

                target_iso = target_dt.strftime("%Y-%m-%dT%H:00")
                if target_iso in times:
                    idx = times.index(target_iso)
                else:
                    idx = -1

                if idx >= 0 and idx < len(temps):
                    act_t = temps[idx]
                    act_h = hums[idx]
                    act_r = 1 if (precs[idx] and precs[idx] > 0) else 0
                    update_ground_truth(log_id, act_t, act_h, act_r)
                    verified_count += 1
                else:
                    pending_count += 1
        except Exception:
            pending_count += 1

    return verified_count, pending_count

