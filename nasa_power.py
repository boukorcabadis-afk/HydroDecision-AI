
import sqlite3
from datetime import datetime, timezone

import requests

DB_NAME = "hydro_data.db"

# Approximate representative point for the study area.
# Verify the coordinates before using the results in scientific analysis.
REGION_NAME = "Taadmit–Aïn Ibel"
LATITUDE = 34.45
LONGITUDE = 3.05

START_DATE = "20000101"
END_DATE = "20251231"

NASA_API_URL = (
    "https://power.larc.nasa.gov/api/temporal/daily/point"
)

PARAMETERS = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
]


def create_table(connection):
    connection.execute("""
        CREATE TABLE IF NOT EXISTS climate_data (
            region TEXT NOT NULL,
            observation_date TEXT NOT NULL,
            precipitation_mm REAL,
            temperature_mean_c REAL,
            temperature_max_c REAL,
            temperature_min_c REAL,
            source TEXT NOT NULL,
            source_url TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (region, observation_date, source)
        )
    """)
    connection.commit()


def fetch_climate_data():
    params = {
        "parameters": ",".join(PARAMETERS),
        "community": "AG",
        "longitude": LONGITUDE,
        "latitude": LATITUDE,
        "start": START_DATE,
        "end": END_DATE,
        "format": "JSON",
        "time-standard": "UTC",
    }

    response = requests.get(
        NASA_API_URL,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()

    if "properties" not in payload:
        raise ValueError(
            "NASA POWER returned an unexpected response."
        )

    data = payload["properties"]["parameter"]

    first_parameter = PARAMETERS[0]
    dates = data[first_parameter]

    if not dates:
        raise ValueError("NASA POWER returned no daily records.")

    fetched_at = datetime.now(timezone.utc).isoformat()
    source_url = response.url

    records = []

    for date_key in dates:
        values = []

        for parameter in PARAMETERS:
            raw_value = data[parameter].get(date_key)

            # NASA POWER can use -999 for missing values.
            if raw_value is None or raw_value == -999:
                values.append(None)
            else:
                values.append(float(raw_value))

        observation_date = (
            f"{date_key[:4]}-{date_key[4:6]}-{date_key[6:8]}"
        )

        records.append((
            REGION_NAME,
            observation_date,
            *values,
            "NASA POWER",
            source_url,
            LATITUDE,
            LONGITUDE,
            START_DATE,
            END_DATE,
            fetched_at,
        ))

    return records


def main():
    print("Fetching climate data from NASA POWER...")
    records = fetch_climate_data()

    with sqlite3.connect(DB_NAME) as connection:
        create_table(connection)

        connection.executemany("""
            INSERT INTO climate_data (
                region,
                observation_date,
                precipitation_mm,
                temperature_mean_c,
                temperature_max_c,
                temperature_min_c,
                source,
                source_url,
                latitude,
                longitude,
                period_start,
                period_end,
                fetched_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(region, observation_date, source)
            DO UPDATE SET
                precipitation_mm = excluded.precipitation_mm,
                temperature_mean_c = excluded.temperature_mean_c,
                temperature_max_c = excluded.temperature_max_c,
                temperature_min_c = excluded.temperature_min_c,
                source_url = excluded.source_url,
                fetched_at = excluded.fetched_at
        """, records)

        connection.commit()

        total = connection.execute("""
            SELECT COUNT(*)
            FROM climate_data
            WHERE region = ? AND source = ?
        """, (REGION_NAME, "NASA POWER")).fetchone()[0]

    print(f"Region: {REGION_NAME}")
    print(f"Records received: {len(records)}")
    print(f"Records stored for this region: {total}")
    print(f"Period requested: {START_DATE} to {END_DATE}")
    print("Source: NASA POWER")
    print("Database update completed.")


if __name__ == "__main__":
    main()
