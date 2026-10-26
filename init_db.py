import sqlite3
from pathlib import Path

DB_NAME = Path(__file__).resolve().parent / "hydro_data.db"


def create_database():
    """Create the required tables without inserting demonstration data."""
    print("Creating database tables...")

    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS wells (
            well_id TEXT PRIMARY KEY,
            region TEXT,
            water_level_meters REAL,
            pump_health_score REAL,
            last_maintenance_months_ago INTEGER
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS satellite_data (
            region TEXT PRIMARY KEY,
            vegetation_index REAL,
            rainfall_mm REAL,
            drought_risk_level TEXT
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agriculture (
            region TEXT PRIMARY KEY,
            main_crop TEXT,
            fertilizer_usage_tons REAL,
            soil_type TEXT
        )
        """)

    print(f"Database '{DB_NAME}' initialized successfully.")
    print("No demonstration rows were inserted.")
    print("Existing database records were preserved.")


if __name__ == "__main__":
    create_database()
