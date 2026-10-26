import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "hydro_data.db"

DEMO_REGIONS = (
    "Wadi Al-Janoub",
    "Northern Plains",
    "Eastern Oasis",
)


def main():
    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        tables = cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
        """).fetchall()

        for (table_name,) in tables:
            columns = cursor.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()
            column_names = [column[1] for column in columns]

            if "region" not in column_names:
                continue

            placeholders = ", ".join("?" for _ in DEMO_REGIONS)
            cursor.execute(
                f'DELETE FROM "{table_name}" '
                f'WHERE region IN ({placeholders})',
                DEMO_REGIONS,
            )
            print(f"{table_name}: deleted {cursor.rowcount} rows")

        conn.commit()

        print("\nVerification:")
        for (table_name,) in tables:
            columns = cursor.execute(
                f'PRAGMA table_info("{table_name}")'
            ).fetchall()
            if "region" not in [column[1] for column in columns]:
                continue

            placeholders = ", ".join("?" for _ in DEMO_REGIONS)
            remaining = cursor.execute(
                f'SELECT COUNT(*) FROM "{table_name}" '
                f'WHERE region IN ({placeholders})',
                DEMO_REGIONS,
            ).fetchone()[0]
            print(f"{table_name}: {remaining} matching rows remaining")

    print("\nDeletion completed. No backup was created.")


if __name__ == "__main__":
    main()

