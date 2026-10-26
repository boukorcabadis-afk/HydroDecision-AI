import json
import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain.tools import tool
from tavily import TavilyClient
from spi_analysis import calculate_spi12


DB_NAME = str(Path(__file__).resolve().parent / "hydro_data.db")
tavily_client = TavilyClient()


# =========================================================
# ONLINE CLIMATE DATA: NASA POWER
# =========================================================

def fetch_online_climate(region: str) -> str:
    """
    Retrieve daily climate data from NASA POWER.

    Rainfall totals are withheld when quality-control checks
    identify suspicious daily values or unexpected units.
    Raw data are never rescaled or silently modified.
    """

    try:
        # 1. Geocode the requested place.
        geo_params = urlencode({
            "name": region,
            "count": 10,
            "language": "en",
            "format": "json",
        })

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search?"
            + geo_params
        )

        request = Request(
            geo_url,
            headers={"User-Agent": "HydroDecisionAI/1.0"},
        )

        with urlopen(request, timeout=20) as response:
            geo_data = json.loads(
                response.read().decode("utf-8")
            )

        places = geo_data.get("results", [])

        if not places:
            return (
                f"No geographic match found for '{region}'. "
                "Try a more specific city, province, or country name."
            )

        # Prefer an exact place-name match when available.
        normalized_region = region.strip().casefold()

        exact_matches = [
            place for place in places
            if place.get("name", "").strip().casefold()
            == normalized_region
        ]

        place = exact_matches[0] if exact_matches else places[0]

        latitude = float(place["latitude"])
        longitude = float(place["longitude"])
        place_name = place.get("name", region)
        country = place.get("country", "Unknown country")

        # 2. Define a five-year period ending five days ago.
        end_date = date.today() - timedelta(days=5)

        try:
            start_date = end_date.replace(
                year=end_date.year - 5
            )
        except ValueError:
            # Handle February 29.
            start_date = end_date.replace(
                year=end_date.year - 5,
                day=28,
            )

        print("NASA POWER REQUESTED REGION:", region)
        print("NASA POWER MATCHED LOCATION:", place_name, country)
        print("NASA POWER COORDINATES:", latitude, longitude)
        print(
            "NASA POWER DATE RANGE:",
            start_date.strftime("%Y%m%d"),
            end_date.strftime("%Y%m%d"),
        )

        # 3. Request daily data from NASA POWER.
        parameters = (
            "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,"
            "ALLSKY_SFC_SW_DWN"
        )

        api_params = urlencode({
            "parameters": parameters,
            "community": "AG",
            "longitude": longitude,
            "latitude": latitude,
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "format": "JSON",
        })

        api_url = (
            "https://power.larc.nasa.gov/api/temporal/daily/point?"
            + api_params
        )

        request = Request(
            api_url,
            headers={"User-Agent": "HydroDecisionAI/1.0"},
        )

        with urlopen(request, timeout=90) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

        if data.get("messages"):
            print("NASA POWER API MESSAGES:", data["messages"])

        properties = data.get("properties", {})
        parameters_data = properties.get("parameter", {})

        if not parameters_data:
            return (
                f"NASA POWER returned no climate data for "
                f"{place_name}. Check the API response and date range."
            )

        rainfall = parameters_data.get("PRECTOTCORR", {})
        temperature = parameters_data.get("T2M", {})
        tmax = parameters_data.get("T2M_MAX", {})
        tmin = parameters_data.get("T2M_MIN", {})
        solar = parameters_data.get("ALLSKY_SFC_SW_DWN", {})

        # 4. Inspect metadata and rainfall values.
        parameter_metadata = (
            data.get("parameters", {}).get("PRECTOTCORR", {})
        )

        rainfall_units = parameter_metadata.get("units", "Unknown")

        print("NASA POWER HEADER:", data.get("header"))
        print("NASA POWER RAINFALL METADATA:", parameter_metadata)
        print("NASA POWER RAINFALL UNITS:", rainfall_units)
        print("NASA POWER RAINFALL COUNT:", len(rainfall))
        print(
            "NASA POWER RAINFALL FIRST 10:",
            list(rainfall.items())[:10],
        )
        print(
            "NASA POWER RAINFALL LAST 10:",
            list(rainfall.items())[-10:],
        )

        valid_rainfall = {}
        missing_count = 0
        invalid_count = 0
        extreme_days = []

        for day, raw_value in rainfall.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                invalid_count += 1
                continue

            if not math.isfinite(value) or value <= -900:
                missing_count += 1
                continue

            if value < 0:
                invalid_count += 1
                continue

            valid_rainfall[day] = value

            # Quality-control flag, not proof of an API error.
            if value > 100:
                extreme_days.append((day, value))

        maximum_rainfall = (
            max(valid_rainfall.values())
            if valid_rainfall
            else None
        )

        print("NASA POWER VALID RAINFALL COUNT:", len(valid_rainfall))
        print("NASA POWER MISSING RAINFALL COUNT:", missing_count)
        print("NASA POWER INVALID RAINFALL COUNT:", invalid_count)
        print("NASA POWER RAINFALL MAX:", maximum_rainfall)
        print(
            "NASA POWER EXTREME RAINFALL DAYS (>100 mm/day):",
            extreme_days[:30],
        )

        # Check the reported units rather than guessing a conversion.
        normalized_units = (
            str(rainfall_units)
            .strip()
            .lower()
            .replace(" ", "")
        )

        accepted_units = {
            "mm/day",
            "mmday-1",
            "mmd-1",
            "mmday^-1",
            "mmd^-1",
        }

        units_ok = normalized_units in accepted_units

        # If units are unexpected or any daily value exceeds the
        # screening threshold, withhold rainfall totals.
        # Do not divide, rescale, or replace raw rainfall values.
        rainfall_qc_warning = (
            not units_ok or bool(extreme_days)
        )

        # 5. Aggregate daily data into monthly and annual summaries.
        # Values flagged by QC are never silently changed or discarded.
        months = {}
        years = {}

        all_days = set(rainfall)
        all_days.update(temperature)
        all_days.update(tmax)
        all_days.update(tmin)
        all_days.update(solar)

        for day in sorted(all_days):
            if len(day) != 8 or not day.isdigit():
                continue

            month = day[:6]

            item = months.setdefault(
                month,
                {
                    "rain": [],
                    "temp": [],
                    "max": [],
                    "min": [],
                    "solar": [],
                },
            )

            if day in valid_rainfall:
                item["rain"].append(valid_rainfall[day])
                year = day[:4]
                annual_item = years.setdefault(
                    year, {"rain": [], "days": set()}
                )
                annual_item["rain"].append(valid_rainfall[day])
                annual_item["days"].add(day)

            for source, key in (
                (temperature, "temp"),
                (tmax, "max"),
                (tmin, "min"),
                (solar, "solar"),
            ):
                raw_value = source.get(day)

                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    continue

                if math.isfinite(value) and value > -900:
                    item[key].append(value)

        if not months:
            return (
                f"No usable daily climate values were returned "
                f"for {place_name}."
            )

        def average(values):
            if not values:
                return "N/A"
            return round(sum(values) / len(values), 2)

        lines = [
            f"NASA POWER CLIMATE REPORT: {place_name}, {country}",
            f"Requested region: {region}",
            f"Matched location: {place_name}, {country}",
            f"Latitude: {latitude}",
            f"Longitude: {longitude}",
            f"Requested period: {start_date} to {end_date}",
            "Source: NASA POWER Daily API",
            f"Rainfall units reported by API: {rainfall_units}",
            f"Valid daily rainfall records: {len(valid_rainfall)}",
            "",
        ]

        if rainfall_qc_warning:
            lines.extend([
                "RAINFALL QUALITY-CONTROL WARNING:",
                (
                    "Monthly and annual sums are shown as provisional when "
                    "the units are unexpected or daily values exceed "
                    "100 mm/day. This is a screening flag, not proof that "
                    "a value is wrong."
                ),
                (
                    "Verify the API response, units, coordinates, and "
                    "flagged dates before using rainfall sums for scientific "
                    "conclusions. No values are silently rescaled."
                ),
                f"Reported rainfall units: {rainfall_units}",
                f"Maximum valid daily rainfall: {maximum_rainfall}",
                f"Days above 100 mm/day: {len(extreme_days)}",
            ])

            for day, value in extreme_days[:10]:
                lines.append(f"- {day}: {value} mm/day")

            lines.extend([
                (
                    "No rainfall values have been rescaled, divided by "
                    "24, or replaced with estimated values."
                ),
                "",
            ])

        lines.extend([
            "Monthly summaries (most recent 12 months):",
            (
                "Month | Rainfall total (mm) | Mean temp (C) | "
                "Mean max (C) | Mean min (C) | "
                "Mean solar radiation (kWh/m2/day)"
            ),
        ])

        for month in sorted(months)[-12:]:
            item = months[month]

            if item["rain"]:
                rainfall_total = round(sum(item["rain"]), 2)
                if rainfall_qc_warning:
                    rainfall_total = f"{rainfall_total} (PROVISIONAL: QC)"
            else:
                rainfall_total = "N/A"

            lines.append(
                f"{month[:4]}-{month[4:]} | "
                f"{rainfall_total} | "
                f"{average(item['temp'])} | "
                f"{average(item['max'])} | "
                f"{average(item['min'])} | "
                f"{average(item['solar'])}"
            )

        lines.extend([
            "",
            "Annual rainfall summaries (calendar years):",
            "Year | Valid daily records | Annual rainfall total (mm) | Status",
        ])

        for year in sorted(years):
            item = years[year]
            year_number = int(year)
            expected_days = (
                366 if (
                    year_number % 4 == 0
                    and (year_number % 100 != 0 or year_number % 400 == 0)
                ) else 365
            )
            valid_days = len(item["days"])
            complete = valid_days == expected_days
            if not complete:
                annual_total = "INCOMPLETE: not reported"
                status = f"needs {expected_days} daily records"
            else:
                annual_total = f"{sum(item['rain']):.2f}"
                status = "complete year"
                if rainfall_qc_warning:
                    status += "; provisional QC review"
            lines.append(
                f"{year} | {valid_days}/{expected_days} | "
                f"{annual_total} | {status}"
            )

        lines.extend([
            "",
            "Annual totals are sums of valid daily values for complete "
            "calendar years only. Partial years are not presented as annual "
            "totals. If a QC warning exists, a complete-year sum remains "
            "provisional until flagged values are checked.",
            "",
            "Data interpretation:",
            "- Temperature and solar radiation are daily-value averages.",
            "- Missing values may affect monthly summaries.",
            (
                "- Monthly totals are provisional when QC flags are present; "
                "annual totals are shown only for complete calendar years."
            ),
            (
                "- NASA POWER provides gridded estimates, not necessarily "
                "direct measurements from a local weather station."
            ),
            (
                "- Climate data alone do not establish groundwater "
                "levels, recharge, or aquifer depletion."
            ),
            "",
            "NASA POWER: https://power.larc.nasa.gov/",
        ])

        return "\n".join(lines)

    except Exception as e:
        return (
            f"NASA POWER retrieval failed for '{region}': {e}. "
            "Check your internet connection and the API response."
        )


# =========================================================
# TOOL 1: WELLS DATA — REAL CSV FILES
# =========================================================

from pathlib import Path
import csv
import unicodedata

DATA_DIR = Path(__file__).resolve().parent / "data"

WELL_CSV_FILES = {
    "Ain El Ibel": "F_Ain El Ibel.FAITE.csv",
    "Ain Ibel": "F_Ain El Ibel.FAITE.csv",
    "عين الإبل": "F_Ain El Ibel.FAITE.csv",
    "Moudjbara": "F_Moudjbara.FAITE.csv",
    "Moudjbara ": "F_Moudjbara.FAITE.csv",
    "مجبارة": "F_Moudjbara.FAITE.csv",
    "Taadmit": "F_Taadmite.FAITE.csv",
    "Taadmite": "F_Taadmite.FAITE.csv",
    "تعظميت": "F_Taadmite.FAITE.csv",
    "Zaccar": "F_Zaccar.FAITE.csv",
    "Zeccar": "F_Zaccar.FAITE.csv",
    "زكار": "F_Zaccar.FAITE.csv",
}


def _normalize_region(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.strip().casefold())
    return "".join(
        char for char in value
        if not unicodedata.combining(char)
    )


@tool
def get_well_data(region: str) -> str:
    """Read real groundwater well records from the four local CSV files."""

    try:
        requested_region = _normalize_region(region)

        matched_file = None
        for name, filename in WELL_CSV_FILES.items():
            if _normalize_region(name) == requested_region:
                matched_file = filename
                break

        if matched_file is None:
            return (
                f"Unknown study area: {region}. "
                "Available areas: Ain El Ibel, Moudjbara, "
                "Taadmit, and Zaccar."
            )

        file_path = DATA_DIR / matched_file

        if not file_path.is_file():
            return f"Well data file not found: {file_path.name}"

        with file_path.open(
            "r", encoding="cp1252", newline=""
        ) as csv_file:
            rows = list(csv.DictReader(csv_file))

        if not rows:
            return f"No well records found in {matched_file}."

        lines = [
            f"REAL WELL DATA: {region}",
            f"Source file: {matched_file}",
            f"Number of records: {len(rows)}",
            "",
        ]

        for row in rows:
            well_name = (
                row.get("Nome de forage")
                or row.get("Code de l'ouvrage")
                or "Name unavailable"
            ).strip()

            code = (row.get("Code de l'ouvrage") or "").strip()
            depth = (row.get("Profonder") or "").strip()
            static_level = (row.get("Niveau statique") or "").strip()
            dynamic_level = (row.get("Niveau dynamique") or "").strip()
            status = (row.get("État de l'ouvrage") or "").strip()
            mobilized_flow = (row.get("Débit mobilisé") or "").strip()
            exploited_flow = (row.get("Débit exploité") or "").strip()

            lines.append(
                f"- Well: {well_name}; "
                f"Code: {code or 'not available'}; "
                f"Depth: {depth or 'not available'} m; "
                f"Static level: {static_level or 'not available'} m; "
                f"Dynamic level: {dynamic_level or 'not available'} m; "
                f"Status: {status or 'not available'}; "
                f"Mobilized flow: {mobilized_flow or 'not available'} m3/day; "
                f"Exploited flow: {exploited_flow or 'not available'} m3/day."
            )

        lines.append(
            "\nNote: Missing values are reported as unavailable, "
            "not estimated. Flow units follow the source dataset."
        )

        return "\n".join(lines)

    except (OSError, csv.Error, UnicodeError) as e:
        return f"Well CSV reading error: {e}"


# =========================================================
# TOOL 2: SATELLITE DATA
# =========================================================

@tool
def get_satellite_data(region: str) -> str:
    """Fetch locally stored vegetation, rainfall, and drought indicators."""

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT vegetation_index, rainfall_mm, drought_risk_level
                FROM satellite_data
                WHERE TRIM(region) = TRIM(?)
                """,
                (region,),
            )
            row = cursor.fetchone()

        if not row:
            return (
                f"No satellite data found for {region} "
                "in the local database."
            )

        return (
            f"LOCAL SATELLITE DATA: {region}\n"
            f"Vegetation index={row[0]}; "
            f"rainfall={row[1]} mm; drought risk={row[2]}"
        )

    except sqlite3.Error as e:
        return f"Satellite database error: {e}"


# =========================================================
# TOOL 3: CLIMATE DATA — LOCAL DATABASE + NASA POWER FALLBACK
# =========================================================

@tool
def get_climate_data(region: str) -> str:
    """Get stored climate records, or retrieve climate data from NASA POWER."""

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT COUNT(*), MIN(observation_date),
                       MAX(observation_date)
                FROM climate_data
                WHERE TRIM(region) = TRIM(?)
                """,
                (region,),
            )

            count, first_date, last_date = cursor.fetchone()

            if count:
                cursor.execute(
                    """
                    SELECT
                        SUBSTR(observation_date, 1, 7) AS month,
                        COUNT(*),
                        ROUND(SUM(precipitation_mm), 2),
                        ROUND(AVG(temperature_mean_c), 2),
                        ROUND(AVG(temperature_max_c), 2),
                        ROUND(AVG(temperature_min_c), 2)
                    FROM climate_data
                    WHERE TRIM(region) = TRIM(?)
                    GROUP BY SUBSTR(observation_date, 1, 7)
                    ORDER BY month DESC
                    LIMIT 12
                    """,
                    (region,),
                )
                monthly_rows = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT DISTINCT source, source_url
                    FROM climate_data
                    WHERE TRIM(region) = TRIM(?)
                    LIMIT 5
                    """,
                    (region,),
                )
                sources = cursor.fetchall()

                lines = [
                    f"LOCAL CLIMATE DATA REPORT: {region}",
                    f"Records available: {count}",
                    f"Data coverage: {first_date} to {last_date}",
                    "",
                    "Monthly summaries:",
                    (
                        "Month | Records | Rainfall (mm) | "
                        "Mean temp (C) | Mean max (C) | Mean min (C)"
                    ),
                ]

                for row in reversed(monthly_rows):
                    month, records, rain, mean_t, max_t, min_t = row
                    lines.append(
                        f"{month} | {records} | {rain} | "
                        f"{mean_t} | {max_t} | {min_t}"
                    )

                lines.extend(["", "Recorded data sources:"])

                for source, url in sources:
                    lines.append(f"- {source or 'Source not recorded'}")
                    if url:
                        lines.append(f"  URL: {url}")

                lines.append(
                    "\nThese are records already stored locally. "
                    "They are not automatically verified against NASA POWER."
                )

                return "\n".join(lines)

        return fetch_online_climate(region)

    except sqlite3.Error as e:
        return (
            f"Climate database error: {e}. "
            "Check the local database structure."
        )


# =========================================================
# TOOL 4: AGRICULTURE DATA
# =========================================================

@tool
def get_agriculture_data(region: str) -> str:
    """Fetch locally stored agricultural activity and soil information."""

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT main_crop, fertilizer_usage_tons, soil_type
                FROM agriculture
                WHERE TRIM(region) = TRIM(?)
                """,
                (region,),
            )
            row = cursor.fetchone()

        if not row:
            return (
                f"No agricultural data found for {region} "
                "in the local database."
            )

        return (
            f"LOCAL AGRICULTURE DATA: {region}\n"
            f"Main crop={row[0]}; fertilizer usage={row[1]} tons; "
            f"soil type={row[2]}"
        )

    except sqlite3.Error as e:
        return f"Agriculture database error: {e}"


# =========================================================
# TOOL 5: GENERAL WEB SEARCH
# =========================================================

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for water management information and best practices."""

    try:
        return tavily_client.search(query=query)
    except Exception as e:
        return {"error": str(e)}


# =========================================================
# TOOL 6: OFFICIAL WATER REGULATIONS
# =========================================================

@tool
def search_water_regulations(query: str) -> str:
    """Search official organizations for water regulations and standards."""

    try:
        results = tavily_client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_domains=[
                "who.int",
                "fao.org",
                "un.org",
                "worldbank.org",
                "unep.org",
                "iwmi.org",
            ],
        )

        if not results.get("results"):
            return "No official sources found. Try a broader search."

        lines = ["OFFICIAL SOURCES FOUND:", ""]

        for i, result in enumerate(results["results"][:5], 1):
            lines.extend([
                f"[{i}] {result.get('title', 'Untitled')}",
                f"URL: {result.get('url', 'N/A')}",
                f"Summary: {result.get('content', '')[:250]}",
                "",
            ])

        return "\n".join(lines)

    except Exception as e:
        return f"Regulations search error: {e}"


# =========================================================
# TOOL 7: RECENT WATER NEWS
# =========================================================

@tool
def search_water_news(region: str) -> str:
    """Search for recent water crisis, drought, or contamination news."""

    try:
        results = tavily_client.search(
            query=f"water crisis drought contamination {region}",
            search_depth="advanced",
            max_results=5,
            topic="news",
        )

        if not results.get("results"):
            return f"No recent news found for {region}."

        lines = [f"RECENT WATER NEWS: {region}", ""]

        for i, result in enumerate(results["results"][:5], 1):
            lines.extend([
                f"[{i}] {result.get('title', 'Untitled')}",
                f"URL: {result.get('url', 'N/A')}",
                result.get("content", "")[:200],
                "",
            ])

        return "\n".join(lines)

    except Exception as e:
        return f"News search error: {e}"


# =========================================================
# TOOL 8: WATER ECONOMICS
# =========================================================

@tool
def calculate_water_economics(
    region: str,
    wells_at_risk: int,
    contamination_risk_score: float,
    drought_risk_score: float,
) -> str:
    """Estimate illustrative water-related costs using explicit assumptions."""

    try:
        if wells_at_risk < 0:
            return "Error: wells_at_risk cannot be negative."

        if not 0 <= contamination_risk_score <= 100:
            return "Error: contamination_risk_score must be from 0 to 100."

        if not 0 <= drought_risk_score <= 100:
            return "Error: drought_risk_score must be from 0 to 100."

        # Illustrative assumptions, not verified local prices.
        repair_cost_per_well = 45000
        emergency_water_cost_per_month = 120000
        agricultural_loss_per_risk_point = 8500
        base_mitigation_cost = 25000

        repair_cost = wells_at_risk * repair_cost_per_well
        agricultural_loss = (
            contamination_risk_score * agricultural_loss_per_risk_point
        )
        emergency_cost = emergency_water_cost_per_month * 3
        drought_cost = base_mitigation_cost * drought_risk_score / 100

        total_inaction = (
            repair_cost + agricultural_loss + emergency_cost + drought_cost
        )

        action_cost = (
            base_mitigation_cost
            + wells_at_risk * repair_cost_per_well * 0.6
        )

        savings = total_inaction - action_cost
        roi_percent = (
            savings / action_cost * 100 if action_cost > 0 else 0
        )

        return f"""
ILLUSTRATIVE ECONOMIC ANALYSIS: {region}

Assumptions:
- Repair cost per well: ${repair_cost_per_well:,.0f}
- Emergency supply: ${emergency_water_cost_per_month:,.0f}/month
- These figures are example assumptions, not verified local quotations.

COST OF INACTION SCENARIO:
- Well repairs: ${repair_cost:,.0f}
- Illustrative agricultural losses: ${agricultural_loss:,.0f}
- Emergency water supply (3 months): ${emergency_cost:,.0f}
- Drought-related allowance: ${drought_cost:,.0f}
- TOTAL: ${total_inaction:,.0f}

PREVENTIVE ACTION SCENARIO:
- Mitigation allowance: ${base_mitigation_cost:,.0f}
- Preventive well repairs: ${wells_at_risk * repair_cost_per_well * 0.6:,.0f}
- TOTAL: ${action_cost:,.0f}

COMPARISON:
- Difference (inaction minus action): ${savings:,.0f}
- Illustrative ROI: {roi_percent:.1f}%
- Contamination risk score: {contamination_risk_score:.1f}/100
- Drought risk score: {drought_risk_score:.1f}/100

Important: These are scenario estimates based on assumed costs, not a
validated economic forecast. Verify local prices and avoid interpreting
the scores as measured probabilities.
"""

    except Exception as e:
        return f"Economic calculation error: {e}"


# =========================================================
# TOOL 9: ANNUAL CLIMATE SUMMARY — LOCAL DATA ONLY
# =========================================================

@tool
def get_annual_climate_summary(region: str) -> str:
    """Summarize annual rainfall from local climate records."""

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    SUBSTR(observation_date, 1, 4) AS year,
                    COUNT(*),
                    COUNT(precipitation_mm),
                    ROUND(SUM(precipitation_mm), 2),
                    ROUND(AVG(temperature_mean_c), 2)
                FROM climate_data
                WHERE TRIM(region) = TRIM(?)
                GROUP BY SUBSTR(observation_date, 1, 4)
                ORDER BY year
                """,
                (region,),
            )
            rows = cursor.fetchall()

        if not rows:
            return (
                f"No local annual climate records found for {region}. "
                "Use get_climate_data for the NASA POWER fallback."
            )

        complete_years = []

        for year, records, rain_records, rain, temp in rows:
            year_number = int(year)
            expected_days = (
                366 if (
                    year_number % 4 == 0
                    and (
                        year_number % 100 != 0
                        or year_number % 400 == 0
                    )
                ) else 365
            )

            if (
                records == expected_days
                and rain_records == expected_days
                and rain is not None
            ):
                complete_years.append((year, rain))

        if not complete_years:
            return (
                "No complete years of daily rainfall records were found "
                "in the local database."
            )

        mean_rain = sum(
            rain for _, rain in complete_years
        ) / len(complete_years)

        lines = [
            f"ANNUAL CLIMATE SUMMARY: {region}",
            f"Complete years: {len(complete_years)}",
            f"Period: {complete_years[0][0]} to {complete_years[-1][0]}",
            f"Mean annual rainfall: {mean_rain:.2f} mm",
            "",
            "Year | Records | Annual rainfall (mm) | Mean temperature (C) "
            "| Difference from average (%)",
        ]

        for year, records, rain_records, rain, temp in rows:
            year_number = int(year)
            expected_days = (
                366 if (
                    year_number % 4 == 0
                    and (
                        year_number % 100 != 0
                        or year_number % 400 == 0
                    )
                ) else 365
            )

            if (
                records != expected_days
                or rain_records != expected_days
                or rain is None
            ):
                lines.append(
                    f"{year} | {records} | Incomplete | {temp} | N/A"
                )
                continue

            anomaly = (
                (rain - mean_rain) / mean_rain * 100
                if mean_rain
                else 0
            )

            lines.append(
                f"{year} | {records} | {rain:.2f} | "
                f"{temp} | {anomaly:+.1f}%"
            )

        lines.append(
            "\nThis comparison uses complete years in the local database. "
            "It is not, by itself, a formal drought index."
        )

        return "\n".join(lines)

    except sqlite3.Error as e:
        return f"Annual climate database error: {e}"


# =========================================================
# TOOL 10: SPI-12 — LOCAL DATA THROUGH spi_analysis.py
# =========================================================

@tool
def get_spi12_analysis(region: str) -> str:
    """Get SPI-12 results calculated by spi_analysis.py for a region."""

    try:
        results = calculate_spi12(region)

        if results.empty:
            return (
                f"No SPI-12 results are available for {region}. "
                "Check whether the region has suitable precipitation "
                "records in the data source used by spi_analysis.py."
            )

        latest = results.iloc[-1]
        recent = results.tail(12)

        lines = [
            f"SPI-12 DROUGHT ANALYSIS: {region}",
            f"Latest date: {results.index[-1].strftime('%Y-%m')}",
            f"Latest SPI-12: {latest['SPI_12']:.2f}",
            f"Classification: {latest['Classification']}",
            (
                "12-month accumulated rainfall: "
                f"{latest['12_month_rainfall_mm']:.1f} mm"
            ),
            "",
            "Recent monthly results:",
        ]

        for date_index, row in recent.iterrows():
            lines.append(
                f"{date_index.strftime('%Y-%m')}: "
                f"SPI-12={row['SPI_12']:.2f}; "
                f"{row['Classification']}; "
                f"rainfall={row['12_month_rainfall_mm']:.1f} mm"
            )

        lines.append(
            "\nSPI-12 describes precipitation anomalies. It does not "
            "directly measure groundwater levels or prove depletion."
        )

        return "\n".join(lines)

    except Exception as e:
        return f"SPI-12 analysis failed for {region}: {e}"
