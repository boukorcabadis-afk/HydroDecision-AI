
import sqlite3
import calendar
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


DB_NAME = Path(__file__).resolve().parent / "hydro_data.db"
REGION = "Taadmit–Aïn Ibel"

# Quality-control threshold: values above this require source verification.
# They are NOT automatically rescaled or replaced.
MAX_DAILY_RAINFALL_REVIEW_MM = 100.0

# Minimum number of rolling 12-month observations for each ending month.
MIN_SAMPLE_SIZE = 20
MIN_POSITIVE_SAMPLE_SIZE = 10


def calculate_spi12(region=REGION):
    """Calculate SPI-12 from validated daily precipitation records."""

    # 1. Read daily rainfall from the project database.
    with sqlite3.connect(DB_NAME) as conn:
        daily = pd.read_sql_query(
            """
            SELECT observation_date, precipitation_mm
            FROM climate_data
            WHERE TRIM(region) = TRIM(?)
            ORDER BY observation_date
            """,
            conn,
            params=(region,),
        )

    if daily.empty:
        raise ValueError(
            f"No climate data found for '{region}' in climate_data."
        )

    # 2. Validate dates and rainfall values.
    daily["date"] = pd.to_datetime(
        daily["observation_date"],
        errors="coerce",
    )

    invalid_dates = daily["date"].isna()
    if invalid_dates.any():
        raise ValueError(
            f"{int(invalid_dates.sum())} invalid observation date(s) "
            "were found. Correct the source data before calculating SPI-12."
        )

    daily["precipitation_mm"] = pd.to_numeric(
        daily["precipitation_mm"],
        errors="coerce",
    )

    # Check that rainfall values are finite when present.
    present_rainfall = daily["precipitation_mm"].dropna().to_numpy()
    if not np.isfinite(present_rainfall).all():
        raise ValueError(
            "Infinite or non-finite rainfall values were found. "
            "Check climate_data before calculating SPI-12."
        )

    negative = daily["precipitation_mm"] < 0
    if negative.any():
        raise ValueError(
            f"{int(negative.sum())} negative rainfall value(s) found. "
            "Rainfall must not be negative; check the source data."
        )

    # Flag extreme values rather than silently modifying them.
    extreme = (
        daily["precipitation_mm"] > MAX_DAILY_RAINFALL_REVIEW_MM
    )
    if extreme.any():
        examples = daily.loc[
            extreme, ["observation_date", "precipitation_mm"]
        ].head(10)

        details = examples.to_string(index=False)

        raise ValueError(
            "SPI-12 calculation stopped because daily rainfall exceeds "
            f"{MAX_DAILY_RAINFALL_REVIEW_MM:.0f} mm/day. "
            "These values require verification against the original source. "
            "No values have been changed.\n\n"
            f"Example flagged records:\n{details}"
        )

    # Missing rainfall is allowed in the database, but any month containing
    # missing rainfall will be excluded from the SPI calculation.
    if daily["date"].duplicated().any():
        duplicates = daily.loc[
            daily["date"].duplicated(keep=False), "observation_date"
        ].head(10).tolist()

        raise ValueError(
            "Duplicate daily dates were found. Resolve them before "
            f"calculating SPI-12. Examples: {duplicates}"
        )

    # 3. Aggregate daily rainfall into monthly totals.
    daily["month"] = daily["date"].dt.to_period("M")

    monthly = daily.groupby("month").agg(
        records=("date", "size"),
        valid_rainfall=("precipitation_mm", "count"),
        rainfall_mm=(
            "precipitation_mm",
            lambda values: values.sum(min_count=1),
        ),
    )

    full_period = pd.period_range(
        monthly.index.min(),
        monthly.index.max(),
        freq="M",
    )
    monthly = monthly.reindex(full_period)

    monthly.index = monthly.index.to_timestamp()

    monthly["expected_days"] = [
        calendar.monthrange(date.year, date.month)[1]
        for date in monthly.index
    ]

    # A month is complete only when every calendar day has exactly one
    # record and a valid rainfall value.
    complete = (
        (monthly["records"] == monthly["expected_days"])
        & (monthly["valid_rainfall"] == monthly["expected_days"])
    )

    monthly.loc[~complete, "rainfall_mm"] = np.nan

    if complete.sum() < 12:
        raise ValueError(
            "Fewer than 12 complete months are available. "
            "A reliable SPI-12 series cannot be calculated."
        )

    # 4. Calculate consecutive 12-month accumulated precipitation.
    rolling_rainfall = monthly["rainfall_mm"].rolling(
        window=12,
        min_periods=12,
    ).sum()

    spi = pd.Series(
        np.nan,
        index=rolling_rainfall.index,
        dtype=float,
    )

    # 5. Fit a Gamma distribution separately for each ending calendar month.
    # At least 20 valid rolling observations and 10 positive observations
    # are required for each month of the year.
    for month_number in range(1, 13):
        mask = rolling_rainfall.index.month == month_number
        sample = rolling_rainfall.loc[mask].dropna()

        if len(sample) < MIN_SAMPLE_SIZE:
            continue

        positive = sample[sample > 0]

        if len(positive) < MIN_POSITIVE_SAMPLE_SIZE:
            continue

        try:
            shape, _, scale = gamma_dist.fit(
                positive.to_numpy(),
                floc=0,
            )

            if (
                not np.isfinite(shape)
                or not np.isfinite(scale)
                or shape <= 0
                or scale <= 0
            ):
                continue

            zero_probability = float((sample <= 0).sum() / len(sample))

            probabilities = zero_probability + (
                1.0 - zero_probability
            ) * gamma_dist.cdf(
                sample.clip(lower=0).to_numpy(),
                shape,
                loc=0,
                scale=scale,
            )

            probabilities = np.clip(
                probabilities,
                1e-10,
                1.0 - 1e-10,
            )

            spi_values = norm.ppf(probabilities)

            if np.isfinite(spi_values).all():
                spi.loc[sample.index] = spi_values

        except (
            ValueError,
            FloatingPointError,
            OverflowError,
        ) as exc:
            warnings.warn(
                f"SPI-12 fitting failed for ending month "
                f"{month_number}: {exc}",
                RuntimeWarning,
            )
            continue

    # 6. Keep only dates for which SPI-12 was successfully estimated.
    results = pd.DataFrame(
        {
            "12_month_rainfall_mm": rolling_rainfall,
            "SPI_12": spi,
        }
    ).dropna(subset=["SPI_12"])

    if results.empty:
        raise ValueError(
            "No SPI-12 values could be calculated. The available record "
            "may be too short, contain incomplete months, or have too few "
            "observations for distribution fitting. A longer, quality-checked "
            "precipitation record may be required."
        )

    # 7. Classify SPI values.
    def classify_spi(value):
        if value <= -2.0:
            return "Extremely dry"
        if value <= -1.5:
            return "Severely dry"
        if value <= -1.0:
            return "Moderately dry"
        if value < 1.0:
            return "Near normal"
        if value < 1.5:
            return "Moderately wet"
        if value < 2.0:
            return "Very wet"
        return "Extremely wet"

    results["Classification"] = results["SPI_12"].apply(classify_spi)

    results.index.name = "Date"

    return results


if __name__ == "__main__":
    try:
        results = calculate_spi12()

        print(f"SPI-12 REPORT: {REGION}")
        print(f"Valid SPI-12 results: {len(results)}")
        print("\nMost recent 24 results:")
        print(results.tail(24).round(3).to_string())

        print(
            "\nNote: SPI-12 describes precipitation anomalies, "
            "not groundwater levels."
        )
        print(
            "Quality control: daily rainfall values above "
            f"{MAX_DAILY_RAINFALL_REVIEW_MM:.0f} mm/day are flagged "
            "for verification."
        )

    except (ValueError, sqlite3.Error) as exc:
        print("\nSPI-12 CALCULATION STOPPED")
        print(str(exc))
