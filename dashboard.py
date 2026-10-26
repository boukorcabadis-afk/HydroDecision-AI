import sqlite3
import unicodedata
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from spi_analysis import calculate_spi12

DB_NAME = str(Path(__file__).resolve().parent / "hydro_data.db")
DATA_DIR = Path(__file__).resolve().parent / "data"

REGION_CSV_FILES = {
    "ain el ibel": ["F_Ain El Ibel.FAITE.csv"],
    "moudjbara": ["F_Moudjbara.FAITE.csv"],
    "taadmit": ["F_Taadmite.FAITE.csv"],
    "zaccar": ["F_Zaccar.FAITE.csv"],
}


# --------------------------------------------------
# CSV GROUNDWATER INDICATORS
# --------------------------------------------------

def normalize_region(value):
    value = unicodedata.normalize("NFKD", str(value or "").strip().casefold())
    return "".join(c for c in value if not unicodedata.combining(c))


def get_region_csv_files(region):
    name = normalize_region(region)
    if "taadmit" in name and "ibel" in name:
        filenames = ["F_Taadmite.FAITE.csv", "F_Ain El Ibel.FAITE.csv"]
    else:
        filenames = REGION_CSV_FILES.get(name, [])
    return [DATA_DIR / filename for filename in filenames
            if (DATA_DIR / filename).is_file()]


def get_region_indicators(region):
    files = get_region_csv_files(region)
    if not files:
        return None

    frames = []
    for path in files:
        try:
            try:
                df = pd.read_csv(path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(path, encoding="latin-1")
            df.columns = [str(c).strip() for c in df.columns]
            frames.append(df)
        except Exception as exc:
            st.warning(f"Could not read {path.name}: {exc}")

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    def count_available(column):
        if column not in df.columns:
            return 0
        values = df[column].astype("string").str.strip()
        missing = values.isna() | values.str.casefold().isin(
            ["", "nan", "none", "null", "n/a", "na", "-"]
        )
        return int((~missing).sum())

    return {
        "wells": len(df),
        "static": count_available("Niveau statique"),
        "dynamic": count_available("Niveau dynamique"),
        "exploited": count_available("Débit exploité"),
        "mobilized": count_available("Débit mobilisé"),
    }


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def read_table(query, params=()):
    with sqlite3.connect(DB_NAME) as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_well_data(region):
    return read_table(
        "SELECT * FROM wells WHERE TRIM(region) = TRIM(?)", (region,)
    )


def get_risk_data(region):
    sat = read_table(
        "SELECT * FROM satellite_data WHERE TRIM(region) = TRIM(?)", (region,)
    )
    agr = read_table(
        "SELECT * FROM agriculture WHERE TRIM(region) = TRIM(?)", (region,)
    )
    return sat, agr


def calendar_is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def get_annual_precipitation_data(region):
    """Return annual rainfall only for complete, valid daily records."""
    df = read_table(
        """SELECT observation_date, precipitation_mm
           FROM climate_data
           WHERE TRIM(region) = TRIM(?)
           ORDER BY observation_date""",
        (region,),
    )
    empty = pd.DataFrame(columns=["year", "annual_precipitation_mm"])
    if df.empty:
        return empty

    df["date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    df["precipitation_mm"] = pd.to_numeric(
        df["precipitation_mm"], errors="coerce"
    )
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        return empty

    if df["date"].duplicated().any():
        st.warning(
            "Duplicate daily dates were found in the climate database. "
            "Annual rainfall statistics are withheld until the records are checked."
        )
        return empty

    invalid = (df["precipitation_mm"] < 0) | (df["precipitation_mm"] > 100)
    if invalid.any():
        st.warning(
            "Annual precipitation statistics are withheld because the database "
            "contains negative rainfall values or daily values above 100 mm/day. "
            "Verify these records against the original source. No values have changed."
        )
        return empty

    df["year"] = df["date"].dt.year
    results = []
    for year, group in df.groupby("year"):
        expected = 366 if calendar_is_leap_year(year) else 365
        if len(group) == expected and group["precipitation_mm"].notna().sum() == expected:
            results.append({
                "year": int(year),
                "annual_precipitation_mm": float(group["precipitation_mm"].sum()),
            })

    result = pd.DataFrame(results, columns=["year", "annual_precipitation_mm"])
    if not result.empty:
        result["annual_precipitation_mm"] = result["annual_precipitation_mm"].round(2)
    return result


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def get_numeric_value(df, column):
    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else None


def get_text_value(df, column):
    if df.empty or column not in df.columns:
        return None
    values = df[column].dropna()
    if values.empty:
        return None
    value = str(values.iloc[0]).strip()
    return None if not value or value.lower() in {"nan", "none", "null"} else value


def create_gauge_chart(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 16, "color": "#333"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 33], "color": "#e8f5e9"},
                {"range": [33, 66], "color": "#fff9c4"},
                {"range": [66, 100], "color": "#ffebee"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": 80,
            },
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def show_risk_metric(container, label, value, description=None):
    with container:
        if value is None:
            st.metric(label, "N/A")
            st.caption(description or "Required source data unavailable.")
        else:
            st.metric(label, f"{value:.0f}/100")
            if description:
                st.caption(description)


# --------------------------------------------------
# MAIN DASHBOARD
# --------------------------------------------------

def render_dashboard(region):
    st.markdown("### 💧 Real Groundwater Indicators")
    indicators = get_region_indicators(region)

    if indicators is None:
        st.warning(
            "No CSV data found for this region. Check the selected region "
            "and CSV filenames."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Well Records", indicators["wells"])
        c2.metric("Static Level Data", f'{indicators["static"]}/{indicators["wells"]}')
        c3.metric("Dynamic Level Data", f'{indicators["dynamic"]}/{indicators["wells"]}')
        c4, c5 = st.columns(2)
        c4.metric("Exploited Flow Data", f'{indicators["exploited"]}/{indicators["wells"]}')
        c5.metric("Mobilized Flow Data", f'{indicators["mobilized"]}/{indicators["wells"]}')
        st.caption(
            "Indicators are calculated from the regional CSV records. "
            "Data availability does not imply measurement quality."
        )

    st.markdown(f"### Real-Time Dashboard: {region}")
    wells_df = get_well_data(region)
    sat_df, agr_df = get_risk_data(region)

    # SPI-12
    st.markdown("---")
    st.markdown("#### SPI-12: Meteorological Drought Analysis")
    st.caption(
        "SPI-12 describes precipitation anomalies over a 12-month period. "
        "It does not directly measure groundwater levels or groundwater recharge. "
        "Results depend on the quality and length of the precipitation record."
    )
    try:
        spi_df = calculate_spi12(region)
        if not spi_df.empty:
            latest = spi_df.iloc[-1]
            a, b = st.columns(2)
            a.metric(
                "Latest SPI-12", f"{latest['SPI_12']:.2f}",
                help="Negative values indicate drier-than-usual conditions; "
                     "positive values indicate wetter-than-usual conditions.",
            )
            b.metric("Latest Classification", latest["Classification"])
            fig_spi = go.Figure()
            fig_spi.add_trace(go.Scatter(
                x=spi_df.index, y=spi_df["SPI_12"], mode="lines",
                name="SPI-12", line=dict(width=2),
                hovertemplate="Date: %{x|%Y-%m}<br>SPI-12: %{y:.2f}<extra></extra>",
            ))
            for level in [-2, -1.5, -1, 0, 1, 1.5, 2]:
                fig_spi.add_hline(
                    y=level, line_dash="dash", annotation_text=str(level),
                    annotation_position="right",
                )
            fig_spi.update_layout(
                title=f"12-Month Standardized Precipitation Index — {region}",
                xaxis_title="Date", yaxis_title="SPI-12", height=450,
                hovermode="x unified", margin=dict(l=20, r=50, t=60, b=40),
            )
            st.plotly_chart(fig_spi, use_container_width=True)
            with st.expander("View SPI-12 data table"):
                st.dataframe(spi_df.reset_index(names="Date"), use_container_width=True)
    except Exception as exc:
        st.warning(f"SPI-12 is unavailable. Check the precipitation data and calculation requirements. Details: {exc}")

    # Annual precipitation
    st.markdown("---")
    st.markdown(f"#### Annual Precipitation — {region}")
    climate_df = get_annual_precipitation_data(region)
    if climate_df.empty:
        st.warning(f"No complete, quality-checked annual precipitation records are available for {region}.")
    else:
        mean_rainfall = climate_df["annual_precipitation_mm"].mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Annual Precipitation", f"{mean_rainfall:.2f} mm")
        driest = climate_df.loc[climate_df["annual_precipitation_mm"].idxmin()]
        c2.metric("Driest Year", str(int(driest["year"])), f"{driest['annual_precipitation_mm']:.2f} mm")
        wettest = climate_df.loc[climate_df["annual_precipitation_mm"].idxmax()]
        c3.metric("Wettest Year", str(int(wettest["year"])), f"{wettest['annual_precipitation_mm']:.2f} mm")

        fig_rain = go.Figure()
        fig_rain.add_trace(go.Scatter(
            x=climate_df["year"], y=climate_df["annual_precipitation_mm"],
            mode="lines+markers", name="Annual precipitation", line=dict(width=2),
            marker=dict(size=7),
            hovertemplate="Year: %{x}<br>Precipitation: %{y:.2f} mm<extra></extra>",
        ))
        fig_rain.add_hline(
            y=mean_rainfall, line_dash="dash",
            annotation_text=f"Period average: {mean_rainfall:.2f} mm",
        )
        fig_rain.update_layout(
            title=f"Annual Precipitation — {region}", xaxis_title="Year",
            yaxis_title="Annual Precipitation (mm)", hovermode="x unified",
            height=420, margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_rain, use_container_width=True)
        with st.expander("View annual precipitation data"):
            st.dataframe(
                climate_df.rename(columns={
                    "year": "Year",
                    "annual_precipitation_mm": "Annual Precipitation (mm)",
                }),
                use_container_width=True, hide_index=True,
            )
        st.caption(
            "Source: precipitation records stored in the project database. Only "
            "complete calendar years with daily values within the configured "
            "quality-control range are shown. The mean is descriptive and is not "
            "itself a drought index. Source data should be verified before scientific interpretation."
        )

    # Risk indicators
    st.markdown("---")
    st.markdown("#### Key Risk Indicators")
    drought_risk = None
    drought_label = get_text_value(sat_df, "drought_risk_level")
    drought_map = {"Low": 30, "Medium": 60, "High": 85}
    if drought_label is not None:
        drought_risk = drought_map.get(drought_label.strip().title())

    fert = get_numeric_value(agr_df, "fertilizer_usage_tons")
    soil = get_text_value(agr_df, "soil_type")
    contam_risk = None
    if fert is not None and fert >= 0 and soil is not None:
        contam_risk = min(100, (fert / 5) + (30 if soil.strip().lower() == "sandy" else 10))

    maintenance_risk = None
    critical_wells = None
    if not wells_df.empty and "pump_health_score" in wells_df.columns:
        pump_health = pd.to_numeric(wells_df["pump_health_score"], errors="coerce").dropna()
        pump_health = pump_health[(pump_health >= 0) & (pump_health <= 100)]
        if not pump_health.empty:
            maintenance_risk = 100 - float(pump_health.mean())
            critical_wells = int((pump_health < 50).sum())

    c1, c2, c3, c4 = st.columns(4)
    show_risk_metric(c1, "Drought Risk", drought_risk,
        (f"Source category: {drought_label}. Indicative project score, not a calibrated probability."
         if drought_risk is not None else "No recognized drought-risk category in satellite_data."))
    show_risk_metric(c2, "Contamination Risk", contam_risk,
        ("Simplified project indicator; not a validated groundwater contamination assessment."
         if contam_risk is not None else "Fertilizer use or soil type is unavailable."))
    show_risk_metric(c3, "Maintenance Risk", maintenance_risk,
        ("Calculated from available pump-health scores."
         if maintenance_risk is not None else "Valid pump-health data are unavailable."))
    with c4:
        if critical_wells is None:
            st.metric("Critical Wells", "N/A")
            st.caption("Valid pump-health data are unavailable.")
        else:
            st.metric("Critical Wells", f"{critical_wells}/{len(wells_df)}")
            st.caption("Critical means pump-health score below 50.")

    # Risk gauges
    gauges = [
        ("Drought", drought_risk, "#EF6C00"),
        ("Contamination", contam_risk, "#C62828"),
        ("Maintenance", maintenance_risk, "#1565C0"),
    ]
    gauges = [item for item in gauges if item[1] is not None]
    if gauges:
        st.markdown("---")
        st.markdown("#### Risk Level Visualization")
        for col, (name, value, color) in zip(st.columns(len(gauges)), gauges):
            with col:
                st.plotly_chart(create_gauge_chart(value, name, color), use_container_width=True)

    # Risk heatmap
    risks = [(name, value) for name, value in [
        ("Drought", drought_risk), ("Contamination", contam_risk),
        ("Maintenance", maintenance_risk),
    ] if value is not None]
    if risks:
        st.markdown("---")
        st.markdown("#### Regional Risk Heatmap")
        heat_data = pd.DataFrame(risks, columns=["Category", "Risk Score"])
        heat_fig = px.imshow(
            [heat_data["Risk Score"].values],
            labels={"x": "Risk Category", "y": "", "color": "Indicative Score"},
            x=heat_data["Category"], y=["Current Status"],
            color_continuous_scale=["#4CAF50", "#FFEB3B", "#F44336"],
            zmin=0, zmax=100, text_auto=".0f", aspect="auto",
        )
        heat_fig.update_layout(height=200, coloraxis_colorbar=dict(title="Score"))
        st.plotly_chart(heat_fig, use_container_width=True)

    # Well data from regional CSV files
    st.markdown("---")
    st.markdown("#### Well Water Levels & Pump Health")

    csv_files = get_region_csv_files(region)
    csv_frames = []

    for csv_path in csv_files:
        try:
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding="latin-1")

            df.columns = [str(col).strip() for col in df.columns]
            csv_frames.append(df)
        except Exception as exc:
            st.warning(f"Could not read {csv_path.name}: {exc}")

    if not csv_frames:
        st.warning("No regional CSV records are available for this region.")
    else:
        csv_wells = pd.concat(csv_frames, ignore_index=True)

        # Use real measurements from the CSV files
        id_col = "Nome de forage"
        static_col = "Niveau statique"
        dynamic_col = "Niveau dynamique"

        if static_col not in csv_wells.columns and dynamic_col not in csv_wells.columns:
            st.warning("No static or dynamic water-level columns were found in the CSV files.")
        else:
            if id_col not in csv_wells.columns:
                csv_wells["Well ID"] = [
                    f"Well {i + 1}" for i in range(len(csv_wells))
                ]
                id_col = "Well ID"

            plot_frames = []

            for col, label in [
                (static_col, "Static Water Level"),
                (dynamic_col, "Dynamic Water Level"),
            ]:
                if col in csv_wells.columns:
                    temp = csv_wells[[id_col, col]].copy()
                    temp[col] = pd.to_numeric(
                        temp[col].astype(str).str.replace(",", ".", regex=False),
                        errors="coerce",
                    )
                    temp = temp.dropna(subset=[col])
                    temp = temp.rename(columns={id_col: "Well", col: "Level (m)"})
                    temp["Measurement"] = label
                    plot_frames.append(temp)

            if not plot_frames:
                st.warning("The CSV files contain no valid numeric water-level measurements.")
            else:
                level_data = pd.concat(plot_frames, ignore_index=True)

                fig = px.bar(
                    level_data,
                    x="Well",
                    y="Level (m)",
                    color="Measurement",
                    barmode="group",
                    title=f"Recorded Water Levels — {region}",
                    hover_data=["Measurement"],
                )
                fig.update_layout(
                    xaxis_title="Well",
                    yaxis_title="Recorded Level (m)",
                    height=450,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "Source: regional CSV records. Only available numeric measurements "
                    "are plotted. Static and dynamic levels are shown as recorded; "
                    "missing values are not estimated."
                )

    st.info(
        "Pump health and maintenance indicators are unavailable because the "
        "current CSV records do not provide verified pump-health scores or "
        "maintenance histories."
    )

    # Agriculture
    if not agr_df.empty:
        st.markdown("---")
        st.markdown("#### Agricultural Activity Breakdown")
        main_crop = get_text_value(agr_df, "main_crop")
        ca, cb = st.columns(2)
        with ca:
            st.info(f"**Main Crop:** {main_crop or 'Not available'}")
            st.info(f"**Soil Type:** {soil or 'Not available'}")
        with cb:
            st.info("**Fertilizer Usage:** " + (f"{fert:g} tons" if fert is not None else "Not available"))
            st.caption(
                "The contamination score is a simplified project indicator. It should "
                "not be interpreted as a measured contamination level without groundwater-quality data."
            )
