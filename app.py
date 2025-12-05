import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import altair as alt
from pathlib import Path

# ============= CONFIG & PAGE LAYOUT =============

st.set_page_config(
    page_title="BI Sales Predictor – Autobahn Trucking MH",
    layout="wide",
    page_icon="🚚",
)

# ======= GLOBAL STYLES (HEADER + SECTIONS) =======

st.markdown(
    """
    <style>
    /* Global background tweak for subtle BI feel */
    .stApp {
        background: radial-gradient(circle at top left, #101726 0, #050913 45%, #050710 100%);
    }

    /* Hero container */
    .hero-wrapper {
        margin-bottom: 1.8rem;
        margin-top: 1.1rem;  /* more top spacing so header is not cut */
    }

    .hero-card {
        position: relative;
        overflow: hidden;
        border-radius: 20px;
        padding: 20px 26px 22px 26px;
        border: 1px solid rgba(212, 175, 55, 0.35);
        background: linear-gradient(135deg, #0A1A2F 0%, #10192B 45%, #121826 100%);
        box-shadow:
            0 18px 40px rgba(0, 0, 0, 0.65),
            0 0 0 1px rgba(255, 255, 255, 0.02);
        display: flex;
        flex-direction: column;
        gap: 8px;
        color: #FFFFFF;
    }

    .hero-accent-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 12px;
        border-radius: 999px;
        background: rgba(10, 26, 47, 0.5);
        border: 1px solid rgba(212, 175, 55, 0.55);
        color: #F5F5F5;
        font-size: 11px;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        font-weight: 600;
    }

    .hero-accent-pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #D4AF37;
        box-shadow: 0 0 8px rgba(212, 175, 55, 0.9);
    }

    .hero-main-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 4px;
    }

    .hero-main-text {
        max-width: 70%;
        min-width: 240px;
    }

    .hero-title {
        margin: 0;
        font-size: 30px;
        line-height: 1.1;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #FFFFFF;
    }

    @media (max-width: 768px) {
        .hero-title {
            font-size: 24px;
        }
        .hero-main-text {
            max-width: 100%;
        }
    }

    .hero-subtitle {
        margin: 3px 0 2px 0;
        font-size: 15px;
        font-weight: 500;
        color: #D4AF37;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-caption {
        margin: 2px 0 0 0;
        font-size: 12px;
        color: #CBD2E0;
        max-width: 520px;
    }

    .hero-meta {
        text-align: right;
        min-width: 210px;
    }

    .hero-meta-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #9ea3b3;
        margin-bottom: 4px;
    }

    .hero-meta-value {
        font-size: 13px;
        font-weight: 500;
        color: #FFFFFF;
    }

    .hero-meta-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-top: 6px;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(74, 79, 87, 0.28);
        border: 1px solid rgba(255, 255, 255, 0.06);
        font-size: 11px;
        color: #E5E7EE;
    }

    .hero-meta-chip-icon {
        width: 16px;
        height: 16px;
        border-radius: 999px;
        border: 1px solid rgba(212, 175, 55, 0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        color: #D4AF37;
    }

    /* Geometric accents (tech feel) */
    .hero-geo {
        position: absolute;
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(6px);
    }

    .hero-geo-one {
        width: 160px;
        height: 160px;
        right: -40px;
        top: -45px;
        background: radial-gradient(circle at 30% 30%, rgba(212, 175, 55, 0.35), transparent 60%);
        opacity: 0.9;
    }

    .hero-geo-two {
        width: 220px;
        height: 80px;
        right: -55px;
        bottom: -30px;
        background: linear-gradient(90deg, rgba(74, 79, 87, 0.5), transparent);
        opacity: 0.7;
    }

    /* New KPI row styling */
    .kpi-row {
        display: flex;
        gap: 16px;
        margin-top: 0.8rem;
        margin-bottom: 0.6rem;
        flex-wrap: wrap;
    }

    .kpi-card {
        flex: 1 1 0;
        min-width: 220px;
        padding: 14px 18px;
        border-radius: 16px;
        background: radial-gradient(circle at top left, rgba(16, 23, 38, 0.95), rgba(5, 9, 19, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.55);
        color: #F5F7FA;
    }

    .kpi-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #9EA3B3;
        margin-bottom: 4px;
    }

    .kpi-sub {
        font-size: 12px;
        color: #C3CAD9;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 600;
        line-height: 1.1;
    }

    .kpi-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        margin-top: 8px;
    }

    .kpi-pill-pos {
        background: rgba(16, 185, 129, 0.12);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.5);
    }

    .kpi-pill-neg {
        background: rgba(239, 68, 68, 0.12);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.5);
    }

    .kpi-pill-neutral {
        background: rgba(148, 163, 184, 0.16);
        color: #E5E7EB;
        border: 1px solid rgba(148, 163, 184, 0.5);
    }

    /* Section titles */
    .section-title {
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-top: 1.8rem !important;
        margin-bottom: 0.4rem !important;
        color: #F5F7FA !important;
    }

    .block-container {
        padding-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============= DATA LOADING & PREP =============

DATA_PATH = Path("MH Sales Data 2022-20225 (Oct Updated).xlsx")
SHEET_NAME = "Sales Register"  # change if your sheet name differs
DATE_COL = "Invoice Date"

# Expected columns in your file
REGION_COL = "Region"
BRANCH_COL = "Branch"
CITY_COL = "City"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=SHEET_NAME)

    # Basic checks
    required_cols = [DATE_COL, REGION_COL, BRANCH_COL, CITY_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL)

    # Normalize types
    df[REGION_COL] = df[REGION_COL].astype(str)
    df[BRANCH_COL] = df[BRANCH_COL].astype(str)
    df[CITY_COL] = df[CITY_COL].astype(str)

    # Convenience columns (same names)
    df["Region"] = df[REGION_COL]
    df["Branch"] = df[BRANCH_COL]
    df["City"] = df[CITY_COL]

    return df


@st.cache_data
def build_monthly_aggregates(df: pd.DataFrame):
    # Overall monthly units (All MH)
    monthly_overall = (
        df.groupby(pd.Grouper(key=DATE_COL, freq="MS"))
        .size()
        .rename("units")
        .reset_index()
        .rename(columns={DATE_COL: "Month"})
    )

    # Region-wise monthly units
    monthly_region = (
        df.groupby([pd.Grouper(key=DATE_COL, freq="MS"), "Region"])
        .size()
        .rename("units")
        .reset_index()
        .rename(columns={DATE_COL: "Month"})
    )

    # Branch-wise monthly units (Region -> Branch)
    monthly_branch = (
        df.groupby([pd.Grouper(key=DATE_COL, freq="MS"), "Region", "Branch"])
        .size()
        .rename("units")
        .reset_index()
        .rename(columns={DATE_COL: "Month"})
    )

    # City-wise monthly units
    monthly_city = (
        df.groupby([pd.Grouper(key=DATE_COL, freq="MS"), "City"])
        .size()
        .rename("units")
        .reset_index()
        .rename(columns={DATE_COL: "Month"})
    )

    return monthly_overall, monthly_region, monthly_branch, monthly_city


def ensure_full_monthly_index(ts: pd.Series) -> pd.Series:
    """Ensure monthly frequency & fill missing months with zero."""
    ts = ts.sort_index()
    full_idx = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq="MS")
    ts = ts.reindex(full_idx).fillna(0)
    ts.index.name = "Month"
    return ts


def forecast_series(ts: pd.Series, periods: int) -> pd.Series:
    """Forecast monthly sales using Holt-Winters with fallbacks."""
    ts = ensure_full_monthly_index(ts)
    non_zero = ts[ts > 0]

    # Forecast index
    last_month = ts.index.max()
    future_index = pd.date_range(
        start=last_month + pd.offsets.MonthBegin(1),
        periods=periods,
        freq="MS",
    )

    # Fallback logic for short / sparse series
    if non_zero.empty:
        return pd.Series(0, index=future_index, name="forecast")

    if len(ts) < 6:
        last_val = non_zero.iloc[-1]
        return pd.Series(last_val, index=future_index, name="forecast")

    if len(ts) < 18:
        mean_recent = ts.tail(6).mean()
        return pd.Series(mean_recent, index=future_index, name="forecast")

    # Full Holt-Winters model with yearly seasonality
    try:
        model = ExponentialSmoothing(
            ts,
            trend="add",
            seasonal="add",
            seasonal_periods=12,
        )
        fit = model.fit(optimized=True)
        fc = fit.forecast(periods)
        fc.name = "forecast"
        return fc
    except Exception:
        mean_recent = ts.tail(6).mean()
        return pd.Series(mean_recent, index=future_index, name="forecast")


def make_forecast_chart(actual_ts: pd.Series, fc_ts: pd.Series, label: str):
    actual_df = (
        actual_ts.rename("Units")
        .reset_index()
        .rename(columns={"index": "Month"})
    )
    actual_df["Type"] = "Actual"

    fc_df = (
        fc_ts.rename("Units")
        .reset_index()
        .rename(columns={"index": "Month"})
    )
    fc_df["Type"] = "Forecast"

    chart_df = pd.concat([actual_df, fc_df], ignore_index=True)

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Month:T", title="Month"),
            y=alt.Y("Units:Q", title="Units Sold"),
            color=alt.Color("Type:N", title=""),
            tooltip=[
                alt.Tooltip("Month:T", title="Month"),
                alt.Tooltip("Type:N", title="Series"),
                alt.Tooltip("Units:Q", title="Units", format=".0f"),
            ],
        )
        .properties(
            width="container",
            height=420,
            title=f"Actual vs Forecast – {label}",
        )
    )
    return chart


def compute_level_forecasts(
    monthly_df: pd.DataFrame,
    level_col: str,
    periods: int = 3,
    top_n: int | None = None,
):
    """
    Compute forecast for each value in level_col (Region/Branch/City)
    over the next `periods` months. Optionally restrict to top_n by volume.
    """
    totals = (
        monthly_df.groupby(level_col)["units"]
        .sum()
        .sort_values(ascending=False)
    )
    if top_n is not None:
        entities = totals.head(top_n).index
    else:
        entities = totals.index

    records = []
    for name in entities:
        sub = monthly_df[monthly_df[level_col] == name]
        if sub.empty:
            continue
        ts = sub.set_index("Month")["units"]
        if len(ts) == 0:
            continue
        fc = forecast_series(ts, periods=periods)
        for month, val in fc.items():
            records.append(
                {
                    "Month": month,
                    level_col: name,
                    "Forecast Units": float(val),
                }
            )

    if not records:
        return pd.DataFrame(columns=["Month", level_col, "Forecast Units"])

    df_fc = pd.DataFrame(records)
    return df_fc


def make_level_bar_chart(df_fc: pd.DataFrame, level_col: str, title: str):
    """
    Month-wise grouped bar chart for Region / Branch / City forecasts.
    """
    if df_fc.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_text(
            text="No data available",
            color="#E5E7EE",
        ).properties(
            title=f"{title} (no data)",
            height=200,
        )

    # Clean categorical month label like "Nov 2024"
    df_fc = df_fc.copy()
    df_fc["MonthLabel"] = df_fc["Month"].dt.strftime("%b %Y")

    base = alt.Chart(df_fc).encode(
        x=alt.X(
            "MonthLabel:N",
            title="Month",
            axis=alt.Axis(labelAngle=0),
        ),
        xOffset=alt.XOffset(f"{level_col}:N"),
        y=alt.Y(
            "Forecast Units:Q",
            title="Forecast Units",
        ),
        color=alt.Color(
            f"{level_col}:N",
            title=level_col,
            scale=alt.Scale(scheme="tableau10"),
        ),
        tooltip=[
            alt.Tooltip("MonthLabel:N", title="Month"),
            alt.Tooltip(f"{level_col}:N", title=level_col),
            alt.Tooltip("Forecast Units:Q", title="Forecast Units", format=".0f"),
        ],
    )

    bars = base.mark_bar(opacity=0.9)

    text = base.mark_text(
        dy=-6,
        fontSize=11,
        color="#E5E7EE",
    ).encode(
        text=alt.Text("Forecast Units:Q", format=".0f")
    )

    chart = (bars + text).properties(
        width="container",
        height=350,
        title=title,
    )

    return chart


# ============= MAIN APP =============

def main():
    # --- Load data first so hero can show real stats ---
    if not DATA_PATH.exists():
        st.error(f"Data file not found at: {DATA_PATH}")
        st.stop()

    df = load_data(DATA_PATH)
    (
        monthly_overall,
        monthly_region,
        monthly_branch,
        monthly_city,
    ) = build_monthly_aggregates(df)

    # --- High-level coverage stats for the hero header ---
    min_date = df[DATE_COL].min()
    max_date = df[DATE_COL].max()
    total_units = len(df)
    n_regions = df["Region"].nunique()
    n_branches = df["Branch"].nunique()
    n_cities = df["City"].nunique()

    coverage_text = f"{min_date.strftime('%b %Y')} – {max_date.strftime('%b %Y')}"
    footprint_text = f"{n_regions} Regions · {n_branches} Branches · {n_cities} Cities"

    # --- HERO HEADER ---
    st.markdown(
        f"""
        <div class="hero-wrapper">
          <div class="hero-card">
            <div class="hero-accent-pill">
              <span class="hero-accent-pill-dot"></span>
              BI SALES PREDICTOR · MH COMMERCIAL
            </div>
            <div class="hero-main-row">
              <div class="hero-main-text">
                <h1 class="hero-title">BI Sales Predictor</h1>
                <p class="hero-subtitle">Autobahn Trucking – MH</p>
                <p class="hero-caption">
                  Enterprise-grade predictive analytics for MH sales, combining region, branch
                  and city insights into a single, data-driven decisioning hub.
                </p>
              </div>
              <div class="hero-meta">
                <div class="hero-meta-label">Data Coverage</div>
                <div class="hero-meta-value">{coverage_text}</div>
                <div class="hero-meta-chip">
                  <div class="hero-meta-chip-icon">∑</div>
                  <span>{footprint_text}</span>
                </div>
                <div class="hero-meta-label" style="margin-top:6px;">Total Retail Records</div>
                <div class="hero-meta-value">{total_units:,} invoices</div>
              </div>
            </div>
            <div class="hero-geo hero-geo-one"></div>
            <div class="hero-geo hero-geo-two"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar controls (for main detailed chart) ---
    with st.sidebar:
        st.header("⚙️ Controls")

        regions = sorted(df["Region"].dropna().unique())
        region_choice = st.selectbox(
            "Select Region",
            options=["All Regions"] + regions,
            index=0,
        )

        if region_choice == "All Regions":
            branches = sorted(df["Branch"].dropna().unique())
        else:
            branches = sorted(
                df[df["Region"] == region_choice]["Branch"].dropna().unique()
            )

        branch_choice = st.selectbox(
            "Select Branch / View",
            options=["All MH"] + branches,
            index=0,
            help="Choose 'All MH' for overall forecast, or a specific Branch for branch-wise forecast.",
        )

        horizon = st.slider(
            "Forecast horizon (months)",
            min_value=3,
            max_value=18,
            value=6,
            step=1,
        )

    # --- Select time series based on Region & Branch for main chart ---
    if branch_choice == "All MH":
        ts = monthly_overall.set_index("Month")["units"]
        label = "All MH"
    else:
        if region_choice == "All Regions":
            mb = monthly_branch[monthly_branch["Branch"] == branch_choice]
        else:
            mb = monthly_branch[
                (monthly_branch["Region"] == region_choice)
                & (monthly_branch["Branch"] == branch_choice)
            ]

        if mb.empty:
            st.warning(
                f"No data found for branch '{branch_choice}' with region filter '{region_choice}'."
            )
            st.stop()

        ts = mb.set_index("Month")["units"]
        if region_choice == "All Regions":
            label = f"Branch: {branch_choice}"
        else:
            label = f"{branch_choice} ({region_choice})"

    ts = ensure_full_monthly_index(ts)

    # --- Forecast for main view (independent of history slider) ---
    fc = forecast_series(ts, periods=horizon)

    # --- Metrics for KPI cards ---
    last_actual_month = ts.index.max()
    last_actual_value = ts.loc[last_actual_month]

    first_fc_month = fc.index.min()
    first_fc_value = fc.iloc[0]

    one_year_ago = last_actual_month - pd.DateOffset(years=1)
    yoy_value = ts.loc[one_year_ago] if one_year_ago in ts.index else None

    # Prepare KPI text pieces
    last_actual_str = f"{int(last_actual_value):,} units"
    next_fc_str = f"{int(round(first_fc_value)):,} units"

    delta_val = first_fc_value - last_actual_value
    if delta_val > 0:
        delta_class = "kpi-pill kpi-pill-pos"
        delta_text = f"↑ {delta_val:.0f} vs last month"
    elif delta_val < 0:
        delta_class = "kpi-pill kpi-pill-neg"
        delta_text = f"↓ {abs(delta_val):.0f} vs last month"
    else:
        delta_class = "kpi-pill kpi-pill-neutral"
        delta_text = "No change vs last month"

    if yoy_value is not None and yoy_value != 0:
        yoy_change = (last_actual_value - yoy_value) / yoy_value * 100
        yoy_str = f"{yoy_change:+.1f} %"
        if yoy_change > 0:
            yoy_class = "kpi-pill kpi-pill-pos"
            yoy_delta_text = f"↑ From {int(yoy_value):,} units last year"
        elif yoy_change < 0:
            yoy_class = "kpi-pill kpi-pill-neg"
            yoy_delta_text = f"↓ From {int(yoy_value):,} units last year"
        else:
            yoy_class = "kpi-pill kpi-pill-neutral"
            yoy_delta_text = f"Same as {int(yoy_value):,} units last year"
    else:
        yoy_str = "N/A"
        yoy_class = "kpi-pill kpi-pill-neutral"
        yoy_delta_text = "Not enough data"

    # --- KPI row (three neat cards) ---
    kpi_html = f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label">Last Actual</div>
        <div class="kpi-sub">{last_actual_month.strftime('%b %Y')}</div>
        <div class="kpi-value">{last_actual_str}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Next Month Forecast</div>
        <div class="kpi-sub">{first_fc_month.strftime('%b %Y')}</div>
        <div class="kpi-value">{next_fc_str}</div>
        <div class="{delta_class}">
          {delta_text}
        </div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">YoY Growth</div>
        <div class="kpi-sub">vs same month last year</div>
        <div class="kpi-value">{yoy_str}</div>
        <div class="{yoy_class}">
          {yoy_delta_text}
        </div>
      </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # --- Slider for history window (directly above chart) ---
    min_month = ts.index.min()
    max_month = ts.index.max()
    total_years = max(1, int(np.ceil(max_month.year - min_month.year + 1)))

    history_years = st.slider(
        "History window (years)",
        min_value=1,
        max_value=total_years,
        value=min(3, total_years),
        help="Controls how many years of past actuals are visible in the chart below.",
    )

    # Limit history shown according to slider
    last_month = ts.index.max()
    cutoff = last_month - pd.DateOffset(years=history_years) + pd.offsets.MonthBegin(0)
    ts_display = ts[ts.index >= cutoff]

    # --- Main detailed chart ---
    chart = make_forecast_chart(ts_display, fc, label)
    st.altair_chart(chart, use_container_width=True)

    # --- Data table (Actual + Forecast) ---
    with st.expander("📄 View Actual + Forecast Data for Selected View"):
        hist_df = (
            ts_display.rename("Actual Units")
            .reset_index()
            .rename(columns={"index": "Month"})
        )

        fc_df = (
            fc.rename("Forecast Units")
            .reset_index()
            .rename(columns={"index": "Month"})
        )

        combined = (
            pd.merge(hist_df, fc_df, how="outer", on="Month")
            .sort_values("Month")
        )

        st.dataframe(combined, use_container_width=True)

    # ============= GLOBAL 3-MONTH FORECAST SUMMARY (Region / Branch / City) =============

    st.markdown(
        '<div class="section-title">3-Month Forecast Summary – Region, Branch & City</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Forecast snapshot for the **next 3 months** across key levels, for quick C-suite review."
    )

    REGION_TOP_N = None        # all regions
    BRANCH_TOP_N = 10          # top 10 branches
    CITY_TOP_N = 10            # top 10 cities

    region_fc = compute_level_forecasts(
        monthly_region,
        level_col="Region",
        periods=3,
        top_n=REGION_TOP_N,
    )

    branch_fc = compute_level_forecasts(
        monthly_branch,
        level_col="Branch",
        periods=3,
        top_n=BRANCH_TOP_N,
    )

    city_fc = compute_level_forecasts(
        monthly_city,
        level_col="City",
        periods=3,
        top_n=CITY_TOP_N,
    )

    # Region chart
    region_chart = make_level_bar_chart(
        region_fc,
        level_col="Region",
        title="Region-wise Forecast – Next 3 Months",
    )
    st.altair_chart(region_chart, use_container_width=True)

    # Branch chart
    branch_chart = make_level_bar_chart(
        branch_fc,
        level_col="Branch",
        title=f"Branch-wise Forecast – Next 3 Months (Top {BRANCH_TOP_N})",
    )
    st.altair_chart(branch_chart, use_container_width=True)

    # City chart
    city_chart = make_level_bar_chart(
        city_fc,
        level_col="City",
        title=f"City-wise Forecast – Next 3 Months (Top {CITY_TOP_N})",
    )
    st.altair_chart(city_chart, use_container_width=True)


if __name__ == "__main__":
    main()
