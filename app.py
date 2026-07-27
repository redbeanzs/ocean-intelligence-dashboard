from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import data_quality_summary, filter_attacks, load_csv
from src.gbif_api import match_species

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REAL_DATA_PATH = DATA_DIR / "attacks.csv"
SAMPLE_DATA_PATH = DATA_DIR / "sample_attacks.csv"
MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December", "Unknown",
]
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

st.set_page_config(
    page_title="Ocean Intelligence Dashboard",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Ocean Intelligence Dashboard by Stina Drill. Built with Python, Pandas, "
            "Plotly, Streamlit, and the GBIF Species API."
        )
    },
)

st.markdown(
    """
    <style>
      :root {
        --ocean-dark: #062b3b;
        --ocean: #087e8b;
        --ocean-bright: #159ca4;
        --foam: #e8fbfb;
        --coral: #e85d5d;
        --ink: #0d3542;
        --muted: #365f69;
        --line: #b9dfe1;
      }

      html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #f7ffff !important;
        color: var(--ink) !important;
      }

      [data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
      }

      [data-testid="stAppViewContainer"] .stMarkdown,
      [data-testid="stAppViewContainer"] .stMarkdown p,
      [data-testid="stAppViewContainer"] .stMarkdown li,
      [data-testid="stAppViewContainer"] label,
      [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
      [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
      [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p {
        color: var(--ink) !important;
        opacity: 1 !important;
      }

      h1, h2, h3 { color: var(--ocean-dark) !important; }

      a, button, input, [tabindex="0"] {
        outline-color: var(--ocean-bright) !important;
      }

      section[data-testid="stSidebar"],
      section[data-testid="stSidebar"] > div,
      [data-testid="stSidebarContent"] {
        background: var(--ocean-dark) !important;
      }

      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3,
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] span,
      section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #ffffff !important;
        opacity: 1 !important;
      }

      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: #ffffff !important;
        border: 2px dashed #55c5c7 !important;
        border-radius: 14px !important;
      }

      section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
      section[data-testid="stSidebar"] input,
      section[data-testid="stSidebar"] [data-baseweb="select"] *,
      section[data-testid="stSidebar"] [data-baseweb="tag"] * {
        color: var(--ocean-dark) !important;
      }

      section[data-testid="stSidebar"] input,
      section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff !important;
      }

      section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] {
        background: #ffffff !important;
        border-radius: 12px !important;
      }

      section[data-testid="stSidebar"] [data-testid="stFileUploaderFile"] * {
        color: var(--ocean-dark) !important;
      }

      section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.28) !important;
      }

      section[data-testid="stSidebar"] button[kind="secondary"] {
        background: #ffffff !important;
        color: var(--ocean-dark) !important;
        border: 1px solid #7ed0d2 !important;
        font-weight: 800 !important;
      }

      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 0.35rem;
        overflow-x: auto;
        scrollbar-width: thin;
      }

      button[data-baseweb="tab"] {
        background: #e4f7f7 !important;
        border-radius: 12px 12px 0 0 !important;
        white-space: nowrap !important;
      }

      button[data-baseweb="tab"] p {
        color: var(--ocean-dark) !important;
        font-weight: 800 !important;
        opacity: 1 !important;
      }

      button[data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff !important;
        border-bottom: 4px solid var(--ocean) !important;
      }

      [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 8px 24px rgba(6, 43, 59, 0.08);
      }

      [data-testid="stMetric"] *,
      [data-testid="stMetricLabel"] p,
      [data-testid="stMetricValue"] {
        color: var(--ocean-dark) !important;
        opacity: 1 !important;
      }

      [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.35rem;
        box-shadow: 0 8px 24px rgba(6, 43, 59, 0.06);
        overflow: hidden;
      }

      .hero {
        padding: clamp(22px, 4vw, 40px);
        border-radius: 26px;
        color: #ffffff !important;
        background: linear-gradient(120deg, #062b3b 0%, #087e8b 66%, #35b7b5 100%);
        box-shadow: 0 18px 45px rgba(6, 43, 59, 0.18);
        margin-bottom: 18px;
      }

      .hero h1, .hero p {
        color: #ffffff !important;
        opacity: 1 !important;
      }

      .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 5vw, 4.25rem);
        line-height: 1.05;
      }

      .hero p {
        margin: 12px 0 0;
        max-width: 900px;
        font-size: clamp(1rem, 1.8vw, 1.2rem);
      }

      .source-note {
        padding: 16px 18px;
        border-radius: 14px;
        background: #dff8f8;
        border: 1px solid #8fd4d7;
        border-left: 6px solid var(--ocean);
        color: var(--ocean-dark) !important;
        font-size: 1rem;
        line-height: 1.6;
        margin: 10px 0 22px;
        box-shadow: 0 8px 24px rgba(6, 43, 59, 0.06);
      }

      .source-note, .source-note strong {
        color: var(--ocean-dark) !important;
        opacity: 1 !important;
      }

      .species-card {
        background: #ffffff;
        border: 1px solid var(--line);
        padding: 18px;
        border-radius: 18px;
        min-height: 175px;
        box-shadow: 0 8px 24px rgba(6, 43, 59, 0.06);
      }

      .species-card, .species-card p, .species-card h3 {
        color: var(--ocean-dark) !important;
        opacity: 1 !important;
      }

      .species-card h3 { margin-top: 0; }
      .small-muted { color: var(--muted) !important; font-size: .9rem; opacity: 1 !important; }

      div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 14px;
        overflow: hidden;
      }

      [data-testid="stDownloadButton"] button,
      [data-testid="stLinkButton"] a {
        background: var(--ocean) !important;
        color: #ffffff !important;
        border: 0 !important;
        font-weight: 800 !important;
      }

      [data-testid="stDownloadButton"] button *,
      [data-testid="stLinkButton"] a * {
        color: #ffffff !important;
      }

      [data-testid="stAlert"] *, details summary, details p {
        color: var(--ocean-dark) !important;
        opacity: 1 !important;
      }

      .footer-note {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--line);
        color: var(--muted) !important;
        font-size: 0.9rem;
      }

      @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding-left: 0.75rem; padding-right: 0.75rem; }
        .hero { border-radius: 18px; }
        [data-testid="stMetric"] { padding: 10px 12px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_path(path: str) -> pd.DataFrame:
    return load_csv(Path(path))


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_gbif_lookup(scientific_name: str) -> dict:
    return match_species(scientific_name)


@st.cache_data(show_spinner=False)
def load_species_reference() -> dict:
    return json.loads((DATA_DIR / "species_reference.json").read_text(encoding="utf-8"))


def ocean_layout(fig: go.Figure, title: str, *, height: int = 430) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=20, color="#062b3b")),
        height=height,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=24, r=20, t=68, b=42),
        legend_title_text="",
        font=dict(color="#153d49", size=13),
        hoverlabel=dict(bgcolor="#062b3b", font_color="#ffffff"),
    )
    fig.update_xaxes(
        gridcolor="#deeeee",
        linecolor="#a8cfd2",
        tickfont=dict(color="#153d49"),
        title_font=dict(color="#153d49"),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="#deeeee",
        linecolor="#a8cfd2",
        tickfont=dict(color="#153d49"),
        title_font=dict(color="#153d49"),
        automargin=True,
    )
    return fig


def show_chart(container, fig: go.Figure, key: str) -> None:
    container.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config=PLOTLY_CONFIG,
        key=key,
    )


def meaningful_counts(series: pd.Series, *, limit: int | None = None) -> pd.DataFrame:
    cleaned = series.dropna().astype(str)
    preferred = cleaned[~cleaned.str.lower().isin({"unknown", "unknown shark"})]
    if preferred.empty:
        preferred = cleaned
    counts = preferred.value_counts()
    if limit:
        counts = counts.head(limit)
    frame = counts.rename_axis("Label").reset_index(name="Incidents")
    return frame


st.markdown(
    """
    <div class="hero">
      <h1>Ocean Intelligence Dashboard</h1>
      <p>Explore recorded shark incident patterns by species, country, year, activity, and outcome. The project combines data cleaning, interactive visualization, and live species taxonomy enrichment.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dashboard controls")
    uploaded = st.file_uploader(
        "Upload an attacks CSV",
        type=["csv"],
        help="Common Global Shark Attack File-style columns are standardized automatically.",
    )
    st.caption("Your uploaded file is processed only for the current app session.")

if uploaded is not None:
    try:
        attacks = load_csv(uploaded)
        source_label = f"Uploaded file: {uploaded.name}"
        source_kind = "uploaded"
    except Exception as exc:
        st.error(f"The uploaded file could not be processed: {exc}")
        fallback = REAL_DATA_PATH if REAL_DATA_PATH.exists() else SAMPLE_DATA_PATH
        attacks = load_path(str(fallback))
        source_label = "Bundled project dataset" if REAL_DATA_PATH.exists() else "Bundled synthetic demo dataset"
        source_kind = "project" if REAL_DATA_PATH.exists() else "demo"
elif REAL_DATA_PATH.exists():
    attacks = load_path(str(REAL_DATA_PATH))
    source_label = "Bundled project dataset: attacks.csv"
    source_kind = "project"
else:
    attacks = load_path(str(SAMPLE_DATA_PATH))
    source_label = "Bundled synthetic demo dataset"
    source_kind = "demo"

valid_years = attacks["Year"].dropna().astype(int)
min_year = int(valid_years.min()) if not valid_years.empty else 1900
max_year = int(valid_years.max()) if not valid_years.empty else 2025

with st.sidebar:
    if min_year < max_year:
        year_range = st.slider(
            "Year range",
            min_year,
            max_year,
            (min_year, max_year),
            key="year_filter",
        )
    else:
        year_range = (min_year, max_year)
        st.caption(f"All dated records are from {min_year}.")

    include_unknown_years = st.toggle(
        "Include records with unknown years",
        value=True,
        help="Keeps incomplete records visible instead of silently dropping them.",
    )
    countries = st.multiselect(
        "Country",
        sorted(attacks["Country"].dropna().unique()),
        placeholder="All countries",
    )
    species = st.multiselect(
        "Species group",
        sorted(attacks["Species"].dropna().unique()),
        placeholder="All species groups",
    )
    activities = st.multiselect(
        "Activity",
        sorted(attacks["Activity"].dropna().unique()),
        placeholder="All activities",
    )
    fatal_options = st.multiselect(
        "Outcome",
        ["No", "Yes", "Unknown"],
        placeholder="All outcomes",
    )
    st.divider()
    st.caption("Built with Python, Pandas, Plotly, Streamlit, and the GBIF Species API.")

filtered = filter_attacks(
    attacks,
    countries=countries,
    species=species,
    activities=activities,
    year_range=year_range,
    fatal_options=fatal_options,
    include_unknown_years=include_unknown_years,
)

if source_kind == "demo":
    note = (
        "The included dataset is synthetic so the repository runs immediately. Add your real file "
        "as data/attacks.csv to make it the default public dataset, or upload a CSV for this session."
    )
elif source_kind == "project":
    note = "The dashboard is using the real project dataset bundled with the deployed application. Blank spreadsheet footer rows are removed during cleaning."
else:
    note = "The dashboard is using your uploaded file for this browser session."

st.markdown(
    f'<div class="source-note"><strong>Data source:</strong> {html.escape(source_label)}<br>{html.escape(note)}</div>',
    unsafe_allow_html=True,
)

known_years = attacks["Year"].dropna()
coverage_text = (
    f"{int(known_years.min())}–{int(known_years.max())}"
    if not known_years.empty
    else "Unknown"
)
st.caption(
    f"Dataset after cleaning: {len(attacks):,} incidents · "
    f"{attacks['Country'].replace('Unknown', pd.NA).nunique():,} countries · "
    f"dated coverage {coverage_text}"
)

metric_cols = st.columns(5)
metric_cols[0].metric("Filtered incidents", f"{len(filtered):,}")
metric_cols[1].metric("Countries", f"{filtered['Country'].replace('Unknown', pd.NA).nunique():,}")
metric_cols[2].metric("Species groups", f"{filtered['Species'].replace('Unknown Shark', pd.NA).nunique():,}")
fatal_known = filtered[filtered["Fatal"].isin(["Yes", "No"])]
fatal_rate = (fatal_known["Fatal"].eq("Yes").mean() * 100) if len(fatal_known) else None
metric_cols[3].metric("Fatality rate", "Unknown" if fatal_rate is None else f"{fatal_rate:.1f}%")
median_age = filtered["Age"].median()
metric_cols[4].metric("Median age", "Unknown" if pd.isna(median_age) else f"{median_age:.0f}")

overview_tab, species_tab, quality_tab, data_tab = st.tabs(
    ["Overview", "Species intelligence", "Data quality", "Data table"]
)

with overview_tab:
    if filtered.empty:
        st.warning("No records match the current filters. Clear one or more sidebar selections.")
    else:
        annual = (
            filtered.dropna(subset=["Year"])
            .groupby("Year", as_index=False)
            .size()
            .rename(columns={"size": "Incidents"})
        )
        if annual.empty:
            st.info("The filtered records do not contain usable years for a time-series chart.")
        else:
            line = px.line(
                annual,
                x="Year",
                y="Incidents",
                markers=True,
                color_discrete_sequence=["#087e8b"],
            )
            line.update_traces(line_width=3, marker_size=7, hovertemplate="Year %{x}<br>%{y:,} incidents<extra></extra>")
            show_chart(st, ocean_layout(line, "Incidents over time"), "annual-incidents")

        col1, col2 = st.columns(2)
        country_counts = meaningful_counts(filtered["Country"], limit=12).sort_values("Incidents")
        country_counts = country_counts.rename(columns={"Label": "Country"})
        country_fig = px.bar(
            country_counts,
            x="Incidents",
            y="Country",
            orientation="h",
            text="Incidents",
            color="Incidents",
            color_continuous_scale="Teal",
        )
        country_fig.update_traces(textposition="outside", hovertemplate="%{y}<br>%{x:,} incidents<extra></extra>")
        country_fig.update_layout(coloraxis_showscale=False)
        show_chart(col1, ocean_layout(country_fig, "Top countries"), "top-countries")

        activity_counts = meaningful_counts(filtered["Activity"]).rename(columns={"Label": "Activity"})
        activity_fig = px.bar(
            activity_counts,
            x="Activity",
            y="Incidents",
            text="Incidents",
            color="Activity",
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        activity_fig.update_traces(textposition="outside", hovertemplate="%{x}<br>%{y:,} incidents<extra></extra>")
        activity_fig.update_xaxes(tickangle=-25)
        show_chart(col2, ocean_layout(activity_fig, "Incidents by activity"), "activity-counts")

        col3, col4 = st.columns(2)
        species_counts = meaningful_counts(filtered["Species"], limit=10).sort_values("Incidents")
        species_counts = species_counts.rename(columns={"Label": "Species"})
        species_fig = px.bar(
            species_counts,
            x="Incidents",
            y="Species",
            orientation="h",
            text="Incidents",
            color="Incidents",
            color_continuous_scale="Blues",
        )
        species_fig.update_traces(textposition="outside", hovertemplate="%{y}<br>%{x:,} incidents<extra></extra>")
        species_fig.update_layout(coloraxis_showscale=False)
        show_chart(col3, ocean_layout(species_fig, "Most reported species groups"), "species-counts")

        outcomes = filtered["Fatal"].value_counts().reindex(["No", "Yes", "Unknown"], fill_value=0)
        donut = go.Figure(
            data=[
                go.Pie(
                    labels=outcomes.index,
                    values=outcomes.values,
                    hole=0.58,
                    marker=dict(colors=["#35b7b5", "#e85d5d", "#9baeb3"]),
                    textinfo="label+percent",
                    hovertemplate="%{label}<br>%{value:,} incidents (%{percent})<extra></extra>",
                )
            ]
        )
        show_chart(col4, ocean_layout(donut, "Recorded outcomes"), "outcome-donut")

        month_data = filtered[["Month", "Activity"]].copy()
        month_data["Month"] = pd.Categorical(month_data["Month"], MONTH_ORDER, ordered=True)
        month_counts = month_data.groupby(["Activity", "Month"], observed=True).size().unstack(fill_value=0)
        month_counts = month_counts.reindex(columns=MONTH_ORDER, fill_value=0)
        heat = go.Figure(
            data=go.Heatmap(
                z=month_counts.values,
                x=month_counts.columns.tolist(),
                y=month_counts.index.tolist(),
                colorscale="Teal",
                colorbar=dict(title="Incidents"),
                hovertemplate="%{y}<br>%{x}: %{z:,} incidents<extra></extra>",
            )
        )
        show_chart(st, ocean_layout(heat, "Seasonal pattern by activity", height=470), "season-heatmap")

with species_tab:
    reference = load_species_reference()
    filtered_species = sorted(
        value for value in filtered["Species"].dropna().unique() if value not in {"Unknown Shark"}
    )
    available_species = filtered_species or sorted(reference)

    if not available_species:
        st.info("No species categories are available in the current data.")
    else:
        selected_species = st.selectbox("Select a species group", available_species)
        local_info = reference.get(selected_species, {})
        scientific_name = local_info.get("scientific_name", "")
        selected_rows = filtered[filtered["Species"] == selected_species]
        common_activity = (
            selected_rows["Activity"].mode().iloc[0]
            if not selected_rows.empty and not selected_rows["Activity"].mode().empty
            else "Unknown"
        )

        left, right = st.columns([1, 1.4])
        with left:
            st.markdown(
                f"""
                <div class="species-card">
                  <h3>{html.escape(str(selected_species))}</h3>
                  <p><strong>Scientific name:</strong> <em>{html.escape(scientific_name or 'Not mapped')}</em></p>
                  <p><strong>Incidents in filtered data:</strong> {len(selected_rows):,}</p>
                  <p><strong>Most common activity:</strong> {html.escape(str(common_activity))}</p>
                  <p class="small-muted">{html.escape(local_info.get('note', 'No bundled description is available.'))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        taxonomy = cached_gbif_lookup(scientific_name) if scientific_name else {"ok": False}
        with right:
            st.subheader("GBIF taxonomy match")
            if taxonomy.get("ok"):
                fields = {
                    "Scientific name": taxonomy.get("scientific_name"),
                    "Taxonomic status": taxonomy.get("status"),
                    "Rank": taxonomy.get("rank"),
                    "Class": taxonomy.get("class"),
                    "Order": taxonomy.get("order"),
                    "Family": taxonomy.get("family"),
                    "Genus": taxonomy.get("genus"),
                    "Match type": taxonomy.get("match_type"),
                    "Confidence": taxonomy.get("confidence"),
                }
                taxonomy_df = pd.DataFrame(
                    [(label, value if value not in (None, "") else "Unknown") for label, value in fields.items()],
                    columns=["Field", "GBIF result"],
                )
                st.dataframe(taxonomy_df, hide_index=True, width="stretch")
            else:
                st.info("The live taxonomy lookup is unavailable, so the bundled species mapping is being used.")

        species_year = (
            selected_rows.dropna(subset=["Year"])
            .groupby("Year", as_index=False)
            .size()
            .rename(columns={"size": "Incidents"})
        )
        if species_year.empty:
            st.info("No dated records are available for this species group under the current filters.")
        else:
            species_line = px.area(
                species_year,
                x="Year",
                y="Incidents",
                color_discrete_sequence=["#087e8b"],
            )
            species_line.update_traces(hovertemplate="Year %{x}<br>%{y:,} incidents<extra></extra>")
            show_chart(
                st,
                ocean_layout(species_line, f"{selected_species} incidents over time"),
                "species-time-series",
            )

with quality_tab:
    quality = data_quality_summary(attacks)
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Source rows", f"{len(attacks):,}")
    q2.metric("Duplicate rows", f"{attacks.duplicated().sum():,}")
    q3.metric("Valid dates", f"{attacks['Date'].notna().sum():,}")
    q4.metric("Known species", f"{attacks['Species'].ne('Unknown Shark').sum():,}")

    st.subheader("Completeness by field")
    quality_fig = px.bar(
        quality.sort_values("Completeness"),
        x="Completeness",
        y="Column",
        orientation="h",
        text="Completeness",
        color="Completeness",
        color_continuous_scale="Teal",
        range_x=[0, 105],
    )
    quality_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    quality_fig.update_layout(coloraxis_showscale=False)
    show_chart(st, ocean_layout(quality_fig, "Data completeness", height=500), "data-completeness")
    st.dataframe(quality, hide_index=True, width="stretch")

    st.subheader("Cleaning performed")
    st.markdown(
        """
        - Standardizes common column names, including `Fatal (Y/N)` and extra spaces in `Species`.
        - Coalesces duplicate versions of recognized columns instead of crashing.
        - Parses mixed date formats and derives year and month when possible.
        - Groups similar activities, normalizes fatal outcomes, and standardizes common country names.
        - Converts inconsistent species descriptions into reusable species categories.
        - Preserves unknown values so missingness remains visible instead of being silently removed.
        """
    )
    st.download_button(
        "Download cleaned full dataset",
        data=attacks.to_csv(index=False).encode("utf-8"),
        file_name="ocean_intelligence_cleaned.csv",
        mime="text/csv",
        width="stretch",
    )

with data_tab:
    st.subheader("Filtered records")
    search_text = st.text_input(
        "Search visible records",
        placeholder="Try a country, activity, location, or species...",
    )
    display = filtered.copy()
    if search_text.strip():
        searchable_columns = ["Country", "Area", "Location", "Activity", "Species", "Fatal"]
        search_mask = pd.Series(False, index=display.index)
        for column in searchable_columns:
            search_mask = search_mask | display[column].astype("string").str.contains(
                search_text.strip(), case=False, na=False, regex=False
            )
        display = display[search_mask]
        st.caption(f"{len(display):,} records match the table search.")

    visible_columns = [
        "Date", "Year", "Month", "Country", "Area", "Location",
        "Activity", "Species", "Fatal", "Sex", "Age",
    ]
    display = display[[column for column in visible_columns if column in display.columns]].copy()
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(display, hide_index=True, width="stretch", height=520)
    st.download_button(
        "Download visible CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="ocean_intelligence_filtered.csv",
        mime="text/csv",
        width="stretch",
    )

st.markdown(
    """
    <div class="footer-note">
      This dashboard describes patterns in recorded incidents. It should not be interpreted as a measure of inherent danger, population size, or species behavior. Records may reflect reporting and documentation differences across places and time periods.
    </div>
    """,
    unsafe_allow_html=True,
)
