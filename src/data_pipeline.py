from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

STANDARD_COLUMNS = [
    "Date",
    "Year",
    "Month",
    "Country",
    "Area",
    "Location",
    "Activity",
    "Fatal",
    "Species",
    "Sex",
    "Age",
]

COLUMN_ALIASES = {
    "fatal (y/n)": "Fatal",
    "fatal y/n": "Fatal",
    "fatal": "Fatal",
    "species": "Species",
    "species name": "Species",
    "country": "Country",
    "area": "Area",
    "location": "Location",
    "activity": "Activity",
    "date": "Date",
    "year": "Year",
    "month": "Month",
    "sex": "Sex",
    "gender": "Sex",
    "age": "Age",
}

ACTIVITY_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("Surfing", ("surf", "bodyboard", "boogie board", "paddleboard")),
    ("Swimming", ("swim", "bathing", "snorkel", "float")),
    ("Diving", ("diving", "dive", "scuba", "spearfish")),
    ("Fishing", ("fish", "angling", "netting")),
    ("Wading", ("wading", "standing", "walking in water")),
    ("Boating", ("boat", "kayak", "canoe", "sailing", "rowing", "paddling", "outrigger", "raft")),
]

SPECIES_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("Great White Shark", ("white shark", "great white", "carcharodon")),
    ("Tiger Shark", ("tiger shark", "galeocerdo")),
    ("Bull Shark", ("bull shark", "carcharhinus leucas")),
    ("Blacktip Shark", ("blacktip", "carcharhinus limbatus")),
    ("Caribbean Reef Shark", ("caribbean reef", "reef shark")),
    ("Great Hammerhead Shark", ("hammerhead", "sphyrna")),
    ("Shortfin Mako Shark", ("mako", "isurus")),
    ("Blue Shark", ("blue shark", "prionace")),
    ("Lemon Shark", ("lemon shark", "negaprion")),
    ("Nurse Shark", ("nurse shark", "ginglymostoma")),
    ("Oceanic Whitetip Shark", ("oceanic whitetip", "longimanus")),
    ("Whitetip Reef Shark", ("whitetip reef", "triaenodon")),
]

COUNTRY_ALIASES = {
    "usa": "United States",
    "u.s.a.": "United States",
    "u.s.a": "United States",
    "us": "United States",
    "u.s.": "United States",
    "uk": "United Kingdom",
    "u.k.": "United Kingdom",
    "uae": "United Arab Emirates",
}

MONTH_LOOKUP = {
    **{str(index): name for index, name in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        start=1,
    )},
    **{f"{index:02d}": name for index, name in enumerate(
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        start=1,
    )},
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "sept": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}


def _normalized_column_name(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.columns.duplicated().any():
        return df
    combined: dict[str, pd.Series] = {}
    for name in dict.fromkeys(df.columns):
        matching = df.loc[:, df.columns == name]
        if matching.shape[1] == 1:
            combined[name] = matching.iloc[:, 0]
        else:
            combined[name] = matching.bfill(axis=1).iloc[:, 0]
    return pd.DataFrame(combined, index=df.index)


def clean_activity(value: object) -> str:
    text = _safe_text(value).lower()
    if not text or text in {"nan", "unknown", "n/a", "<na>"}:
        return "Unknown"
    for label, patterns in ACTIVITY_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return label
    return "Other"


def clean_species(value: object) -> str:
    text = _safe_text(value).lower()
    if not text or text in {
        "nan", "unknown", "n/a", "<na>",
        "shark involvement prior to death was not confirmed",
    }:
        return "Unknown Shark"
    for label, patterns in SPECIES_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return label
    if "shark" in text:
        return "Other Shark"
    return "Unknown Shark"


def clean_fatal(value: object) -> str:
    text = _safe_text(value).upper()
    if text in {"Y", "YES", "TRUE", "1", "FATAL"}:
        return "Yes"
    if text in {"N", "NO", "FALSE", "0", "NON-FATAL", "NONFATAL"}:
        return "No"
    return "Unknown"


def clean_sex(value: object) -> str:
    text = _safe_text(value).upper()
    if text in {"M", "MALE"}:
        return "Male"
    if text in {"F", "FEMALE"}:
        return "Female"
    return "Unknown"


def clean_age(value: object) -> float | None:
    if pd.isna(value):
        return None
    match = re.search(r"\d{1,3}", str(value))
    if not match:
        return None
    age = int(match.group())
    return float(age) if 0 < age < 110 else None


def clean_country(value: object) -> str:
    text = _safe_text(value)
    if not text or text.lower() in {"nan", "unknown", "n/a", "<na>"}:
        return "Unknown"
    compact = re.sub(r"\s+", " ", text).strip()
    alias = COUNTRY_ALIASES.get(compact.lower())
    if alias:
        return alias
    return compact.title()


def _title_or_unknown(value: object) -> str:
    text = _safe_text(value)
    if not text or text.lower() in {"nan", "unknown", "n/a", "<na>"}:
        return "Unknown"
    return re.sub(r"\s+", " ", text).title()


def _clean_month(value: object) -> str:
    text = _safe_text(value).lower().rstrip(".")
    if not text or text in {"nan", "unknown", "n/a", "<na>"}:
        return "Unknown"
    return MONTH_LOOKUP.get(text, text.title())


def _parse_dates(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def standardize_attacks(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a shark-incident DataFrame into dashboard-ready columns."""
    if raw.empty:
        raise ValueError("The CSV does not contain any data rows.")

    df = raw.copy()
    df.columns = [str(col).strip() for col in df.columns]

    rename_map: dict[str, str] = {}
    recognized: set[str] = set()
    for col in df.columns:
        normalized = _normalized_column_name(col)
        target = COLUMN_ALIASES.get(normalized)
        if target:
            rename_map[col] = target
            recognized.add(target)

    if len(recognized) < 2:
        raise ValueError(
            "The file does not look like a shark-incident dataset. Include at least two recognized "
            "columns such as Date, Year, Country, Activity, Species, Fatal (Y/N), Sex, or Age."
        )

    df = df.rename(columns=rename_map)
    df = _coalesce_duplicate_columns(df)

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Many spreadsheet exports contain thousands of completely blank footer rows.
    # Keep a row only when at least one analytical field contains a meaningful value.
    meaningful_row = pd.Series(False, index=df.index)
    empty_tokens = {"", "nan", "<na>", "n/a", "none"}
    for col in STANDARD_COLUMNS:
        values = df[col].astype("string").str.strip()
        meaningful_row = meaningful_row | (
            values.notna() & ~values.str.lower().isin(empty_tokens)
        )
    df = df.loc[meaningful_row].copy()

    if df.empty:
        raise ValueError("The CSV does not contain any usable shark-incident records.")

    parsed_dates = _parse_dates(df["Date"])
    numeric_year = pd.to_numeric(df["Year"], errors="coerce")
    df["Year"] = numeric_year.fillna(parsed_dates.dt.year).astype("Int64")

    month_from_input = df["Month"].map(_clean_month)
    month_from_date = parsed_dates.dt.month_name()
    df["Month"] = month_from_input.mask(month_from_input.eq("Unknown"), month_from_date)
    df["Month"] = df["Month"].fillna("Unknown")

    df["Date"] = parsed_dates
    df["Country"] = df["Country"].map(clean_country)
    df["Area"] = df["Area"].map(_title_or_unknown)
    df["Location"] = df["Location"].map(_title_or_unknown)
    df["Activity"] = df["Activity"].map(clean_activity)
    df["Fatal"] = df["Fatal"].map(clean_fatal)
    df["Species"] = df["Species"].map(clean_species)
    df["Sex"] = df["Sex"].map(clean_sex)
    df["Age"] = df["Age"].map(clean_age).astype("Float64")

    df.loc[(df["Year"] < 1500) | (df["Year"] > 2100), "Year"] = pd.NA

    ordered = STANDARD_COLUMNS + [col for col in df.columns if col not in STANDARD_COLUMNS]
    return df[ordered].reset_index(drop=True)


def load_csv(source: str | Path | object) -> pd.DataFrame:
    """Read and standardize a CSV path or Streamlit UploadedFile."""
    try:
        raw = pd.read_csv(source, encoding="utf-8-sig", encoding_errors="replace", low_memory=False)
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        raw = pd.read_csv(source, encoding="latin-1", low_memory=False)
    except pd.errors.ParserError as exc:
        raise ValueError(f"The CSV structure could not be parsed: {exc}") from exc
    return standardize_attacks(raw)


def filter_attacks(
    df: pd.DataFrame,
    countries: list[str] | None = None,
    species: list[str] | None = None,
    activities: list[str] | None = None,
    year_range: tuple[int, int] | None = None,
    fatal_options: list[str] | None = None,
    include_unknown_years: bool = True,
) -> pd.DataFrame:
    filtered = df.copy()
    if countries:
        filtered = filtered[filtered["Country"].isin(countries)]
    if species:
        filtered = filtered[filtered["Species"].isin(species)]
    if activities:
        filtered = filtered[filtered["Activity"].isin(activities)]
    if fatal_options:
        filtered = filtered[filtered["Fatal"].isin(fatal_options)]
    if year_range:
        start, end = year_range
        year_mask = filtered["Year"].between(start, end, inclusive="both")
        if include_unknown_years:
            year_mask = year_mask | filtered["Year"].isna()
        filtered = filtered[year_mask]
    return filtered.reset_index(drop=True)


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = max(len(df), 1)
    unknown_tokens = {"unknown", "unknown shark", "<na>", "nan", ""}
    for column in STANDARD_COLUMNS:
        values = df[column]
        unknown = values.isna()
        if values.dtype == "object" or str(values.dtype).startswith("string"):
            unknown = unknown | values.astype("string").str.strip().str.lower().isin(unknown_tokens)
        missing_count = int(unknown.sum())
        rows.append(
            {
                "Column": column,
                "Complete records": int(len(df) - missing_count),
                "Missing or unknown": missing_count,
                "Completeness": round((len(df) - missing_count) / total * 100, 1),
            }
        )
    return pd.DataFrame(rows)
