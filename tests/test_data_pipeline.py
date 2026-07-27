import pandas as pd
import pytest

from src.data_pipeline import (
    clean_activity,
    clean_country,
    clean_fatal,
    clean_species,
    filter_attacks,
    standardize_attacks,
)


def test_clean_activity_groups_common_phrases():
    assert clean_activity("Surfing near shore") == "Surfing"
    assert clean_activity("Scuba diving") == "Diving"
    assert clean_activity("Kayaking") == "Boating"


def test_clean_outcome_species_and_country():
    assert clean_fatal("Y") == "Yes"
    assert clean_fatal("N") == "No"
    assert clean_species("5 m great white shark") == "Great White Shark"
    assert clean_country("USA") == "United States"
    assert clean_country("UK") == "United Kingdom"


def test_standardize_common_gsaf_columns():
    raw = pd.DataFrame(
        {
            "Date": ["2020-07-04"],
            "Country": ["USA"],
            "Activity": ["Swimming"],
            "Fatal (Y/N)": ["N"],
            "Species ": ["Bull shark"],
        }
    )
    cleaned = standardize_attacks(raw)
    assert cleaned.loc[0, "Year"] == 2020
    assert cleaned.loc[0, "Month"] == "July"
    assert cleaned.loc[0, "Country"] == "United States"
    assert cleaned.loc[0, "Fatal"] == "No"
    assert cleaned.loc[0, "Species"] == "Bull Shark"


def test_duplicate_recognized_columns_are_coalesced():
    raw = pd.DataFrame(
        [["Tiger shark", None, "USA", 2020], [None, "Bull shark", "USA", 2021]],
        columns=["Species", "Species ", "Country", "Year"],
    )
    cleaned = standardize_attacks(raw)
    assert cleaned["Species"].tolist() == ["Tiger Shark", "Bull Shark"]


def test_unknown_years_can_be_retained_or_removed():
    raw = pd.DataFrame(
        {
            "Country": ["USA", "USA"],
            "Year": [2020, None],
            "Species": ["Tiger shark", "Bull shark"],
        }
    )
    cleaned = standardize_attacks(raw)
    kept = filter_attacks(cleaned, year_range=(2020, 2020), include_unknown_years=True)
    removed = filter_attacks(cleaned, year_range=(2020, 2020), include_unknown_years=False)
    assert len(kept) == 2
    assert len(removed) == 1


def test_empty_or_unrelated_files_are_rejected():
    with pytest.raises(ValueError):
        standardize_attacks(pd.DataFrame())
    with pytest.raises(ValueError):
        standardize_attacks(pd.DataFrame({"A": [1], "B": [2]}))


def test_blank_spreadsheet_footer_rows_are_removed():
    raw = pd.DataFrame(
        {
            "Date": ["2020-07-04", "", None],
            "Country": ["USA", "", None],
            "Activity": ["Swimming", "", None],
            "Species": ["Bull shark", "", None],
        }
    )
    cleaned = standardize_attacks(raw)
    assert len(cleaned) == 1
    assert cleaned.loc[0, "Country"] == "United States"
