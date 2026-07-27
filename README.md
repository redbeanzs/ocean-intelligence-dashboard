# Ocean Intelligence Dashboard

An interactive Streamlit dashboard for exploring recorded shark incident patterns by species,
country, year, activity, and outcome.

## Included project data

The repository includes `data/attacks.csv`, the real CSV used for this project. The app loads it
automatically when deployed, so visitors do not need to upload a file.

The cleaning pipeline removes blank spreadsheet footer rows and standardizes common fields including:

- Date and year
- Country, area, and location
- Activity
- Fatal outcome
- Species descriptions
- Sex and age

The bundled file is approximately 3.1 MB and contains
6,302 usable incident records after cleaning.

## What the project demonstrates

- Real-world data cleaning and column standardization
- Interactive filtering and exploratory analysis
- Plotly data visualization
- Data-quality reporting
- GBIF species taxonomy enrichment with an offline fallback
- Defensive handling of incomplete, duplicated, or malformed data

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Use:

- Repository: `redbeanzs/ocean-intelligence-dashboard`
- Branch: `main`
- Main file: `app.py`

After uploading these files to GitHub, Streamlit should redeploy automatically.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Data note

Recorded incident data can contain missing, inconsistent, duplicated, or revised information.
The dashboard is descriptive and should not be interpreted as a measure of inherent danger,
shark population size, or species behavior.
