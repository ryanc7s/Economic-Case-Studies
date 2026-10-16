"""
Helper functions to download data from the Reserve Bank of Australia (RBA).

Every RBA statistical table is published as a CSV at:
    https://www.rba.gov.au/statistics/tables/csv/<table>-data.csv

Each file has ~10 rows of notes at the top, then a "Series ID" row,
then the data. These functions handle that layout so the analysis
scripts can just ask for a table and a list of series IDs.
"""

import csv
import io
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.rba.gov.au/statistics/tables/csv/{}-data.csv"

# All charts and tables are saved here
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Keep downloaded tables in memory so we only fetch each one once
_cache = {}


def _parse_dates(values):
    """RBA files use either 30/06/2026 or 30-Jun-2026 style dates."""
    dates = pd.to_datetime(values, format="%d/%m/%Y", errors="coerce")
    other = pd.to_datetime(values, format="%d-%b-%Y", errors="coerce")
    return dates.fillna(other)


def get_table(table):
    """Download one RBA table (e.g. 'g1') as a DataFrame of series, indexed by date."""
    table = table.lower()
    if table in _cache:
        return _cache[table]

    response = requests.get(BASE_URL.format(table), headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig"))))

    # Find the row with the series IDs - the data starts straight after it
    id_row = next(i for i, row in enumerate(rows) if row and row[0] == "Series ID")
    series_ids = rows[id_row][1:]
    width = len(series_ids) + 1

    data = [row + [""] * (width - len(row)) for row in rows[id_row + 1:] if row and row[0].strip()]
    df = pd.DataFrame([row[1:width] for row in data], columns=series_ids)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = _parse_dates(pd.Series([row[0] for row in data]))
    df.index.name = "date"

    # Drop any unnamed columns and empty rows
    df = df.loc[:, [c for c in df.columns if c]].dropna(how="all").sort_index()
    _cache[table] = df
    return df


def get_series(table, series, start=None):
    """
    Pull selected series from a table.
    `series` is a dict of {series_id: friendly_name}.
    """
    df = get_table(table)[list(series)].rename(columns=series)
    if start:
        df = df.loc[start:]
    return df.dropna(how="all")


def to_quarterly(df, how="mean"):
    """Convert monthly or daily data to quarterly (average or last value in the quarter)."""
    grouped = df.resample("QE")
    return grouped.mean() if how == "mean" else grouped.last()
