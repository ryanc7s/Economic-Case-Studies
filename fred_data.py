"""
Helper function to download data from FRED (Federal Reserve Bank of St. Louis).

Every FRED series can be downloaded as a CSV at:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series_id>

No API key is needed.
"""

import io

import pandas as pd
import requests

BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def get_fred(series):
    """
    Download one or more FRED series.
    `series` is a dict of {series_id: friendly_name}.
    Returns a DataFrame indexed by date (series can have different frequencies).
    """
    frames = []
    for series_id, name in series.items():
        # Note: FRED blocks some custom user agents, so keep the requests default
        response = requests.get(BASE_URL.format(series_id), timeout=60)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True)
        frames.append(pd.to_numeric(df.iloc[:, 0], errors="coerce").rename(name))
    out = pd.concat(frames, axis=1)
    out.index.name = "date"
    return out
