"""
06 - Distressed debt: what drives corporate default rates?

Research on high-yield defaults points to three main drivers:
  1. Economic performance - weaker growth means less cash flow to service debt
  2. Quantity of lower-rated debt - a surge in weak borrowers tends to show up
     as defaults about 3 years later (the "aging effect")
  3. Capital market liquidity - when lenders pull back, weak companies cannot
     refinance and are more likely to default

This script tests how well each factor explains defaults, combines them into
one model, checks how that model would have forecast out of sample, and gives
a one-year-ahead default outlook under a few scenarios.

This uses US data because it has the long default history this needs
(Australia has no comparable public series).

Default rate:
  Free bond default rates are not published, so by default this uses the US
  charge-off rate on commercial & industrial bank loans as a proxy. It follows
  the same cycle as the Moody's / S&P default rates (peaks in 1991, 2001-02,
  2009 and 2020).

  If you have actual default data (e.g. Moody's or S&P speculative-grade
  default rates), save it as data/default_rates.csv with columns:
      date, default_rate                       (required)
      low_rated_share                          (optional - e.g. % of new issuance rated B- or below)
  and the script will use it instead. The data/ folder is excluded in
  .gitignore, so licensed data won't be uploaded to GitHub.

Data: FRED (US Federal Reserve Economic Data)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from fred_data import get_fred
from rba_data import OUTPUT_DIR, to_quarterly

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "default_rates.csv"

AGING_LAG = 12        # quarters between a debt surge and the defaults that follow (~3 years)
HORIZON = 4           # forecast defaults this many quarters ahead
TRAIN_END = "2012-12-31"  # out-of-sample test uses data after this date
DISTRESS_LEVEL = 10.0     # spreads above 10% (1,000bp) are the usual "distressed" line

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
print("Downloading data from FRED...")
quarterly = get_fred({
    "CORBLACBS": "default_rate",       # charge-off rate on C&I loans, % annualised
    "GDPC1": "real_gdp",
    "GDP": "nominal_gdp",
    "NCBDBIQ027S": "corporate_bonds",  # nonfinancial corporate debt securities outstanding
    "DRTSCILM": "lending_standards",   # net % of banks tightening business loan standards
})
monthly = get_fred({
    "INDPRO": "industrial_production",
    "UNRATE": "unemployment_rate",
    "USREC": "recession",              # 1 = NBER recession
})
daily = get_fred({
    "BAA10Y": "baa_spread",            # Moody's Baa corporate yield minus 10-year Treasury
    "NFCI": "financial_conditions",    # Chicago Fed index, positive = tighter than average
})
# ICE BofA spreads - FRED only provides the last 3 years, so used for the current snapshot only
spreads = get_fred({
    "BAMLH0A0HYM2": "high_yield",
    "BAMLH0A1HYBB": "BB",
    "BAMLH0A3HYC": "CCC_and_lower",
}).dropna(how="all")

df = (
    to_quarterly(quarterly, how="last")
    .join(to_quarterly(monthly, how="mean"))
    .join(to_quarterly(daily, how="mean"))
)
df["recession"] = (df["recession"] > 0).astype(int)

# Swap in actual default data if it has been provided
if DATA_FILE.exists():
    own = pd.read_csv(DATA_FILE, index_col="date", parse_dates=True)
    own = to_quarterly(own, how="last")
    df["default_rate"] = own["default_rate"]
    if "low_rated_share" in own:
        df["low_rated_share"] = own["low_rated_share"]
    default_label = "Default rate (own data, %)"
    print(f"Using default data from {DATA_FILE.name}")
else:
    default_label = "C&I loan charge-off rate (default proxy, %)"
    print("Using C&I loan charge-off rate as the default proxy")

# ---------------------------------------------------------------
# 2. Build the three factors
# ---------------------------------------------------------------
# Economic performance
df["gdp_growth"] = df["real_gdp"].pct_change(4) * 100
df["ip_growth"] = df["industrial_production"].pct_change(4) * 100
df["unemployment_change"] = df["unemployment_rate"].diff(4)

# Quantity of lower-rated debt
# If no rating mix data is supplied, use "excess" bond issuance as a proxy:
# corporate bond growth above nominal GDP growth. Rapid issuance tends to pull
# in weaker borrowers, who then default a few years later.
if "low_rated_share" not in df:
    df["low_rated_share"] = (df["corporate_bonds"].pct_change(4) - df["nominal_gdp"].pct_change(4)) * 100
df["low_rated_lagged"] = df["low_rated_share"].shift(AGING_LAG)

# Capital market liquidity: bank lending standards (main), plus market measures as alternatives
df["lending_standards_lagged"] = df["lending_standards"].shift(2)

# Target for the forecast model: the default rate one year from now
df["default_rate_ahead"] = df["default_rate"].shift(-HORIZON)

# ---------------------------------------------------------------
# 3. Check the aging lag: which lag of the debt measure lines up best with defaults?
# ---------------------------------------------------------------
lag_check = pd.Series({
    lag: df["default_rate"].corr(df["low_rated_share"].shift(lag)) for lag in range(0, 21, 2)
}, name="correlation")
lag_check.index.name = "lag_quarters"
print("\n=== Correlation of default rate with lower-rated debt measure, by lag ===")
print(lag_check.round(2).to_string())
print(f"Using a lag of {AGING_LAG} quarters (strongest in this data: {lag_check.idxmax()})")
if abs(lag_check[AGING_LAG]) < 0.3:
    print("Note: the link is weak at this lag. Total bond growth is only a rough stand-in for the share "
          "of low-rated debt - actual rating mix data (see the notes at the top of this script) will test this properly.")

# ---------------------------------------------------------------
# 4. Which single factor explains defaults best?
# ---------------------------------------------------------------
factors = {
    "gdp_growth": "Economy - real GDP growth",
    "ip_growth": "Economy - industrial production growth",
    "unemployment_change": "Economy - change in unemployment",
    "low_rated_lagged": f"Lower-rated debt - lagged {AGING_LAG}q",
    "lending_standards_lagged": "Liquidity - bank lending standards (lag 2q)",
    "baa_spread": "Liquidity - Baa credit spread",
    "financial_conditions": "Liquidity - financial conditions index",
}

rows = []
for col, label in factors.items():
    data = df[["default_rate", col]].dropna()
    fit = smf.ols(f"default_rate ~ {col}", data=data).fit()
    rows.append({
        "factor": label,
        "correlation": data["default_rate"].corr(data[col]),
        "r_squared": fit.rsquared,
        "start": f"{data.index[0]:%Y}",
    })
single = pd.DataFrame(rows).set_index("factor").sort_values("r_squared", ascending=False)

print("\n=== Each factor on its own vs the default rate ===")
print(single.round(2).to_string())

# ---------------------------------------------------------------
# 5. Combined models
#    Model 1 (explain): default rate today vs the three factors today
#    Model 2 (forecast): default rate in one year vs the three factors today
#    Variables are standardised so the coefficients can be compared:
#    each one is the change in defaults (pp) from a one standard deviation move.
# ---------------------------------------------------------------
model_vars = ["gdp_growth", "low_rated_lagged", "lending_standards_lagged"]


def standardise(data, cols):
    out = data.copy()
    out[cols] = (data[cols] - data[cols].mean()) / data[cols].std()
    return out


explain_data = df[["default_rate"] + model_vars].dropna()
explain = smf.ols("default_rate ~ " + " + ".join(model_vars),
                  data=standardise(explain_data, model_vars)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

forecast_data = df[["default_rate_ahead", "default_rate"] + model_vars].dropna()
forecast_formula = "default_rate_ahead ~ default_rate + " + " + ".join(model_vars)
forecast = smf.ols(forecast_formula, data=forecast_data).fit(cov_type="HAC", cov_kwds={"maxlags": HORIZON})

print(f"\n=== Model 1: default rate explained by the three factors "
      f"({explain_data.index[0]:%Y}-{explain_data.index[-1]:%Y}, standardised) ===")
print(explain.summary().tables[1])
print(f"R-squared: {explain.rsquared:.2f}")

print(f"\n=== Model 2: default rate {HORIZON} quarters ahead (forecast model) ===")
print(forecast.summary().tables[1])
print(f"R-squared: {forecast.rsquared:.2f}")

# ---------------------------------------------------------------
# 6. Out-of-sample test
#    Fit the forecast model only on data up to TRAIN_END, then predict after.
#    Compare against a naive guess that defaults stay where they are today.
# ---------------------------------------------------------------
train = forecast_data.loc[:TRAIN_END]
test = forecast_data.loc[TRAIN_END:].iloc[1:]
oos_fit = smf.ols(forecast_formula, data=train).fit()
oos_pred = oos_fit.predict(test)

rmse_model = ((oos_pred - test["default_rate_ahead"]) ** 2).mean() ** 0.5
rmse_naive = ((test["default_rate"] - test["default_rate_ahead"]) ** 2).mean() ** 0.5
print(f"\n=== Out-of-sample test ({test.index[0]:%Y}-{test.index[-1]:%Y}) ===")
print(f"  Model error (RMSE):         {rmse_model:.2f} pp")
print(f"  Naive 'no change' error:    {rmse_naive:.2f} pp")
print(f"  Model {'beats' if rmse_model < rmse_naive else 'does not beat'} the naive forecast")
print(f"  Average forecast {oos_pred.mean():.2f}% vs actual {test['default_rate_ahead'].mean():.2f}%")
if oos_pred.mean() > test["default_rate_ahead"].mean():
    print("  Defaults have come in lower than the drivers suggest - e.g. government support kept "
          "defaults low in 2020-21, and tighter lending in 2023-24 did not lift defaults as much as in past cycles.")

# ---------------------------------------------------------------
# 7. One-year-ahead outlook and scenarios
#    The lower-rated debt factor is already known (it is lagged 3 years),
#    so only the economy and liquidity change between scenarios.
# ---------------------------------------------------------------
latest = df[["default_rate"] + model_vars].dropna().iloc[-1].copy()
latest_date = df[["default_rate"] + model_vars].dropna().index[-1]

scenarios = {
    "Base (latest data)": {},
    "Slowdown": {"gdp_growth": 0.5, "lending_standards_lagged": 25},
    "Recession": {"gdp_growth": -2.0, "lending_standards_lagged": 60},
}
outlook = []
for name, changes in scenarios.items():
    inputs = latest.copy()
    for col, value in changes.items():
        inputs[col] = value
    pred = forecast.get_prediction(inputs.to_frame().T).summary_frame(alpha=0.1)
    outlook.append({
        "scenario": name,
        "gdp_growth": inputs["gdp_growth"],
        "lending_standards": inputs["lending_standards_lagged"],
        "forecast_default_rate": pred["mean"].iloc[0],
        "low_90": pred["mean_ci_lower"].iloc[0],
        "high_90": pred["mean_ci_upper"].iloc[0],
    })
outlook = pd.DataFrame(outlook).set_index("scenario")

print(f"\n=== Default outlook, {HORIZON} quarters after {latest_date:%b %Y} ===")
print(f"Current default rate: {latest['default_rate']:.2f}%  "
      f"(long-run average {df['default_rate'].mean():.2f}%)")
print(outlook.round(2).to_string())

# ---------------------------------------------------------------
# 8. Current market snapshot (high-yield spreads, last 3 years)
# ---------------------------------------------------------------
print("\n=== High-yield spreads (option-adjusted, %) ===")
snap = pd.DataFrame({
    "latest": spreads.ffill().iloc[-1],
    "3y_low": spreads.min(),
    "3y_high": spreads.max(),
    "3y_average": spreads.mean(),
})
print(snap.round(2).to_string())
ccc = spreads["CCC_and_lower"].dropna()
print(f"CCC & lower spread is {'ABOVE' if ccc.iloc[-1] > DISTRESS_LEVEL else 'below'} "
      f"the {DISTRESS_LEVEL:.0f}% distressed line ({ccc.iloc[-1]:.2f}% on {ccc.index[-1]:%d %b %Y})")
print(f"Gap between CCC and BB spreads: {(spreads['CCC_and_lower'] - spreads['BB']).dropna().iloc[-1]:.2f} pp "
      "(a widening gap means investors are singling out the weakest credits)")

# Save results
with open(OUTPUT_DIR / "06_default_model_results.txt", "w") as f:
    f.write(f"Model 1 (explain, standardised)\n{explain.summary()}\n\n")
    f.write(f"Model 2 (forecast {HORIZON}q ahead)\n{forecast.summary()}\n\n")
    f.write(f"Out-of-sample RMSE: model {rmse_model:.2f}, naive {rmse_naive:.2f}\n")
single.round(3).to_csv(OUTPUT_DIR / "06_single_factor_fit.csv")
outlook.round(3).to_csv(OUTPUT_DIR / "06_default_outlook.csv")
df.to_csv(OUTPUT_DIR / "06_default_drivers.csv")

# ---------------------------------------------------------------
# 9. Charts
# ---------------------------------------------------------------
def shade_recessions(ax, data):
    rec = data["recession"]
    starts = rec.index[(rec == 1) & (rec.shift(1, fill_value=0) == 0)]
    ends = rec.index[(rec == 1) & (rec.shift(-1, fill_value=0) == 0)]
    for s, e in zip(starts, ends):
        ax.axvspan(s - pd.offsets.QuarterBegin(1), e, color="grey", alpha=0.2)


plot = df.loc["1988":]
panels = [
    ("gdp_growth", "Real GDP growth (y/y %)", "Economic performance"),
    ("low_rated_lagged", f"Lower-rated debt measure, lagged {AGING_LAG}q", "Quantity of lower-rated debt"),
    ("lending_standards", "Net % of banks tightening standards", "Capital market liquidity"),
]
fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
for ax, (col, label, title) in zip(axes, panels):
    shade_recessions(ax, plot)
    ax.plot(plot.index, plot["default_rate"], color="black", lw=2, label=default_label)
    ax2 = ax.twinx()
    ax2.plot(plot.index, plot[col], color="tab:red", alpha=0.8, label=label)
    ax.set_title(f"Defaults vs {title.lower()} (grey = US recession)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_default_drivers.png", dpi=150)

# Actual vs model, with the out-of-sample period and scenarios
fitted = forecast.predict(forecast_data)
fig, ax = plt.subplots(figsize=(11, 5))
shade_recessions(ax, plot)
target_dates = forecast_data.index + pd.offsets.QuarterEnd(HORIZON)
ax.plot(target_dates, forecast_data["default_rate_ahead"], color="black", lw=2, label="Actual")
ax.plot(target_dates, fitted, color="tab:blue", label="Model (full sample)")
ax.plot(test.index + pd.offsets.QuarterEnd(HORIZON), oos_pred, color="tab:orange",
        ls="--", label=f"Model fitted only to {TRAIN_END[:4]} (out-of-sample)")
outlook_date = latest_date + pd.offsets.QuarterEnd(HORIZON)
for i, ((name, row), color) in enumerate(zip(outlook.iterrows(), ["tab:green", "gold", "tab:red"])):
    # Spread the points out a little so the error bars don't overlap
    ax.errorbar(outlook_date + pd.Timedelta(days=60 * (i - 1)), row["forecast_default_rate"],
                yerr=[[row["forecast_default_rate"] - row["low_90"]], [row["high_90"] - row["forecast_default_rate"]]],
                fmt="o", color=color, capsize=4, label=f"Outlook: {name}")
ax.set_title(f"{default_label}: actual vs {HORIZON}-quarter-ahead model")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_default_forecast.png", dpi=150)

# Which factor matters most? (standardised coefficients from Model 1)
coefs = explain.params.drop("Intercept")
errors = explain.bse.drop("Intercept") * 1.96
labels = ["Economic performance", "Lower-rated debt (lagged)", "Liquidity (lending standards)"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(labels, coefs.values, xerr=errors.values, color=["tab:blue", "tab:purple", "tab:red"], capsize=4)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("Change in default rate (pp) from a 1 standard deviation move")
ax.set_title("Relative importance of each driver (95% intervals)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_driver_importance.png", dpi=150)

# Current high-yield spreads
fig, ax = plt.subplots(figsize=(11, 4))
for col in spreads:
    ax.plot(spreads.index, spreads[col], label=col.replace("_", " "))
ax.axhline(DISTRESS_LEVEL, color="red", ls="--", lw=1, label="Distressed (10%)")
ax.set_title("US high-yield spreads by rating (option-adjusted, %)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_high_yield_spreads.png", dpi=150)

print(f"\nCharts and tables saved to {OUTPUT_DIR}")
