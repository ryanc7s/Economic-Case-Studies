"""
01 - Inflation and the RBA cash rate

Questions:
  1. How has inflation tracked against the RBA's 2-3% target since inflation
     targeting began in 1993?
  2. How tight is monetary policy right now? (real cash rate)
  3. Does the cash rate react to inflation, or does inflation react to the
     cash rate? (lead/lag correlations)

Data: RBA tables G1 (CPI) and F1.1 (cash rate)
"""

import matplotlib.pyplot as plt
import pandas as pd

from rba_data import OUTPUT_DIR, get_series, to_quarterly

START = "1993-01-01"
TARGET_LOW, TARGET_HIGH = 2.0, 3.0

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
cpi = get_series("g1", {
    "GCPIAGYP": "headline_cpi",      # all groups CPI, year-ended %
    "GCPIOCPMTMYP": "trimmed_mean",  # RBA's preferred underlying measure
}, start=START)

cash = get_series("f1.1", {"FIRMMCRT": "cash_rate"}, start=START)

# Put everything on a quarterly basis (cash rate = last month of each quarter)
df = to_quarterly(cpi, how="last").join(to_quarterly(cash, how="last")).dropna()

# Real cash rate = cash rate minus underlying inflation
# Negative = policy is stimulating the economy, positive = policy is restrictive
df["real_cash_rate"] = df["cash_rate"] - df["trimmed_mean"]
df["in_target"] = df["trimmed_mean"].between(TARGET_LOW, TARGET_HIGH)

# ---------------------------------------------------------------
# 2. Summary statistics
# ---------------------------------------------------------------
def era(date):
    if date.year < 2008:
        return "1993-2007"
    if date.year < 2020:
        return "2008-2019"
    return "2020-now"

df["era"] = [era(d) for d in df.index]

summary = df.groupby("era").agg(
    avg_headline=("headline_cpi", "mean"),
    avg_trimmed_mean=("trimmed_mean", "mean"),
    avg_cash_rate=("cash_rate", "mean"),
    avg_real_cash_rate=("real_cash_rate", "mean"),
    pct_quarters_in_target=("in_target", "mean"),
)
summary["pct_quarters_in_target"] *= 100

latest = df.iloc[-1]
recent_peak = df.loc["2021":, "headline_cpi"]

print("\n=== Inflation and the cash rate: summary by era ===")
print(summary.round(2).to_string())

print(f"\nLatest quarter: {df.index[-1]:%b %Y}")
print(f"  Headline CPI:        {latest.headline_cpi:.1f}%")
print(f"  Trimmed mean CPI:    {latest.trimmed_mean:.1f}%")
print(f"  Cash rate:           {latest.cash_rate:.2f}%")
print(f"  Real cash rate:      {latest.real_cash_rate:.2f}%")
print(f"Post-COVID inflation peak: {recent_peak.max():.1f}% in {recent_peak.idxmax():%b %Y}")

# Count how many quarters in a row trimmed mean has been outside the target band
outside = (~df["in_target"])[::-1]
streak = int(outside.cumprod().sum())
print(f"Quarters in a row with trimmed mean outside 2-3%: {streak}")

# ---------------------------------------------------------------
# 3. Lead / lag correlation
# Positive lag = inflation today vs the cash rate change k quarters LATER.
# If the strongest correlation is at a positive lag, the RBA is reacting
# to inflation (policy follows prices).
# ---------------------------------------------------------------
df["cash_rate_change_1y"] = df["cash_rate"].diff(4)

lags = range(-8, 9)
lead_lag = pd.Series(
    {k: df["trimmed_mean"].corr(df["cash_rate_change_1y"].shift(-k)) for k in lags},
    name="correlation",
)
lead_lag.index.name = "lag_quarters"

print("\n=== Correlation: trimmed mean inflation vs 1-year change in cash rate ===")
print(lead_lag.round(2).to_string())
best = lead_lag.idxmax()
if best > 0:
    reading = f"the cash rate tends to move about {best} quarter(s) after inflation"
elif best < 0:
    reading = f"inflation lines up best with cash rate moves made {-best} quarter(s) earlier"
else:
    reading = "inflation and the cash rate tend to move at the same time"
print(f"Strongest correlation at lag {best}: {reading}")

# ---------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

ax = axes[0]
ax.axhspan(TARGET_LOW, TARGET_HIGH, color="green", alpha=0.15, label="RBA target band")
ax.plot(df.index, df["headline_cpi"], label="Headline CPI")
ax.plot(df.index, df["trimmed_mean"], label="Trimmed mean CPI")
ax.set_title("Inflation vs the RBA's 2-3% target (year-ended %)")
ax.legend(loc="upper left")

ax = axes[1]
ax.step(df.index, df["cash_rate"], where="post", color="black", label="Cash rate target")
ax.set_title("RBA cash rate (%)")
ax.legend(loc="upper right")

ax = axes[2]
colors = ["tab:red" if x > 0 else "tab:blue" for x in df["real_cash_rate"]]
ax.bar(df.index, df["real_cash_rate"], width=80, color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Real cash rate (cash rate minus trimmed mean) - red = restrictive")

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_inflation_and_cash_rate.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(lead_lag.index, lead_lag.values, color="tab:purple")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("Lag (quarters) - positive = cash rate moves after inflation")
ax.set_ylabel("Correlation")
ax.set_title("Does the cash rate follow inflation?")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_lead_lag_correlation.png", dpi=150)

# Save the data used
df.to_csv(OUTPUT_DIR / "01_inflation_and_cash_rate.csv")
summary.round(2).to_csv(OUTPUT_DIR / "01_summary_by_era.csv")
print(f"\nCharts and tables saved to {OUTPUT_DIR}")
