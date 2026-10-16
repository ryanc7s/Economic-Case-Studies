"""
03 - Labour market, wages and real incomes

Questions:
  1. How tight is the Australian labour market today compared with history?
  2. Is there still a wage Phillips curve? (lower unemployment -> faster wage growth)
     And has the relationship changed over time?
  3. Have wages kept up with prices? (real wage growth)
  4. Beveridge curve: are vacancies and unemployment moving as expected?

Data: RBA tables H5 (labour force), H4 (wages, unit labour costs,
      productivity), G1 (CPI)
"""

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

from rba_data import OUTPUT_DIR, get_series, to_quarterly

START = "1998-01-01"  # Wage Price Index starts in 1998

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
labour = get_series("h5", {
    "GLFSURSA": "unemployment_rate",
    "GLFSPRSA": "participation_rate",
    "GLFSEPTPOP": "employment_to_population",
})
vacancies = get_series("h5", {"GLFOSVTLF": "vacancy_ratio"})  # job vacancies as % of labour force
wages = get_series("h4", {
    "GWPIYP": "wage_growth",                 # Wage Price Index, year-ended %
    "GWPIQP": "wage_growth_q",               # Wage Price Index, quarterly %
    "GNFULCYP": "unit_labour_cost_growth",
    "GNFPROSQP": "productivity_growth",
})
cpi = get_series("g1", {"GCPIAGYP": "cpi", "GCPIAGQP": "cpi_q"})

df = (
    to_quarterly(labour, how="mean")
    .join(to_quarterly(vacancies, how="last"))
    .join(to_quarterly(wages, how="last"))
    .join(to_quarterly(cpi, how="last"))
    .loc[START:]
)

# Real wage growth = wage growth minus inflation
df["real_wage_growth"] = df["wage_growth"] - df["cpi"]

# Wages respond to labour market slack with a delay
df["unemployment_lag2"] = df["unemployment_rate"].shift(2)


def era(date):
    if date.year < 2013:
        return "1998-2012"
    if date.year < 2020:
        return "2013-2019"
    if date.year < 2022:
        return "2020-2021 (COVID)"
    return "2022-now"


df["era"] = [era(d) for d in df.index]

# ---------------------------------------------------------------
# 2. Labour market tightness vs history
# ---------------------------------------------------------------
latest = df.dropna(subset=["unemployment_rate"]).iloc[-1]
history = df.loc[:"2019", "unemployment_rate"]

print(f"\n=== Labour market snapshot ({latest.name:%b %Y}) ===")
print(f"  Unemployment rate:        {latest.unemployment_rate:.1f}% "
      f"(1998-2019 average {history.mean():.1f}%, "
      f"lower than {(history > latest.unemployment_rate).mean() * 100:.0f}% of pre-COVID quarters)")
print(f"  Participation rate:       {latest.participation_rate:.1f}%")
print(f"  Employment to population: {latest.employment_to_population:.1f}%")

wage_row = df.dropna(subset=["wage_growth"]).iloc[-1]
print(f"  Wage growth (WPI):        {wage_row.wage_growth:.1f}% ({wage_row.name:%b %Y})")
print(f"  Real wage growth:         {wage_row.real_wage_growth:.1f}%")
print(f"  Productivity growth:      {wage_row.productivity_growth:.1f}%")

# Real wages compared with the end of 2019 (index = 100), built from quarterly changes
since_2020 = df.loc["2020":, ["wage_growth_q", "cpi_q"]].dropna()
real_wage_index = ((1 + since_2020["wage_growth_q"] / 100) / (1 + since_2020["cpi_q"] / 100)).cumprod() * 100
print(f"  Real wage index (Dec 2019 = 100): {real_wage_index.iloc[-1]:.1f}")

print("\n=== Averages by era ===")
print(df.groupby("era")[["unemployment_rate", "wage_growth", "cpi",
                         "real_wage_growth", "productivity_growth"]].mean().round(2).to_string())

# ---------------------------------------------------------------
# 3. Wage Phillips curve by era
#    wage growth = a + b * unemployment (lagged 2 quarters)
#    A more negative b = wages more sensitive to a tight labour market
# ---------------------------------------------------------------
print("\n=== Wage Phillips curve: slope by era ===")
fits = {}
for name, group in df.dropna(subset=["wage_growth", "unemployment_lag2"]).groupby("era"):
    if "COVID" in name:
        continue
    fit = smf.ols("wage_growth ~ unemployment_lag2", data=group).fit()
    fits[name] = fit
    print(f"  {name}: slope = {fit.params['unemployment_lag2']:.2f} "
          f"(p = {fit.pvalues['unemployment_lag2']:.3f}, R2 = {fit.rsquared:.2f}, n = {len(group)})")

full = df.dropna(subset=["wage_growth", "unemployment_lag2"])
full = full[~full["era"].str.contains("COVID")]
fit_all = smf.ols("wage_growth ~ unemployment_lag2", data=full).fit(
    cov_type="HAC", cov_kwds={"maxlags": 4}
)
print(f"  All years ex-COVID: slope = {fit_all.params['unemployment_lag2']:.2f} "
      f"(p = {fit_all.pvalues['unemployment_lag2']:.3f})")

with open(OUTPUT_DIR / "03_phillips_curve_results.txt", "w") as f:
    for name, fit in fits.items():
        f.write(f"{name}\n{fit.summary()}\n\n")
    f.write(f"All years ex-COVID (HAC errors)\n{fit_all.summary()}\n")

# ---------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------
colors = {"1998-2012": "tab:blue", "2013-2019": "tab:orange",
          "2020-2021 (COVID)": "lightgrey", "2022-now": "tab:red"}

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Unemployment and participation
ax = axes[0, 0]
ax.plot(df.index, df["unemployment_rate"], label="Unemployment rate (%)")
ax2 = ax.twinx()
ax2.plot(df.index, df["participation_rate"], color="tab:green", label="Participation rate (%, rhs)")
ax.set_title("Unemployment and participation")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")

# Nominal vs real wages
ax = axes[0, 1]
ax.plot(df.index, df["wage_growth"], label="Wage growth (WPI)")
ax.plot(df.index, df["cpi"], label="CPI inflation")
ax.bar(df.index, df["real_wage_growth"], width=80, color="grey", alpha=0.5, label="Real wage growth")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Wages vs prices (year-ended %)")
ax.legend(loc="upper left")

# Phillips curve scatter with a fitted line per era
ax = axes[1, 0]
for name, group in df.dropna(subset=["wage_growth", "unemployment_lag2"]).groupby("era"):
    ax.scatter(group["unemployment_lag2"], group["wage_growth"], color=colors[name], label=name, alpha=0.7)
    if name in fits:
        x = np.linspace(group["unemployment_lag2"].min(), group["unemployment_lag2"].max(), 20)
        ax.plot(x, fits[name].params["Intercept"] + fits[name].params["unemployment_lag2"] * x,
                color=colors[name])
ax.set_xlabel("Unemployment rate, 2 quarters earlier (%)")
ax.set_ylabel("Wage growth (%)")
ax.set_title("Wage Phillips curve")
ax.legend()

# Beveridge curve
ax = axes[1, 1]
bev = df.dropna(subset=["vacancy_ratio", "unemployment_rate"])
for name, group in bev.groupby("era"):
    ax.plot(group["unemployment_rate"], group["vacancy_ratio"], "o-", color=colors[name],
            label=name, alpha=0.7, markersize=4)
ax.set_xlabel("Unemployment rate (%)")
ax.set_ylabel("Job vacancies (% of labour force)")
ax.set_title("Beveridge curve")
ax.legend()

for ax in axes.flat:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_labour_market_and_wages.png", dpi=150)

# Productivity vs unit labour costs
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(df.index, df["unit_labour_cost_growth"], label="Unit labour cost growth")
ax.plot(df.index, df["productivity_growth"], label="Labour productivity growth")
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Unit labour costs vs productivity (non-farm, year-ended %)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_productivity_and_ulc.png", dpi=150)

df.to_csv(OUTPUT_DIR / "03_labour_market.csv")
print(f"\nCharts and tables saved to {OUTPUT_DIR}")
