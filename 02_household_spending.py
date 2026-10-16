"""
02 - Household spending, inflation and interest rates

An updated and extended version of my original case study, which regressed
retail sales on CPI and the cash rate (2000 onwards).

What's different:
  - The ABS monthly retail trade series has been discontinued, so this uses
    real household consumption from the national accounts instead (a broader
    measure of spending).
  - The original write-up put the weak results down to multicollinearity
    (CPI and the cash rate moving together). This script tests that directly
    with VIFs.
  - Interest rates take time to affect spending, so the improved model uses a
    lagged change in the cash rate rather than today's level.
  - Adds real disposable income, consumer sentiment and a COVID dummy.
  - Uses Newey-West (HAC) standard errors because the data overlaps in time.

Data: RBA tables H2 (household spending, income, saving), G1 (CPI),
      F1.1 (cash rate), H3 (consumer sentiment)
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from rba_data import OUTPUT_DIR, get_series, to_quarterly

START = "2000-01-01"
POLICY_LAG = 4  # quarters for rate changes to flow through to spending

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
spending = get_series("h2", {
    "GGDPECCVPSHY": "consumption_growth",  # real household consumption, year-ended %
    "GGDPICHRDIY": "real_income_growth",   # real household disposable income, year-ended %
    "GGDPICHDISR": "saving_ratio",         # household saving ratio, %
})
cpi = get_series("g1", {"GCPIAGYP": "cpi"})
cash = get_series("f1.1", {"FIRMMCRT": "cash_rate"})
sentiment = get_series("h3", {"GICWMICS": "sentiment"})  # Westpac-MI index, 100 = neutral

df = (
    to_quarterly(spending, how="last")
    .join(to_quarterly(cpi, how="last"))
    .join(to_quarterly(cash, how="last"))
    .join(to_quarterly(sentiment, how="mean"))
)

# New variables
df["cash_rate_change_1y"] = df["cash_rate"].diff(4)
df["cash_rate_change_lagged"] = df["cash_rate_change_1y"].shift(POLICY_LAG)
df["covid"] = ((df.index >= "2020-03-31") & (df.index <= "2021-12-31")).astype(int)

df = df.loc[START:]

# ---------------------------------------------------------------
# 2. Model A - replicate the original approach
#    spending growth ~ CPI + cash rate (same quarter)
# ---------------------------------------------------------------
data_a = df[["consumption_growth", "cpi", "cash_rate"]].dropna()
model_a = smf.ols("consumption_growth ~ cpi + cash_rate", data=data_a).fit()

# Multicollinearity check: VIF above ~5 is a warning sign
X = data_a[["cpi", "cash_rate"]].assign(const=1)
vif = pd.Series(
    [variance_inflation_factor(X.values, i) for i in range(2)],
    index=["cpi", "cash_rate"],
    name="VIF",
)

# ---------------------------------------------------------------
# 3. Model B - improved model
# ---------------------------------------------------------------
formula_b = (
    "consumption_growth ~ cash_rate_change_lagged + cpi"
    " + real_income_growth + sentiment + covid"
)
data_b = df[["consumption_growth", "cash_rate_change_lagged", "cpi",
             "real_income_growth", "sentiment", "covid"]].dropna()
model_b = smf.ols(formula_b, data=data_b).fit(cov_type="HAC", cov_kwds={"maxlags": 4})

# Same model excluding the COVID period completely, as a robustness check
data_c = data_b[data_b["covid"] == 0]
model_c = smf.ols(formula_b.replace(" + covid", ""), data=data_c).fit(
    cov_type="HAC", cov_kwds={"maxlags": 4}
)

# ---------------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------------
def sample(data):
    return f"{data.index[0]:%b %Y} to {data.index[-1]:%b %Y}, {len(data)} quarters"

print("\n=== Model A: original approach (spending ~ CPI + cash rate) ===")
print(f"Sample: {sample(data_a)}")
print(model_a.summary().tables[1])
print(f"R-squared: {model_a.rsquared:.3f}")
print(f"Correlation between CPI and cash rate: {data_a['cpi'].corr(data_a['cash_rate']):.2f}")
print("Variance inflation factors:")
print(vif.round(2).to_string())
if vif.max() < 5:
    print("VIFs are well below 5, so multicollinearity is not a major issue in this sample.")
else:
    print("VIFs above 5 suggest CPI and the cash rate overlap too much to separate their effects.")

print("\n=== Model B: lagged rate changes + income + sentiment + COVID dummy (HAC errors) ===")
print(f"Sample: {sample(data_b)} (shorter because the sentiment data starts in 2010)")
print(model_b.summary().tables[1])
print(f"R-squared: {model_b.rsquared:.3f}")

print("\n=== Model C: same as B but excluding 2020-2021 ===")
print(f"Sample: {sample(data_c)}")
print(model_c.summary().tables[1])
print(f"R-squared: {model_c.rsquared:.3f}")

effect = model_c.params["cash_rate_change_lagged"]
print(
    f"\nReading Model C: a 1 percentage point rise in the cash rate over a year is "
    f"associated with real spending growth {abs(effect):.2f} pp "
    f"{'lower' if effect < 0 else 'higher'} {POLICY_LAG} quarters later "
    f"(p = {model_c.pvalues['cash_rate_change_lagged']:.3f})."
)

latest = df.dropna(subset=["consumption_growth"]).iloc[-1]
print(f"\nLatest ({latest.name:%b %Y}): spending growth {latest.consumption_growth:.1f}%, "
      f"real income growth {latest.real_income_growth:.1f}%, saving ratio {latest.saving_ratio:.1f}%")

# Save model output to a text file
with open(OUTPUT_DIR / "02_regression_results.txt", "w") as f:
    for name, model in [("Model A", model_a), ("Model B", model_b), ("Model C", model_c)]:
        f.write(f"{name}\n{model.summary()}\n\n")
    f.write(f"VIFs for Model A\n{vif.round(2).to_string()}\n")

# ---------------------------------------------------------------
# 5. Charts
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

ax = axes[0]
ax.plot(df.index, df["consumption_growth"], label="Real household spending growth (y/y %)")
ax.plot(df.index, df["real_income_growth"], label="Real disposable income growth (y/y %)", alpha=0.8)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylim(-15, 20)  # stop the 2020 swings flattening everything else
ax.set_title("Household spending vs income")
ax.legend(loc="upper left")

ax = axes[1]
ax.plot(df.index, df["cash_rate"], color="black", label="Cash rate")
ax.plot(df.index, df["cpi"], color="tab:red", label="Headline CPI (y/y %)")
ax.set_title("Interest rates and inflation")
ax.legend(loc="upper left")

ax = axes[2]
ax.fill_between(df.index, df["saving_ratio"], color="tab:green", alpha=0.4)
ax.set_title("Household saving ratio (%)")

for ax in axes:
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_household_spending.png", dpi=150)

# Scatter: lagged rate changes vs spending growth (excluding COVID)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(data_c["cash_rate_change_lagged"], data_c["consumption_growth"], alpha=0.6)
ax.set_xlabel(f"Change in cash rate over prior year, lagged {POLICY_LAG} quarters (pp)")
ax.set_ylabel("Real household spending growth (y/y %)")
ax.set_title("Rate rises vs later spending growth (ex-COVID)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_rates_vs_spending_scatter.png", dpi=150)

# Coefficient plot for Model C
coefs = model_c.params.drop("Intercept")
errors = model_c.bse.drop("Intercept") * 1.96
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(coefs.values, range(len(coefs)), xerr=errors.values, fmt="o", capsize=4)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(coefs)), coefs.index)
ax.set_title("Model C coefficients with 95% confidence intervals")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_model_coefficients.png", dpi=150)

df.to_csv(OUTPUT_DIR / "02_household_spending.csv")
print(f"\nCharts and tables saved to {OUTPUT_DIR}")
