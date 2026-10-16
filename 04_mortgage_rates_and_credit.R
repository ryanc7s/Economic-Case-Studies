# 04 - Mortgage rates, housing credit and household debt
#
# Questions:
#   1. How much of each RBA cash rate move gets passed on to mortgage rates?
#      (by rate cycle, and estimated with a simple regression)
#   2. How have bank margins over the cash rate changed?
#   3. How fast is housing credit growing, and who is borrowing
#      (owner-occupiers vs investors)?
#   4. How indebted are Australian households relative to income?
#
# Data: RBA tables F1.1 (cash rate), F5 and F6 (lending rates),
#       D2 (credit), E2 (household debt ratios)
#
# Run from the project root folder: source("R/04_mortgage_rates_and_credit.R")

library(ggplot2)
library(tidyr)

source("R/rba_data.R")

start <- "2000-01-01"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
cash <- get_rba_series("f1.1", c(FIRMMCRT = "cash_rate"), start)

# Advertised bank rates (long history)
advertised <- get_rba_series("f5", c(
  FILRHLBVS = "standard_variable",
  FILRHLBVD = "discounted_variable",
  FILRHL3YF = "fixed_3yr"
), start)

# Rates actually paid by owner-occupiers (available from mid-2019)
actual <- get_rba_series("f6", c(
  FLRHOOVA = "outstanding_variable",
  FLRHOFVA = "new_loans_variable"
), start)

rates <- merge(cash, advertised, by = "date", all.x = TRUE)
rates <- merge(rates, actual, by = "date", all.x = TRUE)

# Spreads over the cash rate (a rough guide to bank margins)
rates$spread_standard <- rates$standard_variable - rates$cash_rate
rates$spread_outstanding <- rates$outstanding_variable - rates$cash_rate

# ---------------------------------------------------------------
# 2. Pass-through by rate cycle
#    Pass-through = change in mortgage rate / change in cash rate
#    100% = banks passed on the full move
# ---------------------------------------------------------------
cycles <- data.frame(
  cycle = c("Easing 2019-20", "Tightening 2022-23", "Easing 2025", "Tightening 2026"),
  from = as.Date(c("2019-07-31", "2022-04-30", "2025-01-31", "2026-01-31")),
  to = as.Date(c("2020-12-31", "2023-12-31", "2025-09-30", NA))
)
# Open-ended cycle runs to the latest data
cycles$to[is.na(cycles$to)] <- max(rates$date)

rate_cols <- c("standard_variable", "discounted_variable", "fixed_3yr",
               "outstanding_variable", "new_loans_variable")

value_at <- function(col, d) rates[[col]][which.min(abs(rates$date - d))]

pass_through <- do.call(rbind, lapply(seq_len(nrow(cycles)), function(i) {
  cash_change <- value_at("cash_rate", cycles$to[i]) - value_at("cash_rate", cycles$from[i])
  row <- data.frame(cycle = cycles$cycle[i], cash_rate_change = round(cash_change, 2))
  for (col in rate_cols) {
    change <- value_at(col, cycles$to[i]) - value_at(col, cycles$from[i])
    row[[paste0(col, "_pct")]] <- round(100 * change / cash_change)
  }
  row
}))

cat("\n=== Pass-through of cash rate changes to mortgage rates (%) ===\n")
print(pass_through, row.names = FALSE)

# ---------------------------------------------------------------
# 3. Regression estimate of pass-through
#    Monthly change in mortgage rate ~ change in cash rate this month + last 2 months
#    Long-run pass-through = sum of the three coefficients
# ---------------------------------------------------------------
lag_vec <- function(x, k) c(rep(NA, k), head(x, -k))

reg <- data.frame(
  d_mortgage = c(NA, diff(rates$discounted_variable)),
  d_cash = c(NA, diff(rates$cash_rate))
)
reg$d_cash_lag1 <- lag_vec(reg$d_cash, 1)
reg$d_cash_lag2 <- lag_vec(reg$d_cash, 2)

model <- lm(d_mortgage ~ d_cash + d_cash_lag1 + d_cash_lag2, data = reg)
long_run <- sum(coef(model)[c("d_cash", "d_cash_lag1", "d_cash_lag2")])

cat("\n=== Regression: monthly change in discounted variable rate ===\n")
print(summary(model)$coefficients, digits = 3)
cat(sprintf("Estimated long-run pass-through: %.0f%%\n", 100 * long_run))

# ---------------------------------------------------------------
# 4. Housing credit growth
# ---------------------------------------------------------------
credit <- get_rba_series("d2", c(
  DLCACOHS = "owner_occupier",
  DLCACIHS = "investor",
  DLCACOPS = "personal",
  DLCACNFBS = "business"
), start)

growth <- function(x) 100 * (x / lag_vec(x, 12) - 1)
credit_growth <- data.frame(
  date = credit$date,
  owner_occupier = growth(credit$owner_occupier),
  investor = growth(credit$investor),
  personal = growth(credit$personal),
  business = growth(credit$business)
)
credit$investor_share <- 100 * credit$investor / (credit$investor + credit$owner_occupier)

latest_credit <- tail(na.omit(credit_growth), 1)
cat(sprintf("\n=== Credit growth, year to %s ===\n", format(latest_credit$date, "%b %Y")))
cat(sprintf("  Owner-occupier housing: %.1f%%\n", latest_credit$owner_occupier))
cat(sprintf("  Investor housing:       %.1f%%\n", latest_credit$investor))
cat(sprintf("  Personal:               %.1f%%\n", latest_credit$personal))
cat(sprintf("  Business:               %.1f%%\n", latest_credit$business))
cat(sprintf("  Investor share of housing credit: %.1f%%\n", tail(credit$investor_share, 1)))

# ---------------------------------------------------------------
# 5. Household debt
# ---------------------------------------------------------------
debt <- get_rba_series("e2", c(
  BHFDDIT = "debt_to_income",
  BHFDDIH = "housing_debt_to_income"
), "1990-01-01")

latest_debt <- tail(debt, 1)
cat(sprintf("\nHousehold debt to income (%s): %.0f%% (peak %.0f%%)\n",
            format(latest_debt$date, "%b %Y"), latest_debt$debt_to_income,
            max(debt$debt_to_income, na.rm = TRUE)))

# ---------------------------------------------------------------
# 6. Charts
# ---------------------------------------------------------------
rates_long <- pivot_longer(
  rates[, c("date", "cash_rate", "standard_variable", "discounted_variable",
            "fixed_3yr", "outstanding_variable")],
  -date, names_to = "series", values_to = "rate"
)

p1 <- ggplot(na.omit(rates_long), aes(date, rate, colour = series)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Cash rate vs mortgage rates", x = NULL, y = "%", colour = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave(file.path(output_dir, "04_mortgage_rates.png"), p1, width = 10, height = 6, dpi = 150)

spreads_long <- pivot_longer(rates[, c("date", "spread_standard", "spread_outstanding")],
                             -date, names_to = "series", values_to = "spread")
p2 <- ggplot(na.omit(spreads_long), aes(date, spread, colour = series)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Mortgage rate spread over the cash rate", x = NULL, y = "Percentage points", colour = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave(file.path(output_dir, "04_mortgage_spreads.png"), p2, width = 10, height = 5, dpi = 150)

credit_long <- pivot_longer(credit_growth, -date, names_to = "type", values_to = "growth")
p3 <- ggplot(na.omit(credit_long), aes(date, growth, colour = type)) +
  geom_hline(yintercept = 0, colour = "grey40") +
  geom_line(linewidth = 0.8) +
  labs(title = "Credit growth by borrower type (year-ended %)", x = NULL, y = "%", colour = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave(file.path(output_dir, "04_credit_growth.png"), p3, width = 10, height = 5, dpi = 150)

debt_long <- pivot_longer(debt, -date, names_to = "measure", values_to = "ratio")
p4 <- ggplot(na.omit(debt_long), aes(date, ratio, colour = measure)) +
  geom_line(linewidth = 0.8) +
  labs(title = "Household debt as a % of annual disposable income", x = NULL, y = "%", colour = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave(file.path(output_dir, "04_household_debt.png"), p4, width = 10, height = 5, dpi = 150)

# Save the data used
write.csv(pass_through, file.path(output_dir, "04_pass_through_by_cycle.csv"), row.names = FALSE)
write.csv(rates, file.path(output_dir, "04_mortgage_rates.csv"), row.names = FALSE)
write.csv(credit_growth, file.path(output_dir, "04_credit_growth.csv"), row.names = FALSE)
cat(sprintf("\nCharts and tables saved to %s/\n", output_dir))
