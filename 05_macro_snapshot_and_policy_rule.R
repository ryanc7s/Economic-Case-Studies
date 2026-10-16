# 05 - Australian macro snapshot and a simple policy rule
#
# Questions:
#   1. Where does the Australian economy stand today vs a year ago?
#   2. What does a simple Taylor-style rule say the cash rate "should" be,
#      and how does that compare with the actual cash rate?
#   3. How sensitive is that answer to the assumptions (neutral rate, NAIRU)?
#
# Data: RBA tables H1 (GDP), G1 (CPI), H5 (labour force), H4 (wages),
#       F1.1 (cash rate), F2 (bond yields), F11 (AUD), H3 (sentiment)
#
# Run from the project root folder: source("R/05_macro_snapshot_and_policy_rule.R")

library(ggplot2)
library(tidyr)

source("R/rba_data.R")

# ---------------------------------------------------------------
# Policy rule assumptions - change these to test other views
# ---------------------------------------------------------------
neutral_real_rate <- 1.0   # real cash rate that neither stimulates nor restrains (%)
inflation_target  <- 2.5   # midpoint of the RBA's 2-3% band
nairu             <- 4.5   # unemployment rate consistent with stable inflation (%)
weight_inflation  <- 0.5   # how hard the rule leans against inflation
weight_unemp      <- 1.0   # how hard the rule leans against unemployment gaps

# ---------------------------------------------------------------
# 1. Load data and put everything on a quarterly basis
# ---------------------------------------------------------------
gdp <- get_rba_series("h1", c(GGDPCVGDPY = "gdp_growth", GOPITT = "terms_of_trade"))
cpi <- get_rba_series("g1", c(GCPIAGYP = "headline_cpi", GCPIOCPMTMYP = "trimmed_mean"))
labour <- to_quarterly(get_rba_series("h5", c(GLFSURSA = "unemployment_rate")), "mean")
wages <- get_rba_series("h4", c(GWPIYP = "wage_growth"))
cash <- to_quarterly(get_rba_series("f1.1", c(FIRMMCRT = "cash_rate")), "last")
bonds <- to_quarterly(get_rba_series("f2", c(FCMYGBAG10D = "bond_10yr", FCMYGBAG3D = "bond_3yr")), "mean")
aud <- to_quarterly(get_rba_series("f11", c(FXRUSD = "aud_usd", FXRTWI = "aud_twi")), "mean")
sentiment <- to_quarterly(get_rba_series("h3", c(GICWMICS = "consumer_sentiment")), "mean")

macro <- Reduce(
  function(a, b) merge(a, b, by = "date", all = TRUE),
  list(to_quarterly(gdp, "last"), to_quarterly(cpi, "last"), labour,
       to_quarterly(wages, "last"), cash, bonds, aud, sentiment)
)
macro <- macro[macro$date >= as.Date("1993-01-01"), ]

# Yield curve slope: 10-year bond minus cash rate (negative = inverted)
macro$yield_curve <- macro$bond_10yr - macro$cash_rate

# ---------------------------------------------------------------
# 2. Snapshot table: latest value vs one year earlier
#    Note: daily/monthly series are quarterly averages, so the latest
#    quarter may only be partly complete
# ---------------------------------------------------------------
indicators <- c(
  gdp_growth = "Real GDP growth (y/y %)",
  headline_cpi = "Headline CPI (y/y %)",
  trimmed_mean = "Trimmed mean CPI (y/y %)",
  unemployment_rate = "Unemployment rate (%)",
  wage_growth = "Wage Price Index (y/y %)",
  cash_rate = "Cash rate (%)",
  bond_10yr = "10-year govt bond yield (%)",
  yield_curve = "10-year minus cash rate (pp)",
  aud_usd = "AUD/USD",
  terms_of_trade = "Terms of trade (index)",
  consumer_sentiment = "Consumer sentiment (100 = neutral)"
)

snapshot <- do.call(rbind, lapply(names(indicators), function(col) {
  rows <- macro[!is.na(macro[[col]]), c("date", col)]
  latest <- tail(rows, 1)
  year_ago <- rows[rows$date == quarter_end(seq(latest$date, by = "-1 year", length.out = 2)[2]), col]
  data.frame(
    indicator = indicators[[col]],
    as_at = format(latest$date, "%b %Y"),
    latest = round(latest[[col]], 2),
    year_ago = if (length(year_ago)) round(year_ago, 2) else NA,
    change = if (length(year_ago)) round(latest[[col]] - year_ago, 2) else NA
  )
}))

cat("\n=== Australian macro snapshot ===\n")
print(snapshot, row.names = FALSE)

# ---------------------------------------------------------------
# 3. Taylor-style policy rule
#    cash rate = neutral real rate + inflation
#                + 0.5 x (inflation - target)
#                - 1.0 x (unemployment - NAIRU)
# ---------------------------------------------------------------
policy_rule <- function(inflation, unemployment, neutral = neutral_real_rate, u_star = nairu) {
  neutral + inflation +
    weight_inflation * (inflation - inflation_target) -
    weight_unemp * (unemployment - u_star)
}

macro$rule_rate <- policy_rule(macro$trimmed_mean, macro$unemployment_rate)
macro$rule_gap <- macro$cash_rate - macro$rule_rate

latest_rule <- tail(macro[!is.na(macro$rule_rate) & !is.na(macro$cash_rate), ], 1)
cat(sprintf("\n=== Policy rule (%s) ===\n", format(latest_rule$date, "%b %Y")))
cat(sprintf("  Actual cash rate:        %.2f%%\n", latest_rule$cash_rate))
cat(sprintf("  Rule-implied cash rate:  %.2f%%\n", latest_rule$rule_rate))
cat(sprintf("  Gap (actual - rule):     %+.2f pp -> policy looks %s than the rule suggests\n",
            latest_rule$rule_gap, ifelse(latest_rule$rule_gap < 0, "looser", "tighter")))

# How well does the rule track history? (correlation since 1993)
cat(sprintf("  Correlation of rule with actual cash rate since 1993: %.2f\n",
            cor(macro$rule_rate, macro$cash_rate, use = "complete.obs")))

# Sensitivity: the rule-implied rate today under different assumptions
grid <- expand.grid(neutral_real_rate = c(0.5, 1.0, 1.5), nairu = c(4.0, 4.5, 5.0))
grid$implied_cash_rate <- round(policy_rule(latest_rule$trimmed_mean, latest_rule$unemployment_rate,
                                            grid$neutral_real_rate, grid$nairu), 2)
sensitivity <- pivot_wider(grid, names_from = nairu, values_from = implied_cash_rate,
                           names_prefix = "NAIRU_")
cat("\nRule-implied cash rate under different assumptions (rows = neutral real rate):\n")
print(as.data.frame(sensitivity), row.names = FALSE)

# ---------------------------------------------------------------
# 4. Charts
# ---------------------------------------------------------------
# Dashboard of key indicators since 2015
dash_cols <- c("gdp_growth", "trimmed_mean", "unemployment_rate", "wage_growth",
               "cash_rate", "bond_10yr", "aud_usd", "consumer_sentiment")
dash <- pivot_longer(macro[macro$date >= as.Date("2015-01-01"), c("date", dash_cols)],
                     -date, names_to = "indicator", values_to = "value")
dash$indicator <- factor(indicators[dash$indicator], levels = indicators[dash_cols])

p1 <- ggplot(na.omit(dash), aes(date, value)) +
  geom_line(colour = "steelblue", linewidth = 0.8) +
  facet_wrap(~indicator, scales = "free_y", ncol = 2) +
  labs(title = "Australian macro dashboard", x = NULL, y = NULL) +
  theme_minimal()
ggsave(file.path(output_dir, "05_macro_dashboard.png"), p1, width = 11, height = 11, dpi = 150)

rule_long <- pivot_longer(macro[, c("date", "cash_rate", "rule_rate")],
                          -date, names_to = "series", values_to = "rate")
p2 <- ggplot(na.omit(rule_long), aes(date, rate, colour = series)) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = c(cash_rate = "black", rule_rate = "tomato"),
                      labels = c("Actual cash rate", "Policy rule")) +
  labs(title = "Actual cash rate vs a simple Taylor-style rule",
       subtitle = sprintf("Neutral real rate %.1f%%, NAIRU %.1f%%, using trimmed mean inflation",
                          neutral_real_rate, nairu),
       x = NULL, y = "%", colour = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom")
ggsave(file.path(output_dir, "05_policy_rule.png"), p2, width = 10, height = 5, dpi = 150)

p3 <- ggplot(na.omit(macro[, c("date", "yield_curve")]), aes(date, yield_curve)) +
  geom_hline(yintercept = 0, colour = "grey40") +
  geom_col(aes(fill = yield_curve < 0), show.legend = FALSE) +
  scale_fill_manual(values = c(`TRUE` = "tomato", `FALSE` = "steelblue")) +
  labs(title = "Yield curve: 10-year bond minus cash rate (red = inverted)", x = NULL, y = "pp") +
  theme_minimal()
ggsave(file.path(output_dir, "05_yield_curve.png"), p3, width = 10, height = 4, dpi = 150)

# Save the data used
write.csv(snapshot, file.path(output_dir, "05_macro_snapshot.csv"), row.names = FALSE)
write.csv(macro, file.path(output_dir, "05_macro_quarterly.csv"), row.names = FALSE)
cat(sprintf("\nCharts and tables saved to %s/\n", output_dir))
