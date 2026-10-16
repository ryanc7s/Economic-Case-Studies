# Helper functions to download data from the Reserve Bank of Australia (RBA).
#
# Every RBA statistical table is published as a CSV at:
#   https://www.rba.gov.au/statistics/tables/csv/<table>-data.csv
#
# Each file has ~10 rows of notes at the top, then a "Series ID" row,
# then the data. These functions handle that layout so the analysis
# scripts can just ask for a table and a list of series IDs.

# All charts and tables are saved here
output_dir <- "output"
dir.create(output_dir, showWarnings = FALSE)

# Keep downloaded tables in memory so we only fetch each one once
.rba_cache <- new.env()

# RBA files use either 30/06/2026 or 30-Jun-2026 style dates
parse_rba_date <- function(x) {
  for (i in 1:12) {
    x <- sub(paste0("-", month.abb[i], "-"), sprintf("/%02d/", i), x)
  }
  as.Date(x, format = "%d/%m/%Y")
}

# Download one RBA table (e.g. "g1") as a data frame: date + one column per series ID
read_rba_table <- function(table) {
  table <- tolower(table)
  if (exists(table, envir = .rba_cache)) {
    return(get(table, envir = .rba_cache))
  }

  url <- sprintf("https://www.rba.gov.au/statistics/tables/csv/%s-data.csv", table)
  tmp <- tempfile(fileext = ".csv")
  download.file(url, tmp, quiet = TRUE, mode = "wb", headers = c("User-Agent" = "Mozilla/5.0"))
  lines <- readLines(tmp, encoding = "UTF-8", warn = FALSE)

  # Find the row with the series IDs - the data starts straight after it
  id_row <- grep("^Series ID,", lines)[1]
  ids <- strsplit(lines[id_row], ",")[[1]][-1]

  df <- read.csv(
    text = lines[(id_row + 1):length(lines)],
    header = FALSE,
    col.names = c("date", ids),
    colClasses = "character",
    fill = TRUE,
    check.names = FALSE
  )
  df <- df[df$date != "", ]
  df$date <- parse_rba_date(df$date)
  df[ids] <- lapply(df[ids], function(col) suppressWarnings(as.numeric(col)))
  df <- df[order(df$date), ]

  assign(table, df, envir = .rba_cache)
  df
}

# Pull selected series from a table.
# `series` is a named vector: c(SERIES_ID = "friendly_name")
get_rba_series <- function(table, series, start = NULL) {
  df <- read_rba_table(table)[, c("date", names(series))]
  names(df) <- c("date", unname(series))
  if (!is.null(start)) {
    df <- df[df$date >= as.Date(start), ]
  }
  # Drop rows where every series is missing
  df[rowSums(!is.na(df[-1])) > 0, ]
}

# Last day of the quarter for any date (e.g. 2026-08-31 -> 2026-09-30)
quarter_end <- function(d) {
  q_month <- ((as.integer(format(d, "%m")) - 1) %/% 3 + 1) * 3
  first_of_next <- as.Date(sprintf("%s-%02d-01", format(d, "%Y"), q_month %% 12 + 1))
  # December quarters roll into January of the next year
  first_of_next[q_month == 12] <- as.Date(paste0(as.integer(format(d[q_month == 12], "%Y")) + 1, "-01-01"))
  first_of_next - 1
}

# Convert monthly or daily data to quarterly (average or last value in the quarter)
to_quarterly <- function(df, how = c("mean", "last")) {
  how <- match.arg(how)
  df$date <- quarter_end(df$date)
  f <- if (how == "mean") {
    function(x) mean(x, na.rm = TRUE)
  } else {
    function(x) if (all(is.na(x))) NA else tail(x[!is.na(x)], 1)
  }
  out <- aggregate(. ~ date, data = df, FUN = f, na.action = na.pass)
  out[] <- lapply(out, function(col) if (is.numeric(col)) replace(col, is.nan(col), NA) else col)
  out
}
