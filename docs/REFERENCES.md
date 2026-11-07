# Related work and data provenance

## Peer-reviewed research

1. Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency. *Journal of Finance, 48*(1), 65–91. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x
2. Daniel, K., & Moskowitz, T. J. (2016). Momentum Crashes. *Journal of Financial Economics, 122*(2), 221–247. https://doi.org/10.1016/j.jfineco.2015.12.002
3. Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via Machine Learning. *Review of Financial Studies, 33*(5), 2223–2273. https://doi.org/10.1093/rfs/hhaa009

These support the background and methodological motivation; they are not direct empirical benchmarks for our custom event labels.

## Data providers

- Uploaded daily stock CSVs for NVDA, AMD, AVGO, MSFT, GOOG, AMZN, META, AAPL, TSLA, INTC; SPY, QQQ, DIA. Historical dates ~2018-01 to 2026-11. User-supplied originally from Yahoo Finance-style adjusted OHLCV exports: https://finance.yahoo.com/
- Federal Reserve Bank of St. Louis, FRED VIXCLS: https://fred.stlouisfed.org/series/VIXCLS
- Kenneth French Data Library (Momentum, Short-Term Reversal, Long-Term Reversal): https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/data_library.html
- FINRA Consolidated NMS Daily Short Sale Volume Files: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files
- FINRA Query API: https://developer.finra.org/docs (authentication and entitlement required). Its consolidated short interest endpoint may apply to OTC stocks; do not assume listed-stock coverage.
- SEC Form 13F Data Sets: https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets
- Nasdaq Data Link NSHRT: https://data.nasdaq.com/databases/NSHRT (premium, not included in package)

## Important data-use and empirical caveats

- The user-provided stock and factor CSVs are copies for the research project. Check each source's terms before publishing their contents in a public GitHub repository.
- Past short-sale *volume* cannot be substituted for short-interest positions. No historical short-interest percentages have been invented.
- 13F summaries are descriptive only, with amendments, duplicated manager reporting, and incomplete accession-to-filing joins in user-supplied files. Never use unverified aggregate shares as true percent institutional ownership.
- Point-in-time assumptions: end-of-day adjusted close and daily market data available after the session; no live prediction before market close.
- Research labels use future prices and are strictly target variables, not predictors.
