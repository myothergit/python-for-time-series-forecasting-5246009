---
excerpt: We explain how to process multiple financial assets and include them in a report highlighting each asset's annual performance.
id_slug: C-BILELLO-1
slug: processing-annual-financial-reports-on-multiple-assets-in-python
tags:  
  ghost:
    - Finance
    - Descriptive Analysis
    - Pandas
    - Tutorial
title: Processing annual financial reports on multiple assets in Python
---

When comparing the performance of financial assets like Bitcoin, ETFs, and stocks, which was the most profitable asset since 2010?

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_asset_performance.png" alt="Annual performance comparison of various financial assets since 2010">
    <figcaption>F1. Annual performance of financial assets since 2010</figcaption>
</figure>

In this tutorial, we explain how to download, process, and report on the performance of financial assets using Python.

The report is inspired by [Charlie Bilello's analyses](https://twitter.com/charliebilello/status/1643699214914822144).

{{ canonical }}

## Data

We define the list of tickers for the assets to analyze.

```python
tickers = ['BTC-USD', 'QQQ', 'IWF', 'SPY', ...]
```

We download daily historical price movements using the `yfinance` library, which downloads data from Yahoo Finance.

```python
import yfinance as yf
df = yf.download(tickers)
```

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_hist.png" alt="Daily historical price chart for Bitcoin, ETFs, and stocks">
    <figcaption>F2. Daily historical prices of financial assets</figcaption>
</figure>

## Questions

1. How to download historical price data for multiple financial assets?
2. How to calculate the annual cumulative return of each asset?
3. Why is it necessary to group data for cumulative calculations?
4. How to select the last day of cumulative return in each year?
5. How to identify the maximum and minimum return values in each year?
6. How to calculate the percentage of positive returns for each asset?

## Methodology

### Annual cumulative return

We select the adjusted closing prices `Adj Close` from 2010 and calculate the annual cumulative return, choosing the last business day of each year.

```python
(df
 .loc['2010':, 'Adj Close']
 .groupby(df.index.year).pct_change().add(1)
 .groupby(df.index.year).cumprod().sub(1)
 .resample('YE').last().T
)
```

`BTC-USD` (Bitcoin) shows missing values in the early years due to its regulation and adoption (see the full story on [Wikipedia](https://en.wikipedia.org/wiki/History_of_bitcoin#2014)).

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_etf_returns.png" alt="Annual cumulative returns for ETFs, showing missing early years for Bitcoin">
    <figcaption>F3. Annual cumulative returns of ETFs</figcaption>
</figure>

### Column summary

How did the assets perform over the years?

We calculate the average annual return and the annual cumulative return for each asset.

> For more details on the calculation of returns, visit [this tutorial](https://datons.ai/preprocess-and-analyze-stock-returns-with-python/).

```python
t_avg = df.mean(axis=1).mul(100)
t_cum = df.add(1).cumprod(axis=1).sub(1).mul(100).iloc[:,[-1]]

pd.DataFrame({'AVG': t_avg, 'CAGR': t_cum})
```

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_etf_summary.png" alt="Summary of average annual return and cumulative return for various financial assets">
    <figcaption>F4. Return summary of financial assets</figcaption>
</figure>

### Row summary

What were the maximum and minimum values each year? What was the percentage of positive returns?

```python
positive_pct = lambda x: (x > 0).mean() * 100
dfr.agg(['max', 'min', positive_pct])
```

All assets showed positive cumulative returns at the end of the period, with `BTC-USD` being the most profitable asset.

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_etf_summary_rows.png" alt="Annual performance summary showing maximum, minimum values, and percentage of positive returns for financial assets">
    <figcaption>F5. Annual performance summary of financial assets</figcaption>
</figure>

### Combining and styling

Finally, we combine all tables and apply styling to highlight maximum and minimum values.

> This tutorial explains how to apply styles to pandas tables to create a heat matrix [this tutorial](https://datons.ai/style-pandas-pivot-table-to-create-heat-matrix/).

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_asset_performance.png" alt="Annual performance comparison of various financial assets since 2010">
    <figcaption>F1. Annual performance of financial assets since 2010</figcaption>
</figure>

Why did almost all assets show negative returns in 2022?

Looking forward to your comments.

## Conclusions

1. **Downloading historical data:** `yf.download(tickers)` to get data for multiple financial assets from Yahoo Finance.
2. **Calculating annual cumulative return:** `.pct_change().add(1).cumprod().sub(1)` to determine an asset's performance over time.
3. **Grouping data for cumulative calculations:** `groupby.cumprod` to reset cumulative calculations at the start of each year.
4. **Selecting the last day of cumulative return:** `.resample('Y').last()` to find the final value of each year, useful for annual analysis.
5. **Identifying maximum and minimum values:** `.agg(['max', 'min'])` to discover the extremes in annual asset performance.
6. **Calculating the percentage of positive returns:** `lambda x: (x > 0).mean() * 100` to evaluate the frequency of an asset's gains.

{{ cta }}