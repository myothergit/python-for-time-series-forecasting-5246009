import plotly.io as pio
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def configure_plotly_template(showlegend=False, width=1000, height=500):
    pio.templates.default = "plotly_dark"
    pd.options.plotting.backend = "plotly"
    pio.templates["plotly_dark"].layout.update(
        width=width, height=height, showlegend=showlegend, autosize=False
    )


def add_time_features(df):
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday
    df["weekend"] = df.index.dayofweek > 4
    return df


import pandas as pd


def collect_lr_results(fit_results: dict) -> pd.DataFrame:
    rows = []
    for name, model in fit_results.items():
        summary = model.summary2().tables
        coef = summary[1].loc["x1", "Coef."]
        pval = summary[1].loc["x1", "P>|t|"]
        stderr = summary[1].loc["x1", "Std.Err."]
        tval = summary[1].loc["x1", "t"]
        r2 = model.rsquared
        nobs = int(model.nobs)
        rows.append(
            {
                "Period": name,
                "Coef": coef,
                "StdErr": stderr,
                "t": tval,
                "p": pval,
                "R²": r2,
                "n": nobs,
            }
        )
    return pd.DataFrame(rows).set_index("Period")


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def plot_residuals_histogram_with_normal_density(residuals):
    # Theoretical normal density
    x_vals = np.linspace(residuals.min(), residuals.max(), 100)
    y_vals = norm.pdf(x_vals, loc=0, scale=residuals.std())

    # Create figure
    plt.figure(figsize=(8, 6))

    # Add histogram
    plt.hist(
        residuals, bins=20, density=True, color="skyblue", alpha=0.75, label="Residuals"
    )

    # Add theoretical normal density
    plt.plot(x_vals, y_vals, color="red", linewidth=2, label="Normal Density")

    # Update layout
    plt.title("Histogram of Residuals with Normal Density Overlay")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
# plot_residuals_histogram_with_normal_density(df)
import matplotlib.pyplot as plt


def plot_residuals_lag(residuals):
    plt.scatter(residuals[:-1], residuals[1:])
    plt.xlabel("Residual t-1")
    plt.ylabel("Residual t")
    plt.title("Residuals vs Lagged Residuals")
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.show()


def plot_residuals_vs_fitted(fitted, residuals):
    plt.figure(figsize=(8, 4))
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={"color": "red"})

    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")

    plt.tight_layout()
    plt.show()


import statsmodels.api as sm


def linear_regression_by_category(
    df, category_column="period", target_column="CPI", feature_columns=["MR"]
):
    model_dict = {}
    for category in df[category_column].unique():
        df_filtered = df[df[category_column] == category].copy()
        df_filtered["intercept"] = 1

        X = df_filtered[["intercept"] + feature_columns]
        y = df_filtered[target_column]

        model = sm.OLS(y, X).fit()

        model_dict[category] = model

    return model_dict


import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def get_model_forecast(
    data_frame,
    column,
    order=(12, 1, 2),
    seasonal_order=None,
    horizon=48,
    column_name=None,
    forecast_exp=False,
    historical_predictions=True,
):
    df = data_frame.copy()
    series = df[column].dropna()
    p, d, q = order

    if seasonal_order is None:
        model = ARIMA(series, order=(p, d, q))
        model_name = f"ARIMA({p},{d},{q})"
    else:
        P, D, Q, m = seasonal_order
        model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, m))
        model_name = f"SARIMA({p},{d},{q})({P},{D},{Q},{m})"

    model_fit = model.fit()

    end = len(series) + horizon - 1
    if historical_predictions:
        forecast = model_fit.predict(start=series.index[0], end=end)
    else:
        forecast = model_fit.predict(start=len(series), end=end)

    if forecast_exp:
        forecast = np.exp(forecast)

    if column_name:
        model_name = column_name

    df_forecast = forecast.to_frame(name=model_name)
    df_combined = pd.concat([df, df_forecast], axis=1)

    return df_combined


def get_train_test_forecast(
    df_train,
    df_test,
    column,
    order=(12, 1, 2),
    seasonal_order=None,
    column_names=None,
    prediction_real=True,
    return_df=["Train", "Test"],
):
    train_series = df_train[column].dropna()
    test_series = df_test[column].dropna()
    p, d, q = order

    if seasonal_order is None:
        model = ARIMA(train_series, order=(p, d, q))
        model_name = f"ARIMA({p},{d},{q})"
    else:
        P, D, Q, m = seasonal_order
        model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, m))
        model_name = f"SARIMA({p},{d},{q})({P},{D},{Q},{m})"

    model_fit = model.fit()

    start, end = train_series.index[[0, -1]]
    forecast_train = model_fit.predict(start=start, end=end)

    start, end = test_series.index[[0, -1]]
    forecast_test = model_fit.predict(start=start, end=end)

    if column_names:
        model_name = column_names

    if prediction_real:
        forecast_train = np.exp(forecast_train)
        forecast_test = np.exp(forecast_test)

    df_train = pd.concat([df_train, forecast_train.to_frame(name=model_name)], axis=1)
    df_test = pd.concat([df_test, forecast_test.to_frame(name=model_name)], axis=1)

    dfs = {
        "Train": df_train,
        "Test": df_test,
    }

    if len(return_df) == 1:
        return dfs[return_df[0]]

    t = ()
    for r in return_df:
        t += (dfs[r],)

    return t


def preprocess_concat_diff(vtype, path):
    df = pd.read_parquet(path)
    df.columns = ["values"]
    df["vtype"] = vtype

    r = df["values"].diff().to_frame(name="values").assign(diff=True, vtype=vtype)
    df = pd.concat([df.assign(diff=False), r], axis=0)

    return df


def plot_residuals_histogram(residuals, bins=50):
    import matplotlib.pyplot as plt
    import numpy as np

    # Calculate histogram data
    counts, bin_edges = np.histogram(residuals, bins=bins, density=True)

    # Plot histogram of residuals using matplotlib
    plt.hist(
        bin_edges[:-1],
        bin_edges,
        weights=counts,
        alpha=0.6,
        color="g",
        label="Residuals",
    )

    # Plot the bell curve
    mu, std = residuals.mean(), residuals.std()
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * ((x - mu) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    plt.plot(x, p, "k", linewidth=2, label="Bell Curve")

    # Add mean and standard deviation lines
    plt.axvline(mu, color="r", linestyle="dashed", linewidth=1, label=f"Mean: {mu:.2f}")
    plt.axvline(
        mu + std,
        color="b",
        linestyle="dashed",
        linewidth=1,
        label=f"Std Dev: {std:.2f}",
    )
    plt.axvline(mu - std, color="b", linestyle="dashed", linewidth=1)

    plt.title("Histogram of Residuals with Bell Curve")
    plt.xlabel("Residuals")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


from statsmodels.tsa.api import ExponentialSmoothing


def get_model_forecast_exponential_smoothing(
    data_frame,
    column,
    horizon=48,
    column_name=None,
    class_config=None,
    fit_config=None,
):
    series = data_frame[column].dropna()

    # Default configurations
    class_config = class_config or {}
    fit_config = fit_config or {}

    # Create and fit the model
    model = ExponentialSmoothing(
        series,
        trend=class_config.get("trend"),
        seasonal=class_config.get("seasonal"),
        seasonal_periods=class_config.get("seasonal_periods"),
        damped_trend=class_config.get("damped_trend"),
    ).fit(
        smoothing_level=fit_config.get("smoothing_level"),
        smoothing_slope=fit_config.get("smoothing_slope"),
        smoothing_seasonal=fit_config.get("smoothing_seasonal"),
        damping_slope=fit_config.get("damping_slope"),
    )

    # Forecast and combine with original data
    forecast = model.forecast(steps=horizon)
    model_name = column_name or "ExponentialSmoothing"
    df_forecast = forecast.to_frame(name=model_name)

    return pd.concat([data_frame, df_forecast], axis=1)


import plotly.graph_objects as go


def plot_prophet_forecast(series, df_prophet, title=None):
    df_prophet = df_prophet.set_index("ds")
    series = series.to_frame(name="historical")
    df = pd.concat([series, df_prophet], axis=1)
    # Rename columns

    # Ensure no negative values in 'FC Lower'
    df["yhat_lower"] = np.where(df["yhat_lower"] < 0, 0, df["yhat_lower"])

    fig = go.Figure()

    # Add lines
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["historical"],
            name="Historical",
            mode="lines+markers",
            line=dict(color="#fa636e", width=1.5),
            marker=dict(color="#fa636e", size=5, symbol="circle"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["yhat"].round(2),
            name="Forecast",
            line=dict(color="#FFD700", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["yhat_upper"].round(2),
            name="Forecast Interval Upper Bound",
            line=dict(color="rgba(212, 212, 217, 0.0)", width=0),
            showlegend=True,
            fill=None,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["yhat_lower"].round(2),
            name="Forecast Interval Lower Bound",
            line=dict(color="rgba(212, 212, 217, 0.0)", width=0),
            fill="tonexty",
            fillcolor="rgba(212, 212, 217, 0.5)",
            showlegend=True,
        )
    )

    # Chart layout updates
    fig.update_layout(
        title=title,
        template="plotly_dark",  # Keep the dark background style
        hovermode="x unified",  # Update hover layout
        xaxis=dict(range=[df.index.min(), df_prophet.index.max()]),
    )

    fig.show()


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing
from prophet import Prophet


class TimeSeriesForecaster:
    def __init__(
        self,
        series=None,
        train=None,
        test=None,
        test_size=0.3,
        freq="ME",
        idx_offset=13,
    ):
        self.freq = freq
        self.idx_offset = idx_offset
        self.last_forecast_df = None  # Stores the last forecast DataFrame
        self.last_combined_df = (
            None  # Stores the last combined (historical + forecast) DataFrame
        )
        if series is not None:
            series = series.asfreq(self.freq)
            self.train, self.test = train_test_split(
                series, test_size=test_size, shuffle=False
            )
        elif train is not None and test is not None:
            self.train = train.asfreq(self.freq)
            self.test = test.asfreq(self.freq)
        else:
            raise ValueError("Provide either `series` or both `train` and `test`.")

    def sarima(self, model_params=None, forecast_exp=False, log_transform=False):
        """
        SARIMA forecast. If log_transform is True, log-transform the data before fitting and exponentiate the forecast.
        """
        model_params = model_params or {
            "order": (0, 1, 1),
            "seasonal_order": (0, 1, 1, 12),
        }
        train = np.log(self.train) if log_transform else self.train
        test = np.log(self.test) if log_transform else self.test
        model = SARIMAX(train, **model_params).fit()
        forecast_train = model.predict(start=train.index[0], end=train.index[-1])
        forecast_test = model.predict(start=test.index[0], end=test.index[-1])
        if log_transform or forecast_exp:
            forecast_train = np.exp(forecast_train)
            forecast_test = np.exp(forecast_test)
        return forecast_train, forecast_test

    def ets(self, model_params=None, log_transform=False):
        """
        Exponential Smoothing forecast. If log_transform is True, log-transform the data before fitting and exponentiate the forecast.
        """
        model_params = model_params or {
            "trend": "add",
            "seasonal": "mul",
            "seasonal_periods": 12,
        }
        train = np.log(self.train) if log_transform else self.train
        test = np.log(self.test) if log_transform else self.test
        model = ExponentialSmoothing(train, **model_params).fit()
        forecast_train = model.predict(start=train.index[0], end=train.index[-1])
        forecast_test = model.predict(start=test.index[0], end=test.index[-1])
        if log_transform:
            forecast_train = np.exp(forecast_train)
            forecast_test = np.exp(forecast_test)
        return forecast_train, forecast_test

    def prophet(self, model_params=None, forecast_exp=False, log_transform=False):
        """
        Prophet forecast. If log_transform is True, log-transform the data before fitting and exponentiate the forecast.
        """
        model_params = model_params or {
            "yearly_seasonality": True,
            "weekly_seasonality": False,
            "daily_seasonality": False,
            "seasonality_mode": "multiplicative",
        }
        train = np.log(self.train) if log_transform else self.train
        test = np.log(self.test) if log_transform else self.test
        df_train = pd.DataFrame({"ds": train.index, "y": train})
        model = Prophet(**model_params)
        model.fit(df_train)
        future = model.make_future_dataframe(periods=len(self.test), freq=self.freq)
        forecast = model.predict(future).set_index("ds").asfreq(self.freq)
        forecast_train = forecast.loc[train.index]["yhat"]
        forecast_test = forecast.loc[test.index]["yhat"]
        if log_transform or forecast_exp:
            forecast_train = np.exp(forecast_train)
            forecast_test = np.exp(forecast_test)
        return forecast_train, forecast_test

    def bulk_forecast(self, configs, metrics=None):
        """
        Run multiple models and return a DataFrame with forecasts and metrics.
        Stores the result in self.last_forecast_df and returns it.
        :param configs: dict of model configs
        :param metrics: dict of metric functions, e.g. {'rmse': root_mean_squared_error}
        :return: pd.DataFrame
        """
        results = []
        metrics = metrics or {}

        for model_name, config in configs.items():
            forecaster = getattr(self, model_name)
            f_train, f_test = forecaster(**config)
            forecast = {"train": f_train, "test": f_test}

            for split in ["train", "test"]:
                # Use idx_offset only for train split
                if split == "train":
                    idx = self.idx_offset
                else:
                    idx = 0
                data_real = getattr(self, split)[idx:]
                data_forecast = forecast[split][idx:]

                row = {
                    "model": model_name,
                    "split": split,
                    "values": data_forecast.values,
                    "datetime": data_forecast.index,
                }
                # Calculate all metrics
                for metric_name, metric_func in metrics.items():
                    row[metric_name] = metric_func(data_real, data_forecast)
                results.append(row)
        self.last_forecast_df = pd.DataFrame(results)
        return self.last_forecast_df

    def combine_with_historical(self, df_forecast=None, idx_offset=None):
        """
        Combine exploded forecast DataFrame with historical real data for both train and test splits.
        Stores the result in self.last_combined_df and returns it.
        :param df_forecast: DataFrame from bulk_forecast, exploded so each row is a datetime-value pair. If None, uses self.last_forecast_df.
        :param idx_offset: Optionally override self.idx_offset for train split
        :return: Combined DataFrame with columns: model, split, datetime, values
        """
        if df_forecast is None:
            df_forecast = getattr(self, "last_forecast_df", None)
            if df_forecast is None:
                raise ValueError(
                    "No forecast DataFrame provided or stored in the class."
                )
        idx_offset = self.idx_offset if idx_offset is None else idx_offset

        # Explode forecast DataFrame
        r = df_forecast.set_index(["model", "split"])[["datetime", "values"]]
        df_forecast_exploded = r.explode(["datetime", "values"])
        df_forecast_exploded = df_forecast_exploded.set_index(
            "datetime", append=True
        ).sort_index()

        # Prepare historical dataframes
        dfs = {"train": self.train, "test": self.test}

        for split in ["train", "test"]:
            idx = idx_offset if split == "train" else 0
            r = (
                dfs[split]
                .iloc[idx:]
                .to_frame(name="values")
                .reset_index()
                .rename(columns={dfs[split].index.name or "index": "datetime"})
                .assign(model="historical", split=split)
                .set_index(["model", "split", "datetime"])
            )
            df_forecast_exploded = pd.concat([df_forecast_exploded, r], axis=0)

        self.last_combined_df = df_forecast_exploded.reset_index()
        return self.last_combined_df


def get_model_forecast_prophet(
    df, column, column_name=None, mode="multiplicative", horizon=48, **kwargs
):
    r = df[[column]].dropna()

    df_prophet = pd.DataFrame(
        {
            "ds": r.index,
            "y": r[column].values,
        },
        index=r.index,
    )

    model = Prophet(seasonality_mode=mode, **kwargs)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=horizon, freq="ME")
    forecast = model.predict(future)

    if column_name is None:
        column_name = f"prophet_{mode}"

    df_forecast = forecast.set_index("ds")[["yhat"]].rename(
        columns={"yhat": column_name}
    )

    idx_forecast = df_forecast.index[-horizon:]
    df_forecast = df_forecast.loc[idx_forecast]

    return pd.concat([df, df_forecast], axis=1)


import pandas as pd
import statsmodels.api as sm
import plotly.express as px


def plot_decomposition_comparison(series: pd.Series, period: int = 12) -> px.line:
    """
    Plot seasonal decomposition (additive and multiplicative) of a given time series using Plotly.

    Parameters:
        series (pd.Series): A time series with a DatetimeIndex.
        period (int): Seasonal period (e.g., 12 for monthly data with yearly seasonality).

    Returns:
        plotly.graph_objects.Figure: The decomposition visualization.
    """
    dfs = {}

    for model in ["additive", "multiplicative"]:
        result = sm.tsa.seasonal_decompose(series, model=model, period=period)
        r = (
            series.to_frame(name="values")
            .assign(trend=result.trend, seasonal=result.seasonal, residual=result.resid)
            .dropna()
        )
        r["model_result"] = (
            r.trend + r.seasonal + r.residual
            if model == "additive"
            else r.trend * r.seasonal * r.residual
        )
        dfs[model] = r

    df_combined = pd.concat(dfs, axis=1).melt(ignore_index=False).reset_index()
    df_combined.columns = ["month", "model", "component", "value"]

    fig = px.line(
        data_frame=df_combined,
        x="month",
        y="value",
        color="component",
        facet_col="model",
        facet_row="component",
        width=1500,
        height=1000,
        facet_col_spacing=0.1,
    )
    fig.update_yaxes(matches=None)
    for attr in dir(fig.layout):
        if attr.startswith("yaxis"):
            axis = getattr(fig.layout, attr)
            if axis:
                axis.showticklabels = True
    return fig
