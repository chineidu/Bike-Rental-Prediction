from typing import Any, Literal, cast

import narwhals as nw
import narwhals.selectors as n_cs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pyarrow as pa
from narwhals.typing import IntoDataFrameT, IntoFrameT
from plotly.subplots import make_subplots
from scipy.stats import entropy, spearmanr


class ExploratoryDataAnalysis:
    def __init__(self, data: IntoDataFrameT, target_column: str | None = None) -> None:
        # Store the original data reference to check its type later
        self._original_data = data
        self.data = nw.from_native(data)
        self.target_column = target_column
        self.numeric_columns = self._get_numeric_columns()
        self.categorical_columns = self._get_categorical_columns()
        self.boolean_columns = self._get_boolean_columns()

    def _convert_to_native(self, df: pl.DataFrame) -> IntoDataFrameT:
        """Convert Polars DataFrame to the original dataframe type.

        Parameters
        ----------
        df : pl.DataFrame
            The Polars DataFrame to convert.

        Returns
        -------
        IntoDataFrameT
            The converted DataFrame in the same format as the original input data.

        """
        if isinstance(self._original_data, pd.DataFrame):
            return cast(IntoDataFrameT, df.to_pandas())

        if isinstance(self._original_data, pa.Table):
            return cast(IntoDataFrameT, df.to_arrow())

        return cast(IntoDataFrameT, df)

    def _get_numeric_columns(self) -> list[str]:
        """Get numeric columns from the DataFrame."""
        return self.data.select(n_cs.numeric()).columns

    def _get_categorical_columns(self) -> list[str]:
        """Get categorical columns from the DataFrame."""
        return self.data.select(n_cs.string()).columns

    def _get_boolean_columns(self) -> list[str]:
        """Get boolean columns from the DataFrame."""
        return self.data.select(n_cs.boolean()).columns

    @staticmethod
    def _select_valid_columns(
        actual_cols: list[str], selected_cols: list[str]
    ) -> list[str]:
        return list(set(actual_cols) & set(selected_cols))

    def _calculate_outliers_iqr(self, series: nw.Series) -> tuple[nw.Series, nw.Series]:
        """Calculate outliers using the Interquartile Range (IQR) method.

        The IQR method identifies outliers as data points that fall below Q1 - 1.5*IQR
        or above Q3 + 1.5*IQR, where Q1 is the 25th percentile, Q3 is the 75th percentile,
        and IQR = Q3 - Q1.

        Parameters
        ----------
        series : nw.Series
            The numeric series to analyze for outliers.

        Returns
        -------
        tuple[nw.Series, nw.Series]
            A tuple containing:
            - normal_points: Series with values within the normal range
            - outliers: Series with values identified as outliers

        Notes
        -----
        This method uses the "nearest" interpolation method for quantile calculations
        to ensure consistent results across different dataframe backends.
        """
        Q1: float = series.quantile(0.25, interpolation="nearest")
        Q3: float = series.quantile(0.75, interpolation="nearest")
        IQR: float = Q3 - Q1
        lower_bound: float = Q1 - 1.5 * IQR
        upper_bound: float = Q3 + 1.5 * IQR
        normal_points: nw.Series = series.filter(
            (series >= lower_bound) & (series <= upper_bound)
        )
        outliers: nw.Series = series.filter(
            (series < lower_bound) | (series > upper_bound)
        )

        return normal_points, outliers

    def _calculate_outliers_zscore(
        self, series: nw.Series, threshold: float = 3.0
    ) -> tuple[nw.Series, nw.Series]:
        """Calculate outliers using the Z-score method.

        The Z-score method identifies outliers as data points whose absolute Z-score
        exceeds the specified threshold. Z-score is calculated as (x - mean) / std.

        Parameters
        ----------
        series : nw.Series
            The numeric series to analyze for outliers.
        threshold : float, default 3.0
            The Z-score threshold above which points are considered outliers.
            Common values are 2.0 (95% confidence) or 3.0 (99.7% confidence).

        Returns
        -------
        tuple[nw.Series, nw.Series]
            A tuple containing:
            - normal_points: Series with Z-scores within the threshold
            - outliers: Series with Z-scores exceeding the threshold

        Notes
        -----
        This method assumes the data follows a normal distribution. For non-normal
        distributions, the IQR method may be more appropriate.
        """
        mean: float = series.mean()
        std: float = series.std()
        z_scores: nw.Series = (series - mean) / std
        normal_points: nw.Series = series.filter(z_scores.abs() <= threshold)
        outliers: nw.Series = series.filter(z_scores.abs() > threshold)

        return normal_points, outliers

    def correlation_analysis(
        self,
        method: Literal["pearson", "spearman"] = "pearson",
        **kwargs: Any,
    ) -> IntoFrameT:
        """Calculate correlation matrix for numeric columns."""

        if len(self.numeric_columns) < 2:
            raise ValueError(
                "At least two numeric columns are required for correlation analysis."
            )

        X: np.ndarray[Any, Any] = self.data.select(self.numeric_columns).to_numpy()

        if method == "pearson":
            matrix: np.ndarray = np.corrcoef(X, rowvar=False, **kwargs)

        else:  # spearman
            matrix = spearmanr(X, axis=0, **kwargs).correlation
            if np.isscalar(matrix):
                # Convert to a matrix
                matrix = np.array([[1.0, matrix], [matrix, 1.0]])

        result: pl.DataFrame = pl.DataFrame(matrix, schema=self.numeric_columns)
        return self._convert_to_native(result)

    def numeric_summary(self, columns: list[str] | None = None) -> IntoFrameT:
        """Get summary statistics for numeric columns."""
        columns = (
            self._select_valid_columns(self.numeric_columns, columns)
            if columns
            else self.numeric_columns
        )

        summary_stats: list[Any] = []

        for col in columns:
            series = self.data[col]

            if len(series) == 0:
                continue

            # Central tendency: mean, median and mode
            mean: float = series.mean().__round__(2)
            median: float = series.median().__round__(2)
            mode: list[float] = series.mode().to_list()

            # Spread: std, variance, range, iqr_value, min, max
            std: float = series.std().__round__(2)
            variance: float = series.var().__round__(2)
            data_range: float = series.max() - series.min()
            iqr_value: float = series.quantile(
                0.75, interpolation="nearest"
            ) - series.quantile(0.25, interpolation="nearest")
            min_value: float = series.min()
            max_value: float = series.max()

            # Distribution shape: skewness and kurtosis
            skewness: float = series.skew().__round__(2)
            kurtosis: float = series.kurtosis().__round__(2)

            # Others: count, missing_values, unique_values
            count: int = series.count()
            missing_values: int = series.is_null().sum()
            missing_pct: float = (missing_values / series.shape[0]).__round__(2)
            unique_values: int = series.n_unique()

            # Outliers
            _, outlier_series_iqr = self._calculate_outliers_iqr(series)
            outlier_count_iqr = outlier_series_iqr.count()

            _, outlier_series_zscore = self._calculate_outliers_zscore(series)
            outlier_count_zscore = outlier_series_zscore.count()

            summary_stats.append(
                {
                    "column": col,
                    "mean": mean,
                    "median": median,
                    "mode": mode,
                    "std": std,
                    "variance": variance,
                    "range": data_range,
                    "iqr_value": iqr_value,
                    "min": min_value,
                    "max": max_value,
                    "skewness": skewness,
                    "kurtosis": kurtosis,
                    "outlier_series_iqr": outlier_series_iqr.to_list(),
                    "outlier_count_iqr": outlier_count_iqr,
                    "outlier_series_zscore": outlier_series_zscore.to_list(),
                    "outlier_count_zscore": outlier_count_zscore,
                    "total_count": count,
                    "missing_values": missing_values,
                    "missing_pct": missing_pct,
                    "unique_values": unique_values,
                }
            )

        # Create summary as Polars DataFrame first
        summary_df: pl.DataFrame = pl.from_records(summary_stats)

        return self._convert_to_native(summary_df)

    def categorical_summary(
        self, columns: list[str] | None = None
    ) -> dict[str, IntoFrameT]:
        """Get summary statistics for categorical columns."""
        columns = (
            self._select_valid_columns(self.categorical_columns, columns)
            if columns
            else self.categorical_columns + self.boolean_columns
        )

        summary_stats: list[dict[str, Any]] = []

        for col in columns:
            series = self.data[col]

            if len(series) == 0:
                continue

            # Frequency counts and percentages
            value_counts = series.value_counts().to_numpy()

            # Basic stats: count, missing_values, missing_pct, unique_values
            count: int = series.count()
            missing_values: int = series.is_null().sum()
            missing_pct: float = (missing_values / series.shape[0] * 100).__round__(2)
            unique_values: int = series.n_unique()

            # Entropy (measure of uncertainty or randomness)
            non_null_series = series.drop_nulls()
            if len(non_null_series) > 0:
                vc_non_null = non_null_series.value_counts(sort=True, normalize=False)
                entropy_value: float = entropy(vc_non_null["count"], base=10).__round__(
                    2
                )
            else:
                entropy_value = 0.0

            summary_stats.append(
                {
                    "column": col,
                    "total_count": count,
                    "unique_values": unique_values,
                    "entropy": entropy_value,
                    "value_counts": value_counts,
                    "missing_values": missing_values,
                    "missing_pct": missing_pct,
                }
            )

        summary_df: pl.DataFrame = pl.from_records(summary_stats)

        return self._convert_to_native(summary_df)

    def group_analysis(
        self, groupby: str, numeric_cols: list[str] | None = None
    ) -> IntoFrameT:
        # Ensure the column is a valid cat column
        if groupby not in self.categorical_columns:
            raise ValueError(f"🚫 {groupby!r} must be a categorical variable")
        numeric_cols = (
            self._select_valid_columns(self.numeric_columns, numeric_cols)
            if numeric_cols
            else self.numeric_columns
        )
        if len(numeric_cols) == 0:
            raise ValueError("🚫 No valid numeric columns found.")

        return (
            self.data.select(numeric_cols + [groupby])
            .group_by(groupby)
            .agg(
                n_cs.numeric().count().name.suffix("_count"),
                n_cs.numeric().mean().round(2).name.suffix("_mean"),
                n_cs.numeric().median().round(2).name.suffix("_median"),
                n_cs.numeric().std().round(2).name.suffix("_std"),
                n_cs.numeric().min().name.suffix("_min"),
                n_cs.numeric().max().name.suffix("_max"),
            )
            .to_native()
        )

    def plot_numeric_distribution(
        self,
        columns: list[str] | None = None,
        plot_type: Literal["all", "histogram", "box", "violin"] = "all",
    ) -> go.Figure:
        columns = (
            self._select_valid_columns(self.numeric_columns, columns)
            if columns
            else self.numeric_columns
        )
        if not columns:
            print("🚫 No numeric columns available for plotting.")
            return go.Figure()

        n_cols: int = min(3, len(columns))
        n_rows: int = (len(columns) + n_cols - 1) // n_cols

        if plot_type == "all":
            fig = make_subplots(
                rows=3,
                cols=len(columns),
                subplot_titles=[f"{col} - Histogram" for col in columns]
                + [f"{col} - Box Plot" for col in columns]
                + [f"{col} - Violin Plot" for col in columns],
                vertical_spacing=0.1,
            )

            for i, col in enumerate(columns):
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=self.data[col], name=f"{col}_hist", showlegend=False
                    ),
                    row=1,
                    col=i + 1,
                )

                # Box plot
                fig.add_trace(
                    go.Box(y=self.data[col], name=f"{col}_box", showlegend=False),
                    row=2,
                    col=i + 1,
                )

                # Violin plot
                fig.add_trace(
                    go.Violin(y=self.data[col], name=f"{col}_violin", showlegend=False),
                    row=3,
                    col=i + 1,
                )

            fig.update_layout(
                height=900, title_text="Numeric Distributions - All Plot Types"
            )

        else:
            fig = make_subplots(
                rows=n_rows, cols=n_cols, subplot_titles=[f"{col}" for col in columns]
            )

            for i, col in enumerate(columns):
                row = i // n_cols + 1
                col_idx = i % n_cols + 1

                if plot_type == "histogram":
                    fig.add_trace(
                        go.Histogram(x=self.data[col], name=col, showlegend=False),
                        row=row,
                        col=col_idx,
                    )
                elif plot_type == "box":
                    fig.add_trace(
                        go.Box(y=self.data[col], name=col, showlegend=False),
                        row=row,
                        col=col_idx,
                    )
                elif plot_type == "violin":
                    fig.add_trace(
                        go.Violin(y=self.data[col], name=col, showlegend=False),
                        row=row,
                        col=col_idx,
                    )

            fig.update_layout(
                height=300 * n_rows,
                title_text=f"Numeric Distributions - {plot_type.title()}",
            )

        return fig

    def plot_categorical_distribution(
        self,
        columns: list[str] | None = None,
        plot_type: Literal["all", "bar", "pie"] = "all",
    ) -> go.Figure:
        columns = (
            self._select_valid_columns(self.categorical_columns, columns)
            if columns
            else self.categorical_columns
        )
        if not columns:
            print("🚫 No categorical columns available for plotting.")
            return go.Figure()

        if plot_type == "all":
            n_rows = len(columns)
            # Create proper subplot titles for alternating bar and pie charts
            subplot_titles = []
            for col in columns:
                subplot_titles.extend([f"{col} - Bar Chart", f"{col} - Pie Chart"])

            fig = make_subplots(
                rows=n_rows,
                cols=2,
                specs=[[{"type": "xy"}, {"type": "domain"}] for _ in range(n_rows)],
                subplot_titles=subplot_titles,
            )

            for i, col in enumerate(columns):
                value_counts = (
                    self.data[col]
                    .value_counts(sort=True, normalize=True)
                    .with_columns((nw.col("proportion") * 100).round(1))
                )

                # Bar chart
                fig.add_trace(
                    go.Bar(
                        x=value_counts[col].to_numpy(),
                        y=value_counts["proportion"].to_numpy(),
                        name=f"{col}_bar",
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=1,
                )

                # Pie chart
                fig.add_trace(
                    go.Pie(
                        labels=value_counts[col].to_numpy(),
                        values=value_counts["proportion"].to_numpy(),
                        name=f"{col}_pie",
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=2,
                )

            fig.update_layout(
                height=400 * len(columns),
                width=420 * len(columns),
                title_text="Categorical Distributions",
            )

        else:
            n_cols = min(3, len(columns))
            n_rows = (len(columns) + n_cols - 1) // n_cols

            if plot_type == "pie":
                specs = [
                    [{"type": "domain"} for _ in range(n_cols)] for _ in range(n_rows)
                ]
            else:
                specs = None

            fig = make_subplots(
                rows=n_rows,
                cols=n_cols,
                specs=specs,
                subplot_titles=[f"{col}" for col in columns],
            )

            for i, col in enumerate(columns):
                row = i // n_cols + 1
                col_idx = i % n_cols + 1
                value_counts = (
                    self.data[col]
                    .value_counts(sort=True, normalize=True)
                    .with_columns((nw.col("proportion") * 100).round(1))
                )

                if plot_type == "bar":
                    fig.add_trace(
                        go.Bar(
                            x=value_counts[col].to_numpy(),
                            y=value_counts["proportion"].to_numpy(),
                            name=col,
                            showlegend=False,
                        ),
                        row=row,
                        col=col_idx,
                    )
                elif plot_type == "pie":
                    fig.add_trace(
                        go.Pie(
                            labels=value_counts[col].to_numpy(),
                            values=value_counts["proportion"].to_numpy(),
                            name=col,
                            showlegend=False,
                        ),
                        row=row,
                        col=col_idx,
                    )

            fig.update_layout(
                height=300 * n_rows,
                title_text=f"Categorical Distributions - {plot_type.title()}",
            )

        return fig

    def plot_correlation_heatmap(
        self, method: Literal["pearson", "spearman"] = "pearson"
    ) -> go.Figure:
        corr_matrix: IntoFrameT = self.correlation_analysis(method=method)

        if len(corr_matrix) == 0:
            return go.Figure()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.to_numpy(),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr_matrix.to_numpy(), 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"{method.title()} Correlation Matrix",
            xaxis_title="Variables",
            yaxis_title="Variables",
        )

        return fig

    def plot_outliers(
        self, columns: list[str] | None = None, method: Literal["iqr", "zscore"] = "iqr"
    ) -> go.Figure:
        columns = (
            self._select_valid_columns(self.numeric_columns, columns)
            if columns
            else self.numeric_columns
        )

        if not columns:
            print("No numeric columns to plot")
            return go.Figure()

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=[f"{col} - Outliers ({method})" for col in columns],
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
        )

        for i, col in enumerate(columns):
            row: int = i // n_cols + 1
            col_idx: int = i % n_cols + 1

            series = self.data[col].drop_nulls()

            if method == "iqr":
                normal_points, outliers = self._calculate_outliers_iqr(series)
            else:  # zscore
                normal_points, outliers = self._calculate_outliers_zscore(series)

            # Plot normal points
            fig.add_trace(
                go.Scatter(
                    x=np.arange(0, len(normal_points)),
                    y=normal_points.to_numpy(),
                    mode="markers",
                    name=f"{col}_normal",
                    marker={
                        "color": "lightblue",
                        "size": 2,
                        "opacity": 0.7,
                        "line": {"width": 1, "color": "blue"},
                    },
                    showlegend=False,
                ),
                row=row,
                col=col_idx,
            )

            # Plot outliers
            if len(outliers) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(0, len(outliers)),
                        y=outliers.to_numpy(),
                        mode="markers",
                        name=f"{col}_outliers",
                        marker={
                            "color": "red",
                            "size": 4,
                            "symbol": "diamond",
                            "line": {"width": 2, "color": "darkred"},
                        },
                        showlegend=False,
                    ),
                    row=row,
                    col=col_idx,
                )

        fig.update_layout(
            height=300 * n_rows, width=400 * n_rows, title_text="Outliers Detection"
        )
        return fig
