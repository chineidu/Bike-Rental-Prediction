from typing import Any, Literal, cast

import narwhals as nw
import narwhals.selectors as n_cs
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pyarrow as pa
from IPython.display import display
from narwhals.typing import IntoDataFrameT, IntoFrameT
from plotly.subplots import make_subplots
from scipy.stats import entropy, spearmanr

from src.utilities.utils import _select_valid_columns, get_vibrant_color


class ExploratoryDataAnalysis:
    EMPTY_DATAFRAME: str = "🚫 Empty dataframe"
    NO_NUMERIC_COLUMNS: str = "🚫 No numeric columns available"
    NO_CATEGORICAL_COLUMNS: str = "🚫 No categorical columns available"

    def __init__(self, data: IntoDataFrameT, target_column: str | None = None) -> None:
        # Store the original data reference to check its type later
        self._original_data = data
        self.data = nw.from_native(data)
        self.target_column = target_column
        self.numeric_columns = self._get_numeric_columns()
        self.categorical_columns = self._get_categorical_columns()
        self.boolean_columns = self._get_boolean_columns()

    def _convert_to_native(self, df: pl.DataFrame | nw.DataFrame) -> IntoDataFrameT:
        """Convert Polars DataFrame or Narwhals DataFrame to the original dataframe type.

        Parameters
        ----------
        df : pl.DataFrame | nw.DataFrame
            The DataFrame to convert (either Polars or Narwhals).

        Returns
        -------
        IntoDataFrameT
            The converted DataFrame in the same format as the original input data.

        """
        # Handle Narwhals DataFrame
        if hasattr(df, "to_native"):  # Narwhals DataFrame
            df = df.to_native()  # type: ignore

        # Now convert to the original format
        if isinstance(self._original_data, pd.DataFrame):
            if isinstance(df, pl.DataFrame):
                return cast(IntoDataFrameT, df.to_pandas())
            return cast(IntoDataFrameT, df)

        if isinstance(self._original_data, pa.Table):
            if isinstance(df, pl.DataFrame):
                return cast(IntoDataFrameT, df.to_arrow())
            return cast(IntoDataFrameT, df)

        # Default: return as Polars DataFrame
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

    def _calculate_outliers_iqr(self, series: nw.Series) -> tuple[nw.Series, nw.Series]:
        """Calculate outliers using the Interquartile Range (IQR) method.

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
        """
        Calculate the correlation matrix for the numeric columns in the dataset.

        This method computes the correlation matrix using either Pearson or Spearman correlation
        for all numeric columns present in the data. It requires at least two numeric columns
        to perform the analysis.

        Parameters
        ----------
        method : Literal["pearson", "spearman"], default "pearson"
            The correlation method to use. "pearson" for linear correlation, "spearman" for
            rank-based correlation.
        **kwargs : Any
            Additional keyword arguments passed to the underlying correlation function
            (np.corrcoef for Pearson or scipy.stats.spearmanr for Spearman).

        Returns
        -------
        IntoFrameT
            A DataFrame containing the correlation matrix, with columns and index labeled
            by the numeric column names.

        Raises
        ------
        ValueError
            If there are fewer than two numeric columns in the dataset.
        """

        if len(self.numeric_columns) < 2:
            raise ValueError(
                "🚫 At least two numeric columns are required for correlation analysis."
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
            _select_valid_columns(self.numeric_columns, columns)
            if columns
            else self.numeric_columns
        )

        summary_stats: list[Any] = []

        for col in columns:
            series = self.data[col]

            if len(series) == 0:
                print(self.EMPTY_DATAFRAME)
                continue

            # Central tendency: mean, median and mode
            mean: float = series.mean().__round__(2)
            median: float = series.median().__round__(2)
            mode: list[float] = series.mode().to_list()[:5]  # Top 5 modes

            # Spread: std, variance, range, iqr_value, min, max
            std: float = series.std().__round__(2)
            variance: float = series.var().__round__(2)
            data_range: float = (series.max() - series.min()).__round__(2)
            iqr_value: float = (
                series.quantile(0.75, interpolation="nearest")
                - series.quantile(0.25, interpolation="nearest")
            ).__round__(2)
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
                    "unique_values": unique_values,
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
                }
            )

        # Create summary as Polars DataFrame first
        summary_df: pl.DataFrame = pl.from_records(summary_stats)

        return self._convert_to_native(summary_df)

    def categorical_summary(self, columns: list[str] | None = None) -> IntoFrameT:
        """Get summary statistics for categorical columns."""
        columns = (
            _select_valid_columns(self.categorical_columns, columns)
            if columns
            else self.categorical_columns + self.boolean_columns
        )

        summary_stats: list[dict[str, Any]] = []

        for col in columns:
            series = self.data[col]

            if len(series) == 0:
                print(self.EMPTY_DATAFRAME)
                continue

            # Frequency counts and percentages
            value_counts: list[list[Any]] = (
                series.value_counts(sort=True).to_numpy().tolist()
            )

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
        summary_df: pl.DataFrame = pl.from_records(summary_stats, strict=False)

        return self._convert_to_native(summary_df)

    def group_analysis(
        self, groupby: str, numeric_cols: list[str] | None = None
    ) -> IntoFrameT:
        # Ensure the column is a valid cat column
        """
        Perform group analysis on a categorical column by aggregating numeric columns.

        Parameters
        ----------
        groupby : str
            The name of the categorical column to group by. Must be a valid categorical
            variable in the dataset.
        numeric_cols : list[str] or None, optional
            List of numeric column names to include in the analysis. If None (default),
            all numeric columns are used.

        Returns
        -------
        IntoFrameT
            A DataFrame containing the grouped aggregations with columns suffixed by
            the statistic name (e.g., '_count', '_mean').

        Raises
        ------
        ValueError
            If `groupby` is not a categorical column or if no valid numeric columns
            are found.
        """
        if groupby not in self.categorical_columns:
            raise ValueError(f"🚫 {groupby!r} must be a categorical variable")
        numeric_cols = (
            _select_valid_columns(self.numeric_columns, numeric_cols)
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
        **kwargs: dict[str, Any],
    ) -> go.Figure:
        """
        Plot the distribution of numeric columns using specified plot types.

        Parameters
        ----------
        columns : list of str or None, optional
            List of column names to plot. If None, all numeric columns are used.
            Default is None.
        plot_type : {'all', 'histogram', 'box', 'violin'}, optional
            Type of plot to generate. If 'all', creates histograms, box plots, and
            violin plots for each column in a 3-row subplot layout. Otherwise,
            creates the specified plot type in a grid layout. Default is 'all'.

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure object containing the distribution plots.
        """
        columns = (
            _select_valid_columns(self.numeric_columns, columns)
            if columns
            else self.numeric_columns
        )
        if not columns:
            print(self.NO_NUMERIC_COLUMNS)
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
                height=kwargs.get("height", 900),
                width=kwargs.get("width", 400 * n_cols),
                title_text="Numeric Distributions - All Plot Types",
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
                width=400 * n_rows,
                title_text=f"Numeric Distributions - {plot_type.title()}",
            )

        return fig

    def plot_categorical_distribution(
        self,
        columns: list[str] | None = None,
        plot_type: Literal["all", "bar", "pie"] = "all",
        **kwargs: dict[str, Any],
    ) -> go.Figure:
        """
        Plot the distribution of categorical columns using bar and/or pie charts.

        Parameters
        ----------
        columns : list of str, optional
            List of categorical column names to plot. If None, all categorical columns
            are used. Only valid columns present in the data are selected.
        plot_type : {"all", "bar", "pie"}, default "all"
            Type of plot to generate. "all" creates both bar and pie charts in subplots,
            "bar" creates only bar charts, and "pie" creates only pie charts.

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly figure object containing the categorical distribution plots.
            If no categorical columns are available, an empty figure is returned.
        """
        columns = (
            _select_valid_columns(self.categorical_columns, columns)
            if columns
            else self.categorical_columns
        )
        if not columns:
            print(self.NO_CATEGORICAL_COLUMNS)
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
                        hovertemplate="%{y:.1f}%<extra></extra>",
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
                        hovertemplate="%{y:.1f}%<extra></extra>",
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=2,
                )

            fig.update_layout(
                height=kwargs.get("height", 300 * n_rows),
                width=kwargs.get("width", 420 * len(columns)),
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
                            hovertemplate="%{y:.1f}%<extra></extra>",
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
                            hovertemplate="%{y:.1f}%<extra></extra>",
                            showlegend=False,
                        ),
                        row=row,
                        col=col_idx,
                    )

            fig.update_layout(
                height=300 * n_rows,
                width=400 * n_rows,
                title_text=f"Categorical Distributions - {plot_type.title()}",
            )

        return fig

    def plot_correlation_heatmap(
        self,
        method: Literal["pearson", "spearman"] = "pearson",
        **kwargs: dict[str, Any],
    ) -> go.Figure:
        """
        Plot a correlation heatmap using Plotly.

        Parameters
        ----------
        method : {"pearson", "spearman"}, default "pearson"
            The correlation method to use. "pearson" for linear correlation,
            "spearman" for rank correlation.

        Returns
        -------
        go.Figure
            A Plotly figure object containing the correlation heatmap. If the
            correlation matrix is empty, an empty figure is returned.
        """
        corr_matrix: IntoFrameT = self.correlation_analysis(method=method)

        if isinstance(self._original_data, pd.DataFrame):
            corr_df = pl.from_pandas(corr_matrix)  # type: ignore

        if isinstance(self._original_data, pa.Table):
            print("🚫 Correlation heatmap is not fully supported for PyArrow Table.")
            corr_df: pl.DataFrame = pl.from_arrow(corr_matrix)  # type: ignore

        else:
            corr_df = corr_matrix

        if len(corr_df) == 0:  # type: ignore
            print(self.EMPTY_DATAFRAME)
            return go.Figure()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_df.to_numpy(),
                x=corr_df.columns,
                y=corr_df.columns,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr_df.to_numpy(), 3),
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            height=kwargs.get("height", 800) * 0.8,  # type: ignore
            width=kwargs.get("width", 800) * 0.8,  # type: ignore
            title=f"{method.title()} Correlation Matrix",
            xaxis_title="Variables",
            yaxis_title="Variables",
        )

        return fig

    def plot_correlation_with_target(self, save_path: str | None = None) -> go.Figure:
        """
        Plots an interactive correlation bar chart for all variables vs. target using Plotly.

        Parameters
        ----------
        save_path : str | None, optional
            Path to save the plot as HTML/PNG, by default None.

        Returns
        -------
        go.Figure
            The generated Plotly figure.
        """
        df = self._original_data
        target_column = self.target_column
        if target_column is None:
            raise ValueError(
                "🚫 target_column must be specified for correlation with target plot."
            )

        if isinstance(df, pd.DataFrame):
            df_polars: pl.DataFrame = pl.from_pandas(df)
        elif isinstance(df, pa.Table):
            df_polars = pl.from_arrow(df)
        else:
            df_polars = df  # type: ignore

        # Select only numeric columns
        df_polars = df_polars.select(self.numeric_columns)  # type: ignore

        # Compute correlations with target
        correlations: pl.DataFrame = (
            df_polars.corr()  # type: ignore
            .with_columns(pl.Series("features", df_polars.columns))  # type: ignore
            .select(["features", target_column])
            .sort(target_column, descending=True)
            .filter(pl.col(target_column) != 1.0)
        )

        # Generate colors for each correlation
        colors = [
            get_vibrant_color(corr) for corr in correlations[target_column].to_list()
        ]

        # Create figure
        fig = go.Figure()

        # Add bars with vibrant colors
        for i, (var, corr) in enumerate(
            zip(correlations["features"], correlations[target_column])
        ):
            # Add subtle shadow effect with opacity
            fig.add_trace(
                go.Bar(
                    x=[corr],
                    y=[var],
                    orientation="h",
                    marker={
                        "color": colors[i],
                        "line": {"color": "white", "width": 1},
                        "opacity": 0.9,
                    },
                    hovertemplate=f"<b>{var}</b><br>Correlation: {corr:.3f}<extra></extra>",
                    width=0.8,
                    showlegend=False,
                )
            )

        # Update layout with enhanced styling
        fig.update_layout(
            title={
                "text": f"<b>Feature Correlation with {target_column.title()}</b>",
                "font": {"size": 26, "color": "#2E2E2E"},
                "x": 0.5,
                "xanchor": "center",
            },
            xaxis_title="<b>Correlation Coefficient</b>",
            yaxis_title="<b>Features</b>",
            template="plotly_white",
            height=700,
            width=1000,
            margin={"l": 180, "r": 80, "t": 100, "b": 80},
            hoverlabel={
                "font_size": 16,
                "font_color": "white",
                "bgcolor": "rgba(0,0,0,0.8)",
            },
            xaxis={
                "tickfont": {"size": 14, "color": "#2E2E2E"},
                "title_font": {"size": 18, "color": "#2E2E2E"},
                "gridcolor": "#E8E8E8",
                "gridwidth": 1,
                "range": [-1, 1],
                "zeroline": True,
                "zerolinecolor": "#666666",
                "zerolinewidth": 2,
            },
            yaxis={
                "tickfont": {"size": 14, "color": "#2E2E2E"},
                "title_font": {"size": 18, "color": "#2E2E2E"},
                "autorange": "reversed",
                "gridcolor": "#F5F5F5",
            },
            plot_bgcolor="rgba(248,249,250,0.8)",
            paper_bgcolor="white",
        )

        # Add enhanced zero-line with annotation
        fig.add_vline(
            x=0,
            line_color="#666666",
            line_dash="dash",
            line_width=2,
            opacity=0.8,
            annotation_text="Zero Correlation",
            annotation_position="top left",
            annotation_font={"size": 8, "color": "#666666"},
        )

        # Add correlation strength indicators
        fig.add_vline(
            x=0.5, line_color="#00AA00", line_dash="dot", line_width=1, opacity=0.5
        )
        fig.add_vline(
            x=-0.5, line_color="#AA0000", line_dash="dot", line_width=1, opacity=0.5
        )

        # Save if path provided
        if save_path:
            if save_path.endswith(".html"):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path, scale=2)

        return fig

    def plot_outliers(
        self,
        columns: list[str] | None = None,
        method: Literal["iqr", "zscore"] = "iqr",
        **kwargs: dict[str, Any],
    ) -> go.Figure:
        """Plot outliers in specified numeric columns using IQR or Z-score method.

        Parameters
        ----------
        columns : list of str, optional
            List of column names to plot. If None, uses all numeric columns.
            Only valid numeric columns are selected.
        method : {"iqr", "zscore"}, default "iqr"
            Method to detect outliers. "iqr" uses Interquartile Range, "zscore" uses Z-score.

        Returns
        -------
        go.Figure
            A Plotly figure object containing subplots for each column's outlier plot.
            If no valid columns are found, returns an empty figure.
        """
        columns = (
            _select_valid_columns(self.numeric_columns, columns)
            if columns
            else self.numeric_columns
        )

        if not columns:
            print(self.NO_CATEGORICAL_COLUMNS)
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
            height=kwargs.get("height", 300 * n_rows),
            width=kwargs.get("width", 450 * n_rows),
            title_text="Outliers Detection",
        )
        return fig

    def plot_group_analysis(
        self,
        groupby: str,
        numeric_col: str,
        sortby: str | None = None,
        plot_type: Literal["bar", "box", "scatter", "violin"] = "bar",
    ) -> go.Figure:
        """
        Generate a Plotly figure for group-based analysis of a numeric column.

        Parameters
        ----------
        groupby : str
            The name of the categorical column to group by. Must be present in
            the dataset's categorical columns.
        numeric_col : str
            The name of the numeric column to analyze. Must be present in the
            dataset's numeric columns.
        sortby: str, default=None
            The name of the numeric column to use for sorting
        plot_type : {"bar", "box", "scatter", "violin"}, optional
            The type of plot to generate. Default is "bar".
            - "bar": Bar plot of the mean of the numeric column per group.
            - "box": Box plot of the numeric column distribution per group.
            - "scatter": Scatter plot with groupby on x-axis and numeric_col on y-axis.
            - "violin": Violin plot of the numeric column distribution per group.

        Returns
        -------
        go.Figure
            A Plotly figure object containing the generated plot.

        Raises
        ------
        ValueError
            If `groupby` is not a categorical column or `numeric_col` is not a
            numeric column, or if `plot_type` is not one of the allowed values.
        """

        data = self.data.to_polars()
        if groupby not in self.categorical_columns:
            raise ValueError(f"🚫 {groupby!r} is not a categorical column")

        if numeric_col not in self.numeric_columns:
            raise ValueError(f"🚫 {numeric_col!r} is not a numeric column")

        if sortby is not None and sortby not in self.numeric_columns:
            raise ValueError(f"🚫 {sortby!r} is not a numeric column")

        if plot_type == "box":
            fig = px.box(
                data,
                x=groupby,
                y=numeric_col,
                title=f"{numeric_col} by {groupby} - Box Plot",
            )
        elif plot_type == "violin":
            fig = px.violin(
                data,
                x=groupby,
                y=numeric_col,
                title=f"{numeric_col} by {groupby} - Violin Plot",
            )
        elif plot_type == "bar":
            grouped_data = self.data.group_by(groupby).agg(n_cs.numeric().mean())
            if sortby:
                grouped_data = grouped_data.sort(sortby, descending=True)

            grouped_data = self._convert_to_native(grouped_data)
            fig = px.bar(
                grouped_data,
                x=groupby,
                y=numeric_col,
                title=f"Average {numeric_col} by {groupby}",
            )
        elif plot_type == "scatter":
            # For scatter plot, we'll use the index as x-axis
            fig = px.scatter(
                data,
                x=groupby,
                y=numeric_col,
                title=f"{numeric_col} by {groupby} - Scatter Plot",
            )
        else:
            raise ValueError(
                '🚫 plot_type must be "box", "violin", "bar", or "scatter"'
            )

        return fig

    def plot_scatter(
        self,
        x_col: str,
        y_col: str,
        color_col: str | None = None,
        size_col: str | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """
        Create a scatter plot for two numeric columns with optional color and size encoding.

        Parameters
        ----------
        x_col : str
            Name of the column to plot on the x-axis. Must be a numeric column.
        y_col : str
            Name of the column to plot on the y-axis. Must be a numeric column.
        color_col : str or None, optional
            Name of the column to use for color encoding. Can be categorical, boolean, or numeric.
            If None, points will not be colored by a column. Default is None.
        size_col : str or None, optional
            Name of the numeric column to use for point size encoding. If None, points will use
            a constant size. Default is None.
        **kwargs : Any
            Additional keyword arguments forwarded to Plotly Express (for example: height, width,
            trendline, opacity, etc.).

        Returns
        -------
        plotly.graph_objects.Figure
            A Plotly Figure object containing the generated scatter plot.

        Raises
        ------
        ValueError
            If `x_col` or `y_col` are not numeric columns, if `color_col` is not a valid column
            for color encoding, or if `size_col` is not a numeric column.

        Examples
        --------
        >>> fig = eda.plot_scatter('temp', 'count', color_col='season', size_col='windspeed', trendline='ols')
        >>> fig.show()
        """
        if x_col not in self.numeric_columns:
            raise ValueError(f"🚫 {x_col!r} is not a numeric column")

        if y_col not in self.numeric_columns:
            raise ValueError(f"🚫 {y_col!r} is not a numeric column")

        if (
            color_col is not None
            and color_col
            not in self.categorical_columns
            + self.boolean_columns
            + self.numeric_columns
        ):
            raise ValueError(
                f"🚫 {color_col!r} is not a valid column for color encoding"
            )

        if size_col is not None and size_col not in self.numeric_columns:
            raise ValueError(f"🚫 {size_col!r} is not a numeric column")

        # Convert to native format for Plotly
        data = self._convert_to_native(self.data)

        # Create a copy of kwargs without title and trendline if they exist
        plot_kwargs: dict[str, Any] = kwargs.copy()
        plot_kwargs.pop("title", None)  # Remove title from kwargs if present
        plot_kwargs.pop("trendline", None)  # Remove trendline from kwargs if present

        # Create the scatter plot
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=f"Scatter Plot: {x_col} vs {y_col}",
            opacity=0.6,
            labels={
                x_col: x_col.replace("_", " ").title(),
                y_col: y_col.replace("_", " ").title(),
                color_col: color_col.replace("_", " ").title() if color_col else None,
                size_col: size_col.replace("_", " ").title() if size_col else None,
            },
            **plot_kwargs,
        )

        # Update layout
        fig.update_layout(
            height=kwargs.get("height", 600),
            width=kwargs.get("width", 800),
            xaxis_title=x_col.replace("_", " ").title(),
            yaxis_title=y_col.replace("_", " ").title(),
        )

        # Add trend line if requested
        if kwargs.get("trendline", False):
            # For trendline, we need to use plotly express with trendline parameter
            fig = px.scatter(
                data,
                x=x_col,
                y=y_col,
                color=color_col,
                size=size_col,
                trendline=kwargs.get("trendline"),
                title=f"Scatter Plot: {x_col} vs {y_col} with Trendline",
                **plot_kwargs,
            )

        return fig

    def generate_full_report(self) -> dict[str, Any]:
        """Generate a comprehensive report summarizing the dataset.

        This method compiles various statistics and analyses into a single
        dictionary, including dataset information, summaries for numeric and
        categorical columns, and a correlation matrix.

        Returns
        -------
        dict[str, Any]
            A dictionary with the following keys:
            - 'dataset_info': dict containing shape, column counts, row count,
              total missing values, and memory usage.
            - 'numeric_summary': summary statistics for numeric columns.
            - 'categorical_summary': summary statistics for categorical columns.
            - 'correlation_matrix': correlation analysis results.
        """
        return {
            "dataset_info": {
                "shape": self.data.shape,
                "numeric_columns": len(self.numeric_columns),
                "categorical_columns": len(self.categorical_columns),
                "boolean_columns": len(self.boolean_columns),
                "total_columns": self.data.shape[1],
                "total_rows": self.data.shape[0],
                "missing_values_total": self.data.null_count().to_numpy().sum().item(),
                "memory_usage": f"{round(self.data.estimated_size(unit='mb'), 2)} MB",
            },
            "numeric_summary": self.numeric_summary(),
            "categorical_summary": self.categorical_summary(),
            "correlation_matrix": self.correlation_analysis(),
        }

    def display_all_plots(
        self,
        outlier_method: Literal["iqr", "zscore"] = "iqr",
        numeric_cols: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Display all exploratory data analysis plots.

        This method generates and displays four key visualizations for the dataset:
        - Numeric column distributions
        - Categorical column distributions
        - Correlation heatmap for numeric columns
        - Outlier detection plots using the specified method

        Parameters
        ----------
        outlier_method : {"iqr", "zscore"}, default "iqr"
            The method to use for outlier detection. Options are "iqr" (Interquartile Range)
            or "zscore" (Z-Score).
        numeric_cols : list of str or None, default None
            List of numeric column names to include in the analysis. If None, all numeric
            columns from the dataset are used.

        Returns
        -------
        None
            This method does not return any value; it displays plots directly.
        """
        numeric_cols = (
            _select_valid_columns(self.numeric_columns, numeric_cols)
            if numeric_cols
            else self.numeric_columns
        )

        # Create visualizations
        fig1 = self.plot_numeric_distribution(**kwargs)
        fig1.show()

        fig2 = self.plot_categorical_distribution(**kwargs)
        fig2.show()

        fig3 = self.plot_correlation_heatmap(**kwargs)
        fig3.show()

        fig4 = self.plot_outliers(method=outlier_method, **kwargs)
        fig4.show()

    def print_summary(self) -> None:
        """Print a summary of the dataset's key statistics and structure.

        This method provides an overview of the dataset, including its shape,
        column types, missing values, memory usage, and lists of numeric,
        categorical, and boolean columns. If a target column is specified,
        it is also displayed.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        print("=" * 60)
        print("🚀 EXPLORATORY DATA ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"* Dataset Shape: {self.data.shape}")
        print(f"* Total Rows: {self.data.shape[0]}")
        print(f"* Total Columns: {self.data.shape[1]}")
        print(f"* Numeric Columns: {len(self.numeric_columns)}")
        print(f"* Categorical Columns: {len(self.categorical_columns)}")
        print(f"* Boolean Columns: {len(self.boolean_columns)}")
        print(
            f"* Total Missing Values: {self.data.null_count().to_numpy().sum().item()}"
        )
        print(f"* Memory Usage: {round(self.data.estimated_size(unit='mb'), 2)}  MB")

        if self.target_column:
            print(f"* Target Column: {self.target_column}")

        print("\n* Numeric Columns:")
        for col in self.numeric_columns:
            print(f"  - {col}")

        print("\n* Categorical Columns:")
        for col in self.categorical_columns:
            print(f"  - {col}")

        print("=" * 60)
        print()

        # Get numeric statistics
        print("\n 📈 Numeric Statistics:")
        print("--" * 12)
        numeric_stats = self.numeric_summary()
        display(numeric_stats)

        # Get categorical statistics
        print("\n 📈 Categorical Statistics:")
        print("--" * 14)
        cat_stats = self.categorical_summary()

        display(cat_stats)
