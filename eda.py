import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import entropy

warnings.filterwarnings("ignore")


class EDA:
    """
    Comprehensive Exploratory Data Analysis class for any dataset.

    This class provides methods for analyzing both numeric and categorical data,
    including statistical summaries, visualizations, and cross-data type analysis.
    """

    def __init__(self, data: pd.DataFrame, target_column: str | None = None) -> None:
        """
        Parameters
        ----------
        data : pd.DataFrame
            The dataset to analyze.
        target_column : str, optional
            Target column name for supervised learning analysis. Defaults to None.

        Attributes
        ----------
        data : pd.DataFrame
            A copy of the input dataset.
        target_column : str or None
            The target column name.
        numeric_columns : list
            List of numeric column names in the dataset.
        categorical_columns : list
            List of categorical column names in the dataset.
        """
        self.data = data.copy()
        self.target_column = target_column
        self.numeric_columns = self._get_numeric_columns()
        self.categorical_columns = self._get_categorical_columns()

    def _get_numeric_columns(self) -> list[str]:
        """Get list of numeric columns."""
        return self.data.select_dtypes(include=[np.number]).columns.tolist()

    def _get_categorical_columns(self) -> list[str]:
        """Get list of categorical columns."""
        return self.data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    def _calculate_outliers_iqr(
        self, series: pd.Series
    ) -> tuple[np.ndarray, float, float]:
        """
        Calculate outliers using IQR method.

        Args:
            series (pd.Series): Numeric series to analyze

        Returns:
            Tuple[np.ndarray, float, float]: (outlier_mask, lower_bound, upper_bound)
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        return outlier_mask, lower_bound, upper_bound

    def _calculate_outliers_zscore(
        self, series: pd.Series, threshold: float = 3
    ) -> np.ndarray:
        """
        Calculate outliers using Z-score method.

        Args:
            series (pd.Series): Numeric series to analyze
            threshold (float): Z-score threshold for outlier detection

        Returns:
            np.ndarray: Boolean mask for outliers
        """
        z_scores = np.abs(stats.zscore(series.dropna()))
        return z_scores > threshold

    def numeric_summary(self, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Generate comprehensive summary statistics for numeric columns.

        Args:
            columns (List[str], optional): Specific columns to analyze

        Returns:
            pd.DataFrame: Summary statistics
        """
        if columns is None:
            columns = self.numeric_columns

        summary_stats = []

        for col in columns:
            if col not in self.numeric_columns:
                continue

            series = self.data[col].dropna()

            if len(series) == 0:
                continue

            # Central Tendency
            mean_val = series.mean()
            median_val = series.median()
            try:
                mode_val = series.mode().iloc[0] if not series.mode().empty else np.nan
            except ValueError:
                mode_val = np.nan

            # Spread
            std_val = series.std()
            var_val = series.var()
            range_val = series.max() - series.min()
            iqr_val = series.quantile(0.75) - series.quantile(0.25)

            # Distribution Shape
            skewness = series.skew()
            kurt = series.kurtosis()

            # Percentiles
            percentiles = series.quantile([0.25, 0.5, 0.75]).to_dict()

            # Other metrics
            min_val = series.min()
            max_val = series.max()
            missing_count = self.data[col].isna().sum()
            missing_pct = (missing_count / len(self.data)) * 100
            unique_count = series.nunique()

            # Outliers
            outliers_iqr, _, _ = self._calculate_outliers_iqr(series)
            outliers_zscore = self._calculate_outliers_zscore(series)
            outlier_count_iqr = outliers_iqr.sum()
            outlier_count_zscore = outliers_zscore.sum()

            stats_dict = {
                "Column": col,
                "Count": len(series),
                "Missing": missing_count,
                "Missing_Pct": missing_pct,
                "Unique": unique_count,
                "Mean": mean_val,
                "Median": median_val,
                "Mode": mode_val,
                "Std": std_val,
                "Variance": var_val,
                "Min": min_val,
                "Max": max_val,
                "Range": range_val,
                "IQR": iqr_val,
                "Q25": percentiles[0.25],
                "Q50": percentiles[0.5],
                "Q75": percentiles[0.75],
                "Skewness": skewness,
                "Kurtosis": kurt,
                "Outliers_IQR": outlier_count_iqr,
                "Outliers_ZScore": outlier_count_zscore,
            }

            summary_stats.append(stats_dict)

        return pd.DataFrame(summary_stats)

    def categorical_summary(
        self, columns: list[str] | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Generate comprehensive summary statistics for categorical columns.

        Args:
            columns (List[str], optional): Specific columns to analyze

        Returns:
            Dict[str, pd.DataFrame]: Summary statistics for each categorical column
        """
        if columns is None:
            columns = self.categorical_columns

        summary_dict = {}

        for col in columns:
            if col not in self.categorical_columns:
                continue

            series = self.data[col]

            # Frequency analysis
            value_counts = series.value_counts()
            value_percentages = series.value_counts(normalize=True) * 100

            # Basic metrics
            total_count = len(series)
            missing_count = series.isna().sum()
            missing_pct = (missing_count / total_count) * 100
            unique_count = series.nunique()

            # Mode (most frequent category)
            mode_val = value_counts.index[0] if not value_counts.empty else None
            mode_count = value_counts.iloc[0] if not value_counts.empty else 0
            mode_pct = value_percentages.iloc[0] if not value_percentages.empty else 0

            # Entropy (measure of disorder/uncertainty)
            non_null_series = series.dropna()
            if len(non_null_series) > 0:
                entropy_val = entropy(non_null_series.value_counts())
            else:
                entropy_val = 0

            # Create summary DataFrame
            summary_df = pd.DataFrame(
                {
                    "Category": value_counts.index,
                    "Count": value_counts.values,
                    "Percentage": value_percentages.values,
                }
            )

            # Add metadata
            summary_df.attrs = {
                "column_name": col,
                "total_count": total_count,
                "missing_count": missing_count,
                "missing_pct": missing_pct,
                "unique_count": unique_count,
                "mode": mode_val,
                "mode_count": mode_count,
                "mode_pct": mode_pct,
                "entropy": entropy_val,
            }

            summary_dict[col] = summary_df

        return summary_dict

    def correlation_analysis(self, method: str = "pearson") -> pd.DataFrame:
        """
        Calculate correlation matrix for numeric columns.

        Args:
            method (str): Correlation method ("pearson", "spearman", "kendall")

        Returns:
            pd.DataFrame: Correlation matrix
        """
        if len(self.numeric_columns) < 2:
            print("Need at least 2 numeric columns for correlation analysis")
            return pd.DataFrame()

        return self.data[self.numeric_columns].corr(method=method)

    def group_analysis(
        self, group_by: str, numeric_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Calculate group-wise statistics for numeric columns.

        Args:
            group_by (str): Categorical column to group by
            numeric_cols (List[str], optional): Numeric columns to analyze

        Returns:
            pd.DataFrame: Group-wise statistics
        """
        if group_by not in self.categorical_columns:
            raise ValueError(f"'{group_by}' is not a categorical column")

        if numeric_cols is None:
            numeric_cols = self.numeric_columns

        # Filter to only existing numeric columns
        numeric_cols = [col for col in numeric_cols if col in self.numeric_columns]

        if not numeric_cols:
            print("No valid numeric columns found")
            return pd.DataFrame()

        return (
            self.data.groupby(group_by)[numeric_cols]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(3)
        )

    # Visualization Methods

    def plot_numeric_distribution(
        self, columns: list[str] | None = None, plot_type: str = "histogram"
    ) -> go.Figure:
        """
        Plot distribution of numeric columns.

        Args:
            columns (List[str], optional): Columns to plot
            plot_type (str): "histogram", "box", "violin", or "all"

        Returns:
            plotly.graph_objects.Figure: Distribution plot
        """
        if columns is None:
            columns = self.numeric_columns[:6]  # Limit to first 6 columns

        if not columns:
            print("No numeric columns to plot")
            return go.Figure()

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        if plot_type == "all":
            # Create subplots for histogram, box, and violin plots
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
        self, columns: list[str] | None = None, plot_type: str = "bar"
    ) -> go.Figure:
        """
        Plot distribution of categorical columns.

        Args:
            columns (List[str], optional): Columns to plot
            plot_type (str): "bar", "pie", or "both"

        Returns:
            plotly.graph_objects.Figure: Distribution plot
        """
        if columns is None:
            columns = self.categorical_columns[:6]  # Limit to first 6 columns

        if not columns:
            print("No categorical columns to plot")
            return go.Figure()

        if plot_type == "both":
            n_rows = len(columns)
            # Create proper subplot titles in the correct order (alternating bar and pie for each row)
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
                value_counts = self.data[col].value_counts()

                # Bar chart
                fig.add_trace(
                    go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        name=f"{col}_bar",
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=1,
                )

                # Pie chart
                fig.add_trace(
                    go.Pie(
                        labels=value_counts.index,
                        values=value_counts.values,
                        name=f"{col}_pie",
                        showlegend=False,
                    ),
                    row=i + 1,
                    col=2,
                )

            fig.update_layout(
                height=400 * len(columns), title_text="Categorical Distributions"
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
                value_counts = self.data[col].value_counts()

                if plot_type == "bar":
                    fig.add_trace(
                        go.Bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            name=col,
                            showlegend=False,
                        ),
                        row=row,
                        col=col_idx,
                    )
                elif plot_type == "pie":
                    fig.add_trace(
                        go.Pie(
                            labels=value_counts.index,
                            values=value_counts.values,
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

    def plot_correlation_heatmap(self, method: str = "pearson") -> go.Figure:
        """
        Plot correlation heatmap for numeric columns.

        Args:
            method (str): Correlation method

        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        corr_matrix = self.correlation_analysis(method=method)

        if corr_matrix.empty:
            return go.Figure()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr_matrix.values, 3),
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
        self,
        columns: list[str] | None = None,
        method: str = "iqr",
        plot_type: str = "box",
    ) -> go.Figure:
        """
        Plot outliers detection for numeric columns with multiple visualization options.

        Args:
            columns (List[str], optional): Columns to analyze. If None, uses first 6 numeric columns.
            method (str): "iqr" or "zscore" for outlier detection method
            plot_type (str): "box", "scatter", or "both" for visualization type

        Returns:
            plotly.graph_objects.Figure: Outliers plot
        """
        if columns is None:
            columns = self.numeric_columns[:6]

        if not columns:
            print("No numeric columns to plot")
            return go.Figure()

        # Validate columns exist and are numeric
        valid_columns = [col for col in columns if col in self.numeric_columns]
        if not valid_columns:
            print("No valid numeric columns found to plot")
            return go.Figure()

        columns = valid_columns
        n_cols = min(2, len(columns))  # Reduced from 3 to 2 for better width

        if plot_type == "both":
            # Double the rows for both plot types
            n_rows = ((len(columns) * 2) + n_cols - 1) // n_cols
            subplot_titles = []
            for col in columns:
                subplot_titles.extend([f"{col} - Box Plot", f"{col} - Scatter Plot"])
        else:
            n_rows = (len(columns) + n_cols - 1) // n_cols
            subplot_titles = [
                f"{col} - {plot_type.title()} ({method.upper()})" for col in columns
            ]

        fig = make_subplots(
            rows=n_rows,
            cols=n_cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.2,
        )

        for i, col in enumerate(columns):
            series = self.data[col].dropna()

            if len(series) == 0:
                continue

            # Calculate outliers based on method
            if method == "iqr":
                outlier_mask, lower_bound, upper_bound = self._calculate_outliers_iqr(
                    series
                )
                outliers = series[outlier_mask]
                normal_points = series[~outlier_mask]
            else:  # zscore
                outlier_mask = self._calculate_outliers_zscore(series)
                outliers = series[outlier_mask]
                normal_points = series[~outlier_mask]

            if plot_type in ["box", "both"]:
                # Calculate subplot position for box plot
                if plot_type == "both":
                    row = (i * 2) // n_cols + 1
                    col_idx = (i * 2) % n_cols + 1
                else:
                    row = i // n_cols + 1
                    col_idx = i % n_cols + 1

                # Add box plot
                fig.add_trace(
                    go.Box(
                        y=series.values,
                        name=f"{col}",
                        boxpoints="outliers",
                        marker={"color": "red", "size": 8},
                        line={"color": "blue"},
                        showlegend=False,
                    ),
                    row=row,
                    col=col_idx,
                )

            if plot_type in ["scatter", "both"]:
                # Calculate subplot position for scatter plot
                if plot_type == "both":
                    row = (i * 2 + 1) // n_cols + 1
                    col_idx = (i * 2 + 1) % n_cols + 1
                else:
                    row = i // n_cols + 1
                    col_idx = i % n_cols + 1

                # Plot normal points
                if len(normal_points) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=normal_points.index,
                            y=normal_points.values,
                            mode="markers",
                            name=f"{col} Normal" if i == 0 else "",
                            marker={"color": "blue", "size": 4, "opacity": 0.6},
                            showlegend=(i == 0),
                            legendgroup="normal",
                        ),
                        row=row,
                        col=col_idx,
                    )

                # Plot outliers
                if len(outliers) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=outliers.index,
                            y=outliers.values,
                            mode="markers",
                            name=f"{col} Outliers" if i == 0 else "",
                            marker={"color": "red", "size": 8, "symbol": "diamond"},
                            showlegend=(i == 0),
                            legendgroup="outliers",
                        ),
                        row=row,
                        col=col_idx,
                    )

        # Update layout with improved spacing and sizing
        height = (
            400 * n_rows if plot_type != "both" else 500 * n_rows
        )  # Increased height
        title_text = f"Outliers Detection - {plot_type.title()} Plot{'s' if plot_type == 'both' else ''} ({method.upper()})"

        fig.update_layout(
            height=height,
            width=1200,  # Set explicit width for better display
            title={
                "text": title_text,
                "x": 0.5,
                "font": {"size": 18, "color": "darkblue"},
            },
            showlegend=True if plot_type in ["scatter", "both"] else False,
            template="plotly_white",
            margin={"l": 80, "r": 80, "t": 100, "b": 80},  # Increased margins
        )

        return fig

    def plot_group_analysis(
        self, group_by: str, numeric_col: str, plot_type: str = "box"
    ) -> go.Figure:
        """
        Plot group-wise analysis of numeric column by categorical column.

        Args:
            group_by (str): Categorical column to group by
            numeric_col (str): Numeric column to analyze
            plot_type (str): "box", "violin", "bar", or "scatter"

        Returns:
            plotly.graph_objects.Figure: Group analysis plot
        """
        if group_by not in self.categorical_columns:
            raise ValueError(f"'{group_by}' is not a categorical column")

        if numeric_col not in self.numeric_columns:
            raise ValueError(f"'{numeric_col}' is not a numeric column")

        if plot_type == "box":
            fig = px.box(
                self.data,
                x=group_by,
                y=numeric_col,
                title=f"{numeric_col} by {group_by} - Box Plot",
            )
        elif plot_type == "violin":
            fig = px.violin(
                self.data,
                x=group_by,
                y=numeric_col,
                title=f"{numeric_col} by {group_by} - Violin Plot",
            )
        elif plot_type == "bar":
            grouped_data = self.data.groupby(group_by)[numeric_col].mean().reset_index()
            fig = px.bar(
                grouped_data,
                x=group_by,
                y=numeric_col,
                title=f"Average {numeric_col} by {group_by}",
            )
        elif plot_type == "scatter":
            # For scatter plot, we'll use the index as x-axis
            fig = px.scatter(
                self.data,
                x=group_by,
                y=numeric_col,
                title=f"{numeric_col} by {group_by} - Scatter Plot",
            )
        else:
            raise ValueError('plot_type must be "box", "violin", "bar", or "scatter"')

        return fig

    def generate_full_report(self) -> dict[str, Any]:
        """
        Generate a comprehensive EDA report.

        Returns:
            Dict[str, Any]: Complete analysis results
        """
        return {
            "dataset_info": {
                "shape": self.data.shape,
                "numeric_columns": len(self.numeric_columns),
                "categorical_columns": len(self.categorical_columns),
                "missing_values_total": self.data.isnull().sum().sum(),
                "memory_usage": self.data.memory_usage(deep=True).sum(),
            },
            "numeric_summary": self.numeric_summary(),
            "categorical_summary": self.categorical_summary(),
            "correlation_matrix": self.correlation_analysis(),
        }

    def print_summary(self) -> None:
        """Print a quick summary of the dataset."""
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"* Dataset Shape: {self.data.shape}")
        print(f"* Numeric Columns: {len(self.numeric_columns)}")
        print(f"* Categorical Columns: {len(self.categorical_columns)}")
        print(f"* Total Missing Values: {self.data.isnull().sum().sum()}")
        print(
            f"* Memory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

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


# Example usage and helper functions
def demo_eda(data: pd.DataFrame) -> None:
    """
    Demonstrate the EDA class capabilities.

    Args:
        data (pd.DataFrame): Sample dataset
    """
    eda = EDA(data)

    # Print summary
    eda.print_summary()

    # Generate numeric summary
    print("\nNumeric Summary:")
    print(eda.numeric_summary())

    # Generate categorical summary
    print("\nCategorical Summary:")
    cat_summary = eda.categorical_summary()
    for col, summary in cat_summary.items():
        print(f"\n{col}:")
        print(summary.head())

    # Correlation analysis
    print("\nCorrelation Matrix:")
    print(eda.correlation_analysis())

    # Create visualizations
    if eda.numeric_columns:
        fig_numeric = eda.plot_numeric_distribution(plot_type="histogram")
        fig_numeric.show()

    if eda.categorical_columns:
        fig_categorical = eda.plot_categorical_distribution(plot_type="bar")
        fig_categorical.show()

    if len(eda.numeric_columns) >= 2:
        fig_corr = eda.plot_correlation_heatmap()
        fig_corr.show()


if __name__ == "__main__":
    # Example with sample data
    rng = np.random.default_rng(42)
    sample_data = pd.DataFrame(
        {
            "numeric1": rng.normal(100, 15, 1000),
            "numeric2": rng.exponential(2, 1000),
            "numeric3": rng.uniform(0, 100, 1000),
            "category1": rng.choice(["A", "B", "C"], 1000),
            "category2": rng.choice(["High", "Medium", "Low"], 1000, p=[0.3, 0.5, 0.2]),
            "target": rng.normal(50, 10, 1000),
        }
    )

    # Add some missing values
    sample_data.loc[rng.choice(sample_data.index, 50, replace=False), "numeric1"] = (
        np.nan
    )
    sample_data.loc[rng.choice(sample_data.index, 30, replace=False), "category1"] = (
        np.nan
    )

    demo_eda(sample_data)
