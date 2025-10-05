from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src import PACKAGE_PATH, create_logger

logger = create_logger(name="visualization")


def create_grouped_metrics_barchart(
    data: pd.DataFrame | pl.DataFrame,
    model_col: str = "model_name",
    metrics: list[str] | None = None,
    title: str = "Model Metrics Comparison (Lower is Better)",
    save_path: str | Path | None = None,
    height: int = 600,
    width: int | None = None,
    show_values: bool = True,
) -> go.Figure:
    """
    Create a grouped bar chart comparing multiple metrics across different models.

    Parameters
    ----------
    data : pd.DataFrame | pl.DataFrame
        Input DataFrame containing model names and metric columns.
    model_col : str, default="model_name"
        Column name containing model names.
    metrics : list[str] | None, optional
        List of metric column names to plot. If None, uses all numeric columns
        except the model_col, by default None.
    title : str, optional
        Chart title, by default "Model Metrics Comparison".
    save_path : str | Path | None, optional
        Path to save the chart as HTML. If None, chart is not saved, by default None.
    height : int, optional
        Chart height in pixels, by default 600.
    width : int | None, optional
        Chart width in pixels. If None, uses Plotly default, by default None.
    show_values : bool, optional
        Whether to display values on bars, by default True.

    Returns
    -------
    go.Figure
        Plotly figure object containing the grouped bar chart.
    """
    # Convert Polars to Pandas if needed
    if isinstance(data, pl.DataFrame):
        data = data.to_pandas()

    # Auto-detect metrics if not provided
    if metrics is None:
        metrics = [
            col
            for col in data.columns
            if col != model_col and pd.api.types.is_numeric_dtype(data[col])
        ]

    # Generate colors for each metric
    import plotly.express as px

    color_scale = px.colors.qualitative.Plotly
    metric_colors = {
        metric: color_scale[i % len(color_scale)] for i, metric in enumerate(metrics)
    }

    # Create figure
    fig = go.Figure()

    # Add a bar for each metric
    for metric in metrics:
        fig.add_trace(
            go.Bar(
                name=metric,
                x=data[model_col],
                y=data[metric],
                marker_color=metric_colors[metric],
                text=data[metric] if show_values else None,
                texttemplate="%{text:.2f}" if show_values else None,
                textposition="outside" if show_values else None,
            )
        )

    # Update layout
    layout_kwargs: dict[str, Any] = {
        "title": {"text": f"<b>{title}</b>", "x": 0.5, "xanchor": "center"},
        "xaxis_title": "Model",
        "yaxis_title": "Metric Value",
        "barmode": "group",
        "height": height,
        "hovermode": "closest",
        "legend": {
            "title": {"text": "Metrics"},
            "orientation": "v",
            "yanchor": "top",
            "y": 1,
            "xanchor": "left",
            "x": 1.02,
        },
    }

    if width is not None:
        layout_kwargs["width"] = width

    fig.update_layout(**layout_kwargs)

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    # Save if path provided
    if save_path is not None:
        save_path = Path(PACKAGE_PATH / "reports" / save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Saved grouped metrics chart to {save_path}")

    return fig


def create_multi_model_residuals_plot(
    results: list[dict[str, Any]],
    y_true: np.ndarray | pd.Series | pl.Series,
    title: str = "Residual Comparison Across Models",
    save_path: str | Path | None = None,
    height: int = 600,
    width: int | None = None,
) -> go.Figure:
    """
    Create a comparative residual plot for multiple models.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of result dictionaries containing 'model_name' and 'predictions' keys.
    y_true : np.ndarray | pd.Series | pl.Series
        True target values.
    title : str, optional
        Chart title, by default "Residual Comparison Across Models".
    save_path : str | Path | None, optional
        Path to save the chart as HTML. If None, chart is not saved, by default None.
    height : int, optional
        Chart height in pixels, by default 600.
    width : int | None, optional
        Chart width in pixels. If None, uses Plotly default, by default None.

    Returns
    -------
    go.Figure
        Plotly figure object containing the multi-model residual plot.

    Examples
    --------
    >>> # Compare residuals from multiple models
    >>> results = [
    ...     {"model_name": "XGBoost", "predictions": xgb_preds},
    ...     {"model_name": "RandomForest", "predictions": rf_preds}
    ... ]
    >>> fig = create_multi_model_residuals_plot(
    ...     results=results,
    ...     y_true=y_test,
    ...     save_path="residuals_comparison.html"
    ... )

    Notes
    -----
    - Each model's residuals are plotted with a different color
    - Useful for comparing error patterns across different models
    - Models with smaller, more centered residuals generally perform better
    """
    # Convert y_true to numpy if needed
    if isinstance(y_true, (pd.Series, pl.Series)):
        y_true = y_true.to_numpy()

    # Generate colors for each model
    color_scale = px.colors.qualitative.Plotly
    model_colors = {
        result["model_name"]: color_scale[i % len(color_scale)]
        for i, result in enumerate(results)
    }

    # Create figure
    fig = go.Figure()

    # Add residuals for each model
    for result in results:
        model_name = result["model_name"]
        y_pred = result["predictions"]

        # Calculate residuals
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)

        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                name=model_name,
                marker={
                    "color": model_colors[model_name],
                    "size": 6,
                    "opacity": 0.5,
                    "line": {"width": 0.5, "color": "DarkSlateGrey"},
                },
                hovertemplate=f"<b>{model_name}</b><br>"
                + "<b>Predicted:</b> %{x:.2f}<br>"
                + "<b>Residual:</b> %{y:.2f}<br>"
                + f"<b>Mean Residual:</b> {mean_residual:.2f}<br>"
                + "<extra></extra>",
            )
        )

    # Add zero reference line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="black",
        line_width=2,
        annotation_text="Zero Line",
        annotation_position="right",
    )

    # Update layout
    layout_kwargs: dict[str, Any] = {
        "title": {"text": f"<b>{title}</b>", "x": 0.5, "xanchor": "center"},
        "xaxis_title": "Predicted Values",
        "yaxis_title": "Residuals (True - Predicted)",
        "height": height,
        "hovermode": "closest",
        "showlegend": True,
        "legend": {"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    }

    if width is not None:
        layout_kwargs["width"] = width

    fig.update_layout(**layout_kwargs)

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    # Save if path provided
    if save_path is not None:
        save_path = Path(PACKAGE_PATH / "reports" / save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path))
        logger.info(f"Saved multi-model residual plot to {save_path}")

    return fig
