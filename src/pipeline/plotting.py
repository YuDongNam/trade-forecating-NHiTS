"""
Plotting utilities for visualization.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


def plot_forecast(
    dates: pd.Series,
    y_actual: pd.Series,
    y_pred: pd.Series,
    y_lower: Optional[pd.Series] = None,
    y_upper: Optional[pd.Series] = None,
    title: str = "Forecast",
    metrics_text: Optional[str] = None,
    train_split_date: Optional[pd.Timestamp] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
    show: bool = True
) -> plt.Figure:
    """
    Plot forecast with actual values and optional confidence intervals.
    
    Args:
        dates: Date series
        y_actual: Actual values
        y_pred: Predicted values
        y_lower: Lower bound of confidence interval (optional)
        y_upper: Upper bound of confidence interval (optional)
        title: Plot title
        metrics_text: Text to display in title (e.g., "RMSE: ..., MAE: ...")
        train_split_date: Date to draw vertical line for train/val split
        save_path: Path to save figure (optional)
        figsize: Figure size
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confidence interval if provided
    if y_lower is not None and y_upper is not None:
        ax.fill_between(
            dates, y_lower, y_upper,
            color="gold", alpha=0.3,
            label="95% Confidence Interval"
        )
    
    # Plot actual values
    ax.plot(dates, y_actual, "ko-", label="Actual", linewidth=2, markersize=6)
    
    # Plot predictions
    ax.plot(dates, y_pred, "bs-", label="Forecast", linewidth=2, markersize=6, alpha=0.7)
    
    # Draw train/val split line if provided
    if train_split_date is not None:
        ax.axvline(
            x=train_split_date,
            color="red", linestyle=":", linewidth=2, alpha=0.7,
            label="Train/Val Split"
        )
    
    # Set title
    if metrics_text:
        full_title = f"{title}\n{metrics_text}"
    else:
        full_title = title
    
    ax.set_title(full_title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Trade Balance", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_full_period_with_validation(
    train_dates: pd.Series,
    train_actual: pd.Series,
    val_dates: pd.Series,
    val_actual: pd.Series,
    val_pred: pd.Series,
    val_lower: Optional[pd.Series] = None,
    val_upper: Optional[pd.Series] = None,
    target_name: str = "Target",
    metrics_text: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
    show: bool = True
) -> plt.Figure:
    """
    Plot full period with training data and validation period.
    
    Args:
        train_dates: Training period dates
        train_actual: Training period actual values
        val_dates: Validation period dates
        val_actual: Validation period actual values
        val_pred: Validation period predictions
        val_lower: Lower bound of confidence interval (optional)
        val_upper: Upper bound of confidence interval (optional)
        target_name: Name of the target
        metrics_text: Text to display in title
        save_path: Path to save figure (optional)
        figsize: Figure size
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot training data (gray)
    ax.plot(
        train_dates, train_actual,
        color="gray", alpha=0.6, linewidth=1.5,
        label="Training Data (Actual)", linestyle="-"
    )
    
    # Plot validation actual (black)
    ax.plot(
        val_dates, val_actual,
        color="black", linewidth=2.5, marker="o", markersize=6,
        label="Validation Actual", linestyle="-"
    )
    
    # Plot confidence interval if provided
    if val_lower is not None and val_upper is not None:
        ax.fill_between(
            val_dates, val_lower, val_upper,
            color="gold", alpha=0.3,
            label="95% Confidence Interval"
        )
    
    # Plot validation predictions (blue)
    ax.plot(
        val_dates, val_pred,
        color="blue", linewidth=2, marker="s", markersize=6, alpha=0.8,
        label="Validation Forecast", linestyle="--"
    )
    
    # Draw train/val split line
    if len(val_dates) > 0:
        split_date = val_dates.iloc[0] - pd.DateOffset(days=1)
        ax.axvline(
            x=split_date,
            color="red", linestyle=":", linewidth=2, alpha=0.7,
            label="Train/Val Split"
        )
    
    # Set title
    if metrics_text:
        full_title = f"{target_name} - Full Period with Validation\n{metrics_text}"
    else:
        full_title = f"{target_name} - Full Period with Validation"
    
    ax.set_title(full_title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Trade Balance", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig

