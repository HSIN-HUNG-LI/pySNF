from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(csv_path: Path, CATEGORY_MAP: dict) -> pd.DataFrame:
    """
    Load the CSV, replace Type values, sort by Type, and return the DataFrame.
    """
    df = pd.read_csv(csv_path)
    # df["Type"] = df["Type"].replace(CATEGORY_MAP)
    return df


def compute_relative_errors(df: pd.DataFrame, ERROR_METRICS: list) -> pd.DataFrame:
    """
    For each metric in ERROR_METRICS, compute (grid / triton)
    and store it in a new column 'error_<metric>'.
    """
    for metric, triton_col, grid_col, _ in ERROR_METRICS:
        df[f"error_{metric}"] = df[grid_col].div(df[triton_col])
    return df


def plot_error_boxplots(df: pd.DataFrame, ERROR_METRICS) -> None:
    """
    Create a 2x2 grid of boxplots for each 'error_<metric>' column,
    grouped by the 'Type' category.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    for ax, (metric, _, _, ylim) in zip(axes, ERROR_METRICS):
        err_col = f"error_{metric}"
        title = f"{metric} Relative Error"

        sns.boxplot(data=df, x="Type", y=err_col, ax=ax)
        ax.set_xlabel("")  # remove the x-axis label
        ax.set_ylabel(title, fontsize=18)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)

    plt.tight_layout(
        rect=(0.12, 0.05, 0.98, 0.95), h_pad=0.25, w_pad=3  # left, bottom, right, top
    )
    plt.show()


def plot_one_error_boxplots(
    df: pd.DataFrame,
    ERROR_METRICS: list,
    title_boxplot: str,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a single figure with four boxplots—one for each error metric—
    combining all samples (no grouping by Type).
    """
    # 1. Build a list of the four error column names
    error_cols = [f"error_{metric}" for metric, _, _, _, in ERROR_METRICS]

    # 2. Melt the DataFrame into long form: columns = ['Metric', 'Error']
    df_long = df[error_cols].melt(var_name="Metric", value_name="Error")
    fontsize_all = 14

    # 3. Draw a single row of boxplots
    plt.figure(figsize=(8, 6), dpi=600)
    plt.suptitle(f"{title_boxplot}", fontsize=fontsize_all+6)
    sns.boxplot(data=df_long, x="Metric", y="Error")

    # 4. Tidy up labels and fonts
    plt.xlabel("")  # no x-axis label
    plt.ylabel("Interpolation / TRITON Relative Error", fontsize=fontsize_all+2)
    plt.xticks(fontsize=fontsize_all)
    y_ticks = np.arange(0.4, 1.6 + 1e-8, 0.1)
    plt.yticks(y_ticks, fontsize=fontsize_all)
    plt.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label='y = 1')
    
    # 5. Tight layout so nothing overlaps
    plt.tight_layout()
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

    # Return the long-form DataFrame for post-plot analysis
    return df_long


def summarize_error_stats(
    df_long: pd.DataFrame, output_path: Path, ERROR_METRICS: list
) -> pd.DataFrame:
    """
    Compute descriptive statistics (mean, std, min, 25%, 50%, 75%, max)
    for each error metric in the long-form DataFrame, then pivot so that:
      - columns = [FN, FG, HG, DH]
      - index  = [mean, std, min, 25%, 50%, 75%, max]
    Save to Excel and return the pivoted DataFrame.
    """
    # 1. Compute describe() by Metric
    stats = df_long.groupby("Metric")["Error"].describe(percentiles=[0.25, 0.5, 0.75])
    # 2. Keep only the wanted stat columns
    stats = stats[["mean", "std", "min", "25%", "50%", "75%", "max"]]

    # 3. Define the exact order of the error columns
    #    (these must match the melted df_long 'Metric' values)
    metric_order = [
        f"error_{m}" for m, *_ in ERROR_METRICS
    ]  # ["error_FN","error_FG",...]
    # 4. Reorder by Metric, then transpose so stats → rows, metrics → columns
    stats = stats.loc[metric_order].T

    # 5. Clean up column names (drop the "error_" prefix)
    stats.columns = [col.replace("error_", "") for col in stats.columns]
    stats.index.name = None

    # 6. Write to Excel
    stats.to_excel(output_path)

    return stats
