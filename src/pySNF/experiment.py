import math
import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from base import PredictSNFs_interpolate
from io_file import (
    load_dataset,
    get_grid_ParqFile_path,
    get_stdh_path,
    create_output_dir,
    get_output_dir_path,
)


class GridResolutionExperiment:
    """Run grid‐resolution interpolation experiments for SNF data."""

    def __init__(self):
        # Load grid‐interpolation data and standard dataset
        self.grid_data = pd.read_parquet(get_grid_ParqFile_path())
        self.df_in = load_dataset(get_stdh_path())

        # Define metrics: (label, Triton col, grid col, y‐axis limits)
        self.error_metrics = [
            ("FN", "FN_0y", "FN_prediction", (-0.07, 0.16)),
            ("FG", "FG_0y", "FG_prediction", (-0.03, 0.215)),
            ("HG", "HG_0y", "HG_prediction", (-0.05, 0.58)),
            ("DH", "DH_0y", "DH_prediction", (-0.025, 0.16)),
        ]

    def run(
        self,
        exp_folder_name: str = "1111",
    ):
        """
        Start the prediction process in varying grid resolutions.
        This method is called by Notebook to run the experiment.
        """
        enrich_factor = int(exp_folder_name[0])
        sp_factor = int(exp_folder_name[1])
        bp_factor = int(exp_folder_name[2])
        cool_factor = int(exp_folder_name[3])
        # ============ Start Exp ============
        enrich_step = 0.5 * enrich_factor
        sp_step = 5 * sp_factor
        burnup_step = 3000 * bp_factor
        cool_step = cool_factor
        # ============ End Exp ============
        print(f'enrich_step: {enrich_step}, sp_step: {sp_step}, burnup_step: {burnup_step}, cool_step: {cool_step}')

        enrich_space = np.arange(1.5, 6.1, enrich_step)
        sp_space = np.arange(5, 46, sp_step)
        burnup_space = np.arange(5000, 74100, burnup_step)
        # cool_space = np.logspace(-5.75, 6.215, cool_step, base=math.e)
        cool_space = np.logspace(-5.75, 6.215, 150, base=math.e)
        cool_space = cool_space[1::cool_step]
        print(f"enrich_space:{len(enrich_space)}, sp_space:{len(sp_space)}, burnup_space:{len(burnup_space)}, cool_space:{len(cool_space)}")
        out_cols = [f"{p}_prediction" for p in ("DH", "FN", "HG", "FG")]
        series_list: list[pd.Series] = []

        # Load dataframe
        df_in_copy = self.df_in.copy()
        desired_cols = ["Enrich", "SP", "Burnup", "Cool"]
        df_in_copy = df_in_copy.loc[:, desired_cols].copy()

        PredAssy = PredictSNFs_interpolate(
            self.grid_data,
            enrich_space,
            sp_space,
            burnup_space,
            cool_space,
            out_cols,
        )
        for i, (_, row) in enumerate(df_in_copy.iterrows()):
            series_list.append(
                PredAssy.interpolate(
                    row["Enrich"],
                    row["Burnup"],
                    row["SP"],
                    row["Cool"],
                )
            )

        df_preds = pd.DataFrame(series_list)
        df_out = pd.concat(
            [self.df_in.reset_index(drop=True), df_preds.reset_index(drop=True)],
            axis=1,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_folder = (
            get_output_dir_path()
            / f"PredictEXP_Results"
            / f"{timestamp}_output_{exp_folder_name}"
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_folder / f"Prediction_{exp_folder_name}.csv", index=False)
        df = compute_relative_errors(df_out, self.error_metrics)
        # Plot the combined boxplots and get the melted DataFrame
        title_boxplot = (
            f"Relative Error across Source term and Decay heat \n (Grid Resolutions: En:{exp_folder_name[0]}, SP:{exp_folder_name[1]}, Bp:{exp_folder_name[2]}, Ct:{exp_folder_name[3]})"
        )
        plot_results = output_folder / f"error_summary_{exp_folder_name}.png"
        df_long = plot_one_error_boxplots(
            df, self.error_metrics, title_boxplot, plot_results
        )
        # Summarize statistics and export to Excel
        stats_output = output_folder / f"error_summary_stats_{exp_folder_name}.xlsx"
        summarize_error_stats_save(df_long, stats_output, self.error_metrics)
        print(f"Summary plot and statistics saved to: {output_folder}")


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
    plt.suptitle(f"{title_boxplot}", fontsize=fontsize_all + 2)
    sns.boxplot(data=df_long, x="Metric", y="Error")

    # 4. Tidy up labels and fonts
    plt.xlabel("")  # no x-axis label
    plt.ylabel("Interpolation / TRITON Relative Error", fontsize=fontsize_all + 2)
    plt.xticks(fontsize=fontsize_all)
    y_ticks = np.arange(0.4, 1.6 + 1e-8, 0.1)
    plt.yticks(y_ticks, fontsize=fontsize_all)
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="y = 1")

    # 5. Tight layout so nothing overlaps
    plt.tight_layout()
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    # plt.show()

    # Return the long-form DataFrame for post-plot analysis
    return df_long


def summarize_error_stats_save(
    df_long: pd.DataFrame, output_path: Path, ERROR_METRICS: list
) -> None:
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
