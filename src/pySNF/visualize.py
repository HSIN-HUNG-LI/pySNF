from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from typing import Optional

def plot_4x4_scatterplot(
    output_path: Path, df: pd.DataFrame, y_vars: list[str], plot_title: str
) -> None:
    """
    4×4 scatterplot matrix with:
    - Centralized, easily adjustable font sizes (larger defaults).
    - Per-subplot x/y scale control (linear or log) via SCALES mapping.
    """

    def apply_decade_ticks(ax, axis: str = "x"):
        """
        Force clean log ticks: 10^k only (no 2×10^k minors), using mathtext.
        axis ∈ {"x","y"}.
        """
        major = mticker.LogLocator(base=10.0)  # decades only
        fmt = mticker.LogFormatterMathtext(base=10.0, labelOnlyBase=True)

        if axis == "x":
            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(fmt)
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        else:
            ax.yaxis.set_major_locator(major)
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # -----------------------------
    # Configurable parameters
    # -----------------------------

    x_vars = ["Enrich", "SP", "Burnup", "Cool"]

    # Central font-size config (increase/decrease here)
    FS = {
        "title": 40,  # figure suptitle
        "axis_label": 28,  # x/y axis labels
        "tick": 24,  # tick labels
        "legend": 28,  # legend text
    }
    # Map Y variable names to display labels (strip "_0y")
    Y_LABEL_MAP: dict[str, str] = {}
    for y in y_vars:
        # Remove the last 3 chars only if the suffix is exactly "_0y"
        label = y[:2] if y.endswith("_0y") or y.endswith("_prediction") else y
        Y_LABEL_MAP[y] = label

    X_LABEL_MAP = {
        "Enrich": "Enrichment (%U235)",
        "SP": "Specific Power (MW)",
        "Burnup": "Burnup (MWd/MTU)",
        "Cool": "Cooling time (Year)",
    }

    MARKER_SIZE = 16  # scatter marker size
    ALPHA = 0.7  # point transparency

    # Per-subplot scale control:
    # Key = (y_var, x_var), Value = (xscale, yscale) where each is "linear" or "log".
    # Defaults to linear; override any cell you want.
    SCALES = {(y, x): ("linear", "linear") for y in y_vars for x in x_vars}
    for y in y_vars:
        SCALES[(y, "Cool")] = ("log", "log")
    # --- EXAMPLES (uncomment/modify as needed) ---
    # SCALES[("HG_0y", "Cool")] = ("log", "log")
    # SCALES[("FG_0y", "Burnup")] = ("linear", "log")
    # SCALES[("FN_0y", "SP")] = ("linear", "linear")

    PALETTE_NAME = "tab10"

    # -----------------------------
    # Load & prepare data
    # -----------------------------
    use_cols = x_vars + y_vars + ["Type"]
    # df = pd.read_csv(CSV_PATH, usecols=use_cols)

    # Enforce numeric dtypes and category for 'Type'
    for c in x_vars + y_vars:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Type"] = df["Type"].astype("category")
    df = df.dropna(subset=use_cols)

    type_order = sorted(df["Type"].cat.categories.tolist())
    df["Type"] = df["Type"].cat.reorder_categories(type_order, ordered=False)

    palette = sns.color_palette(PALETTE_NAME, n_colors=len(type_order))
    type_to_color = dict(zip(type_order, palette))

    # -----------------------------
    # Plot: independent tick_params, shared xlabel/ylabel per row/column
    # -----------------------------
    sns.set_theme(context="notebook", style="whitegrid")
    n_rows, n_cols = len(y_vars), len(x_vars)

    # No sharex/sharey so tick_params are independent per axis
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 18), constrained_layout=False)

    for i, y in enumerate(y_vars):
        for j, x in enumerate(x_vars):
            ax = axes[i, j]

            sns.scatterplot(
                data=df,
                x=x,
                y=y,
                hue="Type",
                hue_order=type_order,
                palette=type_to_color,
                s=MARKER_SIZE,
                alpha=ALPHA,
                linewidth=0,
                legend=False,
                ax=ax,
            )

            # Per-subplot axis scales
            xscale, yscale = SCALES[(y, x)]
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)

            # If the X variable is Cool and it’s log, show 10^k on X (no 2×10^k)
            if x == "Cool" and xscale == "log":
                apply_decade_ticks(ax, "x")

            # If Y is log (e.g., FN/HG/FG/DH on log), also force 10^k on Y
            if yscale == "log":
                apply_decade_ticks(ax, "y")

            # Independent tick styling per subplot
            ax.tick_params(axis="both", which="both", labelsize=FS["tick"])

            # Shared labels: only left column + bottom row show axis labels
            y_label = Y_LABEL_MAP.get(y, y)  # fallback: generic strip
            ax.set_ylabel(y_label if j == 0 else "", fontsize=FS["axis_label"])
            x_label = X_LABEL_MAP.get(x, x)  # fallback to original if not mapped
            ax.set_xlabel(x_label if i == n_rows - 1 else "", fontsize=FS["axis_label"])

            # Independent tick styling for EVERY subplot
            ax.tick_params(axis="both", which="both", labelsize=FS["tick"])

    # Figure title (keep slightly high to make room for legend below it)
    fig.suptitle(
        f"{plot_title}",
        fontsize=FS["title"],
        y=0.94,  # a bit higher; tweak 0.978–0.986 if needed
    )

    n_types = len(type_order)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=type_to_color[t],
            markeredgecolor=type_to_color[t],
            label=t,
        )
        for t in type_order
    ]
    ncol = min(n_types, 5)

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.9,
        ),  # closer to the title (raise this to get even tighter)
        ncol=ncol,
        frameon=True,
        borderaxespad=0.1,  # shrink space between legend and figure
        borderpad=0.3,  # shrink padding inside the legend frame
        handletextpad=0.4,
        columnspacing=0.8,
        labelspacing=0.3,
    )
    plt.setp(leg.get_texts(), fontsize=FS["legend"])
    plt.setp(leg.get_title(), fontsize=FS["legend"])

    # Keep room for the tighter header; raise 'top' if legend overlaps
    fig.tight_layout(rect=(0.03, 0.03, 0.97, 0.90))

    # Save/show as before
    fig.savefig(output_path, dpi=300)
    # plt.show()
    # print(f"Saved figure to: {output_path.resolve()}")


def compute_relative_errors(df: pd.DataFrame, ERROR_METRICS: list) -> pd.DataFrame:
    """
    For each metric in ERROR_METRICS, compute (grid / triton)
    and store it in a new column 'error_<metric>'.
    """
    for metric, triton_col, grid_col in ERROR_METRICS:
        df[f"error_{metric}"] = df[grid_col].div(df[triton_col])
    return df

def plot_stdh_RelativeError_boxplots(
    df: pd.DataFrame,
    ERROR_METRICS: list,
    title_boxplot: str,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create a single figure with four boxplots—one for each ST&DH metric—
    combining all samples (no grouping by Type).

    - X-axis labels: ["DH", "FN", "HG", "FG"]
    - Y-axis label: "Ratio" (formerly "Error")
    - Returns the long-form DataFrame used for plotting with cleaned labels.
    """
    # Compute relative error matrix (expects columns like: error_DH, error_FN, ...)
    df_error_matrix = compute_relative_errors(df, ERROR_METRICS)

    # Columns to plot (expected to exist in df_error_matrix)
    stdh_metrics = ["DH", "FN", "HG", "FG"]
    error_cols = [f"error_{m}" for m in stdh_metrics]

    # Validate presence of required columns early with a clear error
    missing = [c for c in error_cols if c not in df_error_matrix.columns]
    if missing:
        raise ValueError(f"Missing expected columns in error matrix: {missing}")

    # Melt to long form and rename the value field to 'Ratio'
    df_long = df_error_matrix[error_cols].melt(var_name="Metric", value_name="Ratio")

    # Strip the 'error_' prefix so x labels are just DH / FN / HG / FG
    df_long["Metric"] = df_long["Metric"].str.replace("error_", "", regex=False)

    # Enforce a consistent, desired ordering on the x-axis
    cat_order = pd.CategoricalDtype(categories=stdh_metrics, ordered=True)
    df_long["Metric"] = df_long["Metric"].astype(cat_order)

    # --- Plot ---
    fontsize_all = 14
    fig, ax = plt.subplots(figsize=(8, 6), dpi=600)
    sns.boxplot(data=df_long, x="Metric", y="Ratio", ax=ax)

    # Labels, ticks, and reference line at y = 1
    ax.set_title(title_boxplot, fontsize=fontsize_all + 2)
    ax.set_xlabel("")  # no x-axis label
    ax.set_ylabel("Ratio", fontsize=fontsize_all + 2)
    ax.tick_params(axis="x", labelsize=fontsize_all)
    y_ticks = np.arange(0.4, 1.6 + 1e-8, 0.1)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="y", labelsize=fontsize_all)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Return the long-form DataFrame for any post-plot analysis
    return df_long
