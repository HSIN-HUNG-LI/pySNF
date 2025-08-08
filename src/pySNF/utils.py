import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, cast, Optional


class SNFParquetConverter:
    """
    A converter that reads all CSV files in a given folder,
    concatenates them into a single DataFrame (with a 'source_file' column),
    and writes the result to a Parquet file.
    Also supports reading back individual subsets by the original filename.
    """

    def __init__(self, input_folder: str, output_file: str):
        """
        :param input_folder: Path to folder containing CSV files.
        :param output_file: Path where the combined Parquet will be saved.
        """
        self.input_folder = input_folder
        self.output_file = output_file

    def convert_to_parquet(self) -> None:
        """
        - Scan for *.csv under input_folder
        - Read each with the first column as index (nuclide IDs)
        - Add a 'source_file' column
        - Concatenate and write to Parquet (index preserved)
        """
        pattern = os.path.join(self.input_folder, "*.csv")
        file_paths = glob.glob(pattern)
        dfs = []

        for path in file_paths:
            # Read CSV, using the first column (Unnamed: 0) as the index
            df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
            # Keep track of which file each row came from
            df["source_file"] = os.path.basename(path)
            dfs.append(df)

        # Combine all
        combined = pd.concat(dfs, axis=0, ignore_index=False)
        # give the index a real name
        # combined.index.name = "nuclide"

        # Write to Parquet, preserving the index (nuclide ID)
        combined.to_parquet(self.output_file)
        print(
            f"Wrote {combined.shape[0]} rows "
            f"with index name '{combined.index.name}' "
            f"from {len(file_paths)} files into '{self.output_file}'"
        )

    @staticmethod
    def read_parquet(parquet_file: Union[str, Path]) -> pd.DataFrame:
        """
        Read the entire Parquet file into a DataFrame.

        :param parquet_file: Path to the Parquet file.
        :return: pandas DataFrame with all records.
        """
        return pd.read_parquet(parquet_file)

    @staticmethod
    def read_by_source(
        parquet_file: Union[str, Path], source_filename: str
    ) -> pd.DataFrame:
        """
        Read only the rows that came from a specific original CSV file.

        :param parquet_file: Path to the Parquet file.
        :param source_filename: The CSV filename to filter by (e.g. 'C1A001.csv').
        :return: pandas DataFrame subset.
        """
        df = pd.read_parquet(parquet_file)
        filtered_df = cast(pd.DataFrame, df.loc[df["source_file"] == source_filename])
        return filtered_df


def restricted_year(x: str) -> float:
    since_year = 2022.0
    year = float(x)
    if not (since_year <= year <= since_year + 500):
        raise TypeError(
            f"Year must be between {int(since_year)} and {int(since_year + 500)}"
        )
    return year


def plot_Gram_Ci(
    df: pd.DataFrame,
    assy_name: str,
    plt_name: str,
    plt_unit: str,
    series_info: pd.DataFrame,
    storage_path: Path | str,
    percentage: bool = False,
    linear: bool = True,
) -> None:
    """
    Plot top-20 nuclide contributions for a given ASSY over defined time steps.

    Steps:
    1) Compute year labels (2022+y).
    2) If percentage mode: normalize by total.
    3) Sort by initial ('0y'), drop total/subtotal rows.
    4) Convert MTU to ASSY via provided converter.
    5) Style and render a stacked bar chart.
    6) Adjust axis scales and limits for 'log' and concentration cases.
    7) Save figure to `storage_path` with a descriptive filename.
    """

    # Ensure valid storage path
    storage_path = Path(storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)

    # 1) Prepare time labels
    steps = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    year_cols = [f"{2022 + y}y" for y in steps]

    # 2) Optional percentage normalization
    if percentage:
        df = df.div(df.loc["total"], axis=0)
        plt_unit = plt_unit.split("(")[0] + "_Contribution"

    # 3) Sort, drop first two summary rows
    df_sorted = df.sort_values(by="0y", ascending=False).iloc[2:]

    # 4) Convert units (assumes Converter_MTU2ASSY is imported elsewhere)
    df_converted = Converter_MTU2ASSY(df_sorted, series_info)

    # 5) Plot styling
    sns.set(rc={"figure.figsize": (12, 8)}, style="whitegrid", font_scale=1.5)
    ax = df_converted.iloc[:20].T.plot(kind="bar", stacked=True)

    # 6) Axis labels and scales
    ax.set_xticklabels(year_cols, rotation=30)
    ax.set_xlabel("")
    ax.set_ylabel(plt_unit)
    ax.set_yscale("linear" if linear else "log")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Special y-limit for concentration on log scale
    if not linear and "Concentration" in plt_unit:
        ax.set_ylim(2e5, 2e6)

    # Legend outside plot
    ax.legend(title=f"Top20 ({assy_name})", bbox_to_anchor=(1, 0.5), loc="center left")

    plt.tight_layout()

    # 7) Save output
    filename = f"{assy_name}_{plt_name}.png"
    plt.savefig(storage_path / filename)
    plt.close()


def Converter_MTU2ASSY(values, series_info):
    """
    Convert values from MTU (metric tons uranium) to assembly units.

    Parameters:
        values (pd.DataFrame or numeric): DataFrame of nuclide values with columns like '0y', '1y', ..., or a single numeric value/Series.
        series_info (pd.DataFrame): DataFrame containing a single-row 'MTU' column representing MTU per assembly.

    Returns:
        pd.DataFrame or numeric: Converted values in assembly units.
    """
    # Validate input
    if not isinstance(series_info, pd.DataFrame):
        raise ValueError("series_info must be a pandas DataFrame.")

    # Extract the MTU-to-assembly conversion factor
    mtu_per_assy = series_info["MTU"].iat[0]

    # Define the expected year columns
    year_labels = [
        f"{y}y" for y in ["0", "1", "2", "5", "10", "20", "50", "100", "200", "500"]
    ]

    if isinstance(values, pd.DataFrame):
        # Multiply each year column by the conversion factor and return a new DataFrame
        converted = values.copy()
        for col in year_labels:
            if col in converted:
                converted[col] = converted[col] * mtu_per_assy
        return converted

    # Handle scalar or pandas Series
    return values * mtu_per_assy
