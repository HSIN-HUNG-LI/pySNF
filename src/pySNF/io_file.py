import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Union, List

import pandas as pd
from tkinter import messagebox

# Keep existing logging behavior but also get a module logger for local use.
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)


def get_stdh_path() -> Path:
    """
    Return the absolute path to the aggregated STDH dataset CSV.

    The path is resolved relative to the repository root:
    <root>/data/DataBase_SNFs/all_stdh_dataset.csv
    """
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "DataBase_SNFs" / "all_stdh_dataset.csv"


def get_snfs_dir_path() -> Path:
    """
    Return the directory containing per-SNF detail CSVs.

    The path is resolved relative to the repository root:
    <root>/data/DataBase_SNFs
    """
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "DataBase_SNFs"


def get_output_dir_path() -> Path:
    """
    Return the application's root output directory.

    The path is resolved relative to the repository root:
    <root>/output
    """
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "output"


def get_grid_ParqFile_path() -> Path:
    """
    Return the absolute path to the grid Parquet file used by prediction.

    The path is resolved relative to the repository root:
    <root>/data/DataBase_GridPoint/grid_database.parq
    """
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "DataBase_GridPoint" / "grid_database.parq"


def load_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV or Excel dataset from the given absolute file path.

    On failure, shows a Tk error dialog and terminates the process with exit code 1,
    matching existing behavior.

    Parameters
    ----------
    file_path : Path
        Absolute path to the CSV/XLS/XLSX file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame; on unsupported extension returns an empty DataFrame.
    """
    extension = file_path.suffix.lower()

    if not file_path.exists():
        messagebox.showerror("Error", f"File '{file_path}' not found.")
        sys.exit(1)

    try:
        if extension == ".csv":
            # memory_map speeds up I/O on large files without altering semantics
            df = pd.read_csv(file_path, header=0, memory_map=True)
        elif extension in (".xls", ".xlsx"):
            # Use a consistent engine for predictability
            df = pd.read_excel(file_path, header=0, engine="openpyxl")
        else:
            # Preserve behavior: inform user and return an empty frame
            messagebox.showerror(
                "Error",
                f"Unsupported file extension '{extension}'. "
                "Please provide a .csv or .xlsx file.",
            )
            return pd.DataFrame()

        return df

    except Exception as e:
        messagebox.showerror("Error", f"Failed to read '{file_path}': {e}")
        sys.exit(1)


def save_PredData(df: pd.DataFrame) -> None:
    """
    Save a DataFrame to CSV under <output>/Prediction with a timestamped filename.

    Filename pattern: YYYY-MM-DD_HH-MM-SS_output.csv
    """
    output_dir = get_output_dir_path() / "Prediction"
    # Fix: avoid quote collision in f-string format pattern.
    filename = f"{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}_output.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)


def create_output_dir(parent_folder_name: Union[str, Path]) -> Path:
    """
    Create a timestamped output directory under <output>/<parent_folder_name>.

    Parameters
    ----------
    parent_folder_name : str | Path
        Name (or Path) of the subfolder under the global output directory.

    Returns
    -------
    Path
        Path to the newly created timestamped directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = (
        get_output_dir_path() / Path(parent_folder_name) / f"{timestamp}_output"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def write_excel(
    df_stdh: pd.DataFrame,
    df_act: pd.DataFrame,
    df_conc: pd.DataFrame,
    output_dir: Union[str, Path],
    file_name: str,
) -> None:
    """
    Write STDH, Concentration, and Activity sheets to a single Excel workbook,
    overwriting any existing file with the same name.
    """
    path = Path(output_dir) / f"{file_name}.xlsx"
    if path.exists():
        path.unlink()
    with pd.ExcelWriter(path) as writer:
        df_stdh.to_excel(writer, sheet_name="STDH", index=False)
        df_conc.to_excel(writer, sheet_name="Concentration", index=False)
        df_act.to_excel(writer, sheet_name="Activity", index=False)


def set_SNFdetail_info(option: int = 1) -> List[str]:
    """
    Return the ordered list of SNF detail field keys used for lookups/rendering.

    Parameters
    ----------
    option : int
        Reserved for future variants; only option=1 is supported.

    Returns
    -------
    list[str]
        Field keys (internal names). Empty list for unsupported options.
    """
    if option == 1:
        return [
            "SNF_id",
            "Type",
            "MTU",
            "Length",
            "Down",
            "Cycles",
            "Enrich",
            "SP",
            "Burnup",
            "Cool",
            "SP1",
            "SP2",
            "SP3",
            "SP4",
            "SP5",
            "SP6",
            "UP1",
            "UP2",
            "UP3",
            "UP4",
            "UP5",
            "UP6",
            "Down1",
            "Down2",
            "Down3",
            "Down4",
            "Down5",
        ]
    return []


def get_SNFdetail_TableUnit(option: int = 1) -> List[str]:
    """
    Return the ordered list of SNF detail field labels (with units) for display.

    Parameters
    ----------
    option : int
        Reserved for future variants; only option=1 is supported.

    Returns
    -------
    list[str]
        Human-readable labels matching the order of `set_SNFdetail_info`.
        Empty list for unsupported options.
    """
    if option == 1:
        return [
            "SNF_id",
            "Type",
            "MTU",
            "Length (cm)",
            "Down (Days)",
            "Cycles",
            "Enrich (%U235)",
            "SP (MW)",
            "Burnup (MWD/MTU)",
            "Cool (Years)",
            "SP1 (MW)",
            "SP2 (MW)",
            "SP3 (MW)",
            "SP4 (MW)",
            "SP5 (MW)",
            "SP6 (MW)",
            "UP1 (Days)",
            "UP2 (Days)",
            "UP3 (Days)",
            "UP4 (Days)",
            "UP5 (Days)",
            "UP6 (Days)",
            "Down1 (Days)",
            "Down2 (Days)",
            "Down3 (Days)",
            "Down4 (Days)",
            "Down5 (Days)",
        ]
    return []
