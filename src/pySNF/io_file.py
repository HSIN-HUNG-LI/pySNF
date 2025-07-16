import os
import pandas as pd
import sys
from tkinter import messagebox
from datetime import datetime
from pathlib import Path
from typing import Union

def get_stdh_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_file   = project_root / "data" / "snfs_details" / "all_stdh_dataset.csv"
    return Path(data_file)

def get_snfs_dir_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_file   = project_root / "data" / "snfs_details"
    return Path(data_file)

def get_output_dir_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_file   = project_root / "output"
    return Path(data_file)

def load_dataset(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a CSV dataset from the given path. Shows an error dialog if the file
    is missing or cannot be read, and returns an empty DataFrame on failure.
    """
    if not os.path.exists(path):
        messagebox.showerror("Error", f"File '{path}' not found.")
        sys.exit(1)
    try:
        return pd.read_csv(path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read '{path}': {e}")
        sys.exit(1)


def create_output_dir(parent_folder_name: Union[str, Path]) -> Path:
    """
    Create a timestamped output directory under the specified parent folder.

    Args:
        parent_folder : Name of the directory under the current working directory where the output folder will be created.

    Returns:
        Path: Path object pointing to the newly created output directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = get_output_dir_path() / parent_folder_name / f"{timestamp}_output"
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def write_excel(
    df_stdh, df_act, df_conc, output_dir: Union[str, Path], file_name: str
) -> None:
    """Writes all three sheets to a single workbook, overwriting if exists."""
    path = Path(output_dir, f"{file_name}.xlsx")
    if path.exists():
        path.unlink()
    with pd.ExcelWriter(path) as writer:
        df_stdh.to_excel(writer, sheet_name="STDH", index=False)
        df_conc.to_excel(writer, sheet_name="Concentration", index=False)
        df_act.to_excel(writer, sheet_name="Activity", index=False)



def set_SNFdetail_info(option: int = 1) -> list:
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
    else:
        return []
