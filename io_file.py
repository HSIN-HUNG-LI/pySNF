import os
import pandas as pd
import sys
from tkinter import messagebox
from datetime import datetime
from pathlib import Path
from typing import Union


def load_dataset(path: str) -> pd.DataFrame:
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
    output_path = Path.cwd() / parent_folder_name / f"{timestamp}_output"
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


def write_excel(
    df_stdh, df_act, df_conc, output_dir: Union[str, Path], file_name:str
) -> None:
    """Writes all three sheets to a single workbook, overwriting if exists."""
    path = Path(output_dir, f"{file_name}.xlsx")
    if path.exists():
        path.unlink()
    with pd.ExcelWriter(path) as writer:
        df_stdh.to_excel(writer, sheet_name="STDH", index=False)
        df_conc.to_excel(writer, sheet_name="Concentration", index=False)
        df_act.to_excel(writer, sheet_name="Activity", index=False)