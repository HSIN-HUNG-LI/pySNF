import os
import pandas as pd
import sys
from tkinter import messagebox
from datetime import datetime
from pathlib import Path
from typing import Union
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_stdh_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_file = project_root / "data" / "snfs_details" / "all_stdh_dataset.csv"
    return Path(data_file)


def get_snfs_dir_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_file = project_root / "data" / "snfs_details"
    return Path(data_file)


def get_output_dir_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_file = project_root / "output"
    return Path(data_file)

def get_grid_ParqFile_path() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    data_file = project_root / "data" / "grid_database" / "grid_database.parq"
    return Path(data_file)

def load_dataset(    
    file_path: Path,
) -> pd.DataFrame:
    """
    Load a CSV dataset from the given path. Shows an error dialog if the file
    is missing or cannot be read, and returns an empty DataFrame on failure.
    """
    # file_path = data_directory / file_name
    extension = file_path.suffix.lower()
    if not os.path.exists(file_path):
        messagebox.showerror("Error", f"File '{file_path}' not found.")
        sys.exit(1)
    try:
        if extension in {".csv"}:
            # Use memory_map for faster I/O on large CSVs
            df = pd.read_csv(file_path, header=0, memory_map=True)
        elif extension in {".xls", ".xlsx"}:
            # Explicitly specify engine for consistency
            df = pd.read_excel(file_path, header=0, engine="openpyxl")
        else:
            messagebox.showerror(
                "Unsupported file extension '%s'. Please provide a .csv or .xlsx file.",
                extension
            )
            return pd.DataFrame()
        logging.info("Successfully read %s \n(%d rows, %d columns)",
                     file_path, df.shape[0], df.shape[1])
        return df
    
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read '{file_path}': {e}")
        sys.exit(1)

def save_PredData(df: pd.DataFrame) -> None:
    """
    Save a DataFrame to CSV in the specified directory.
    Parameters:
    - df: pandas DataFrame to save
    """
    output_dir = get_output_dir_path() / "Prediction"
    filename = f"{pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")}_output.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)



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


def get_SNFdetail_TableUnit(option: int = 1) -> list:
    if option == 1:
        return [
            "SNF_id",
            "Type",
            "MTU",
            "Length(cm)",
            "Down(Days)",
            "Cycles",
            "Enrich(%U235)",
            "SP(MW)",
            "Burnup(MWD/MTU)",
            "Cool(Years)",
            "SP1(MW)",
            "SP2(MW)",
            "SP3(MW)",
            "SP4(MW)",
            "SP5(MW)",
            "SP6(MW)",
            "UP1(Days)",
            "UP2(Days)",
            "UP3(Days)",
            "UP4(Days)",
            "UP5(Days)",
            "UP6(Days)",
            "Down1(Days)",
            "Down2(Days)",
            "Down3(Days)",
            "Down4(Days)",
            "Down5(Days)",
        ]
    else:
        return []
