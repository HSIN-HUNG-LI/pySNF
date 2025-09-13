import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Union, List, Optional

import pandas as pd
from tkinter import messagebox

# Keep existing logging behavior but also get a module logger for local use.
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# logger = logging.getLogger(__name__)


def get_stdh_path() -> Path:
    """
    Return the absolute path to the aggregated STDH dataset CSV.

    The path is resolved relative to the repository root:
    <root>/data/DataBase_SNFs/DataBase_All_DHST.csv
    """
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "DataBase_SNFs" / "DataBase_All_DHST.csv"


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
    """
    project_root = Path(__file__).resolve().parents[2]
    parq_name = get_parq_name( project_root / "data" / "DataBase_Grid" / "Default_DataBase_README.txt")
    return project_root / "data" / "DataBase_Grid" / parq_name


def get_grid_space() -> str:
    """
    Return the grid space identifier used in the application.
    """
    project_root = Path(__file__).resolve().parents[2]
    parq_name = get_parq_name( project_root / "data" / "DataBase_Grid" / "Default_DataBase_README.txt")
    return str(extract_last_four_digits(parq_name))

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

def extract_last_four_digits(filename: str) -> Optional[str]:
    """
    Extract the last 4 consecutive digits before the file extension
    from a given filename string.
    
    Parameters
    ----------
    filename : str
        Input string, e.g., 'grid_database_1111.parq'.
    
    Returns
    -------
    Optional[str]
        A 4-digit string (e.g., '1111') if found, else None.
    
    Examples
    --------
    >>> extract_last_four_digits("grid_database_1111.parq")
    '1111'
    >>> extract_last_four_digits("grid_database_999.parq")
    None
    >>> extract_last_four_digits("data/grid_database_1412.parq")
    '1412'
    """
    match = re.search(r"(\d{4})(?=\.\w+$)", filename)
    if match:
        return match.group(1)
    return None

def get_parq_name(
    readme_path: Path = Path("Default_DataBase_README.txt"),
) -> str:
    """
    Parse `Default_DataBase_README.txt` to locate the FIRST usable `.parq` entry.

    Parameters
    ----------
    readme_path : Path
        Path to the README text file.

    Returns
    -------
    str
        Basename of the discovered .parq file (e.g., 'grid_database_1111.parq').

    Examples
    --------
    # Default_DataBase_README.txt
    # Lines starting with '#' are comments
    """
    if not readme_path.exists():
        raise FileNotFoundError(f"README file not found: {readme_path.resolve()}")

    with readme_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()

            # Skip comments and blanks
            if not line or line.startswith("#"):
                continue

            # Accept the first token that ends with '.parq' (case-insensitive)
            # Split on whitespace to be tolerant of trailing notes
            tokens = line.split()
            # Check tokens from left to right; most users will put the path/filename first
            for tok in tokens:
                if tok.lower().endswith(".parq"):
                    parq_path = Path(tok)
                    # Return only the basename, per requirement (e.g., 'grid_database_1111.parq')
                    result = parq_path.name
                    return result

    raise ValueError(
        "No valid '.parq' entry found in README. "
        "Ensure there is at least one non-comment line ending with '.parq'."
    )

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
