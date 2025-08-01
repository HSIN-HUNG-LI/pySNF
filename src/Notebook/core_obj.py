import pandas as pd
from pathlib import Path
from typing import Union, List


class SNFParquetConverter:
    """
    A converter that reads CSV or Excel files (either in a folder
    or as a single file), concatenates them into one DataFrame
    with a 'source_file' column, and writes to Parquet.
    """

    def __init__(self, input_folder: Union[str, Path], output_file: Union[str, Path]):
        """
        :param input_folder: Path to a folder of CSV/XLSX files or to one file.
        :param output_file: Path where the combined Parquet will be saved.
        """
        self.input_folder = Path(input_folder)
        self.output_file = Path(output_file)
        
    def convert_single_to_parquet(self) -> None:
        files_path = self.input_folder
        suffix = files_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(files_path, index_col=0, encoding="utf-8-sig", memory_map=True)
        else:  # .xls or .xlsx
            df = pd.read_excel(files_path, index_col=0, engine="openpyxl")
        df.to_parquet(self.output_file)
        print(
            f"Wrote single file to parq {df.shape[0]} rows "
            f"from {len(df)} file(s) into '{self.output_file}'."
        )

    def convert_to_parquet(self) -> None:
        """
        - If `input_folder` is a file, read just that one.
        - Otherwise, scan for *.csv, *.xls, *.xlsx under `input_folder`.
        - Read each file with the first column as index.
        - Tag each row with `source_file`.
        - Concatenate and write to Parquet (preserve index).
        """
        # 1) Decide whether we're given a single file or a directory
        if self.input_folder.is_file():
            file_paths: List[Path] = [self.input_folder]
        else:
            patterns = ["*.csv", "*.xls", "*.xlsx"]
            file_paths = [
                p for pattern in patterns for p in self.input_folder.glob(pattern)
            ]

        if not file_paths:
            raise ValueError(
                f"No CSV or Excel files found in '{self.input_folder}'."
            )

        dfs: List[pd.DataFrame] = []
        for path in file_paths:
            # 2) Read depending on extension
            suffix = path.suffix.lower()
            if suffix == ".csv":
                df = pd.read_csv(path, index_col=0, encoding="utf-8-sig", memory_map=True)
            else:  # .xls or .xlsx
                df = pd.read_excel(path, index_col=0, engine="openpyxl")

            # 3) Record source file and collect
            df["source_file"] = path.name
            dfs.append(df)

        # 4) Concatenate and write
        combined = pd.concat(dfs, axis=0)
        combined.to_parquet(self.output_file)

        print(
            f"Wrote {combined.shape[0]} rows "
            f"from {len(file_paths)} file(s) into '{self.output_file}'."
        )

    @staticmethod
    def read_parquet(parquet_file: Union[str, Path]) -> pd.DataFrame:
        """Read the entire Parquet into a DataFrame."""
        return pd.read_parquet(parquet_file)

    @staticmethod
    def read_by_source(
        parquet_file: Union[str, Path],
        source_filename: str
    ) -> pd.DataFrame:
        """
        Return only the rows whose `source_file` column matches `source_filename`.
        Always returns a DataFrame (even if empty).
        """
        # 1) Read the full Parquet into a DataFrame
        df = pd.read_parquet(parquet_file)

        # 2) Boolean mask for matching rows
        mask = df["source_file"] == source_filename

        # 3) Wrap the selection in pd.DataFrame(...) to force the return type
        filtered = pd.DataFrame(df.loc[mask, :])

        return filtered
