import time
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd

from utils import plot_Gram_Ci, Converter_MTU2ASSY
from io_file import create_output_dir, get_snfs_dir_path, get_stdh_path

class SNFProcessor:
    """
    Encapsulates all logic for:
      - decay heat / source-term interpolation (STDH)
      - concentration extraction
      - activity extraction
      - Excel writing
      - plotting
    """

    def __init__(
        self,
        series_name: str,
        target_year: float,
        method: str = "log10",
        data_dir: Path = get_snfs_dir_path(),
        st_dataset_path: Path = get_stdh_path(),
    ):
        self.year = target_year
        self.method = method

        # Load source-term dataset// data_dir: Path = Path.cwd() / "snfs_details"/ "all_snfs_details.parquet",
        self.df_STDH_all = pd.read_csv(st_dataset_path, index_col=False)
        self.SNF_id = series_name
        self.series_name = self.get_name_by_snf_id(self.df_STDH_all, series_name)
        self.df_STDH_filtered = self.df_STDH_all[
            self.df_STDH_all["Name"] == self.series_name
        ]
        self.data_conc = pd.read_csv(
            data_dir / f"{self.series_name}_gpMTU.csv", encoding="utf-8-sig", index_col=0
        )
        self.data_Ci = pd.read_csv(
            data_dir / f"{self.series_name}_CipMTU.csv", encoding="utf-8-sig", index_col=0
        )

        # Determine decay bounds bracket
        self.decay_bounds = self._find_decay_bounds()
        self.round_num = 3

    def _find_decay_bounds(self) -> tuple[int, int]:
        """Find the two nearest bracket years around (year - 2022)."""
        span = self.year - 2022.0
        years = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        for lower, upper in zip(years, years[1:]):
            if lower <= span <= upper:
                return lower, upper
        return years[-2], years[-1]  # fallback to 200–500
    
    @staticmethod
    def get_name_by_snf_id(df: pd.DataFrame, snf_id: str) -> str:
        """
        Return the Name corresponding to the given SNF_id.
        If SNF_id is not found, returns None.
        """
        # Filter rows where SNF_id matches, then take the first Name
        match = df.loc[df['SNF_id'] == snf_id, 'Name']
        return match.iat[0] if not match.empty else "None"

    @staticmethod
    def interpolate(
        target: float,
        lower_val: float,
        upper_val: float,
        bounds: tuple[int, int],
        method: str = "log10",
    ) -> float:
        """Linear or log10 interpolation between lower_val and upper_val."""
        t = target - 2022.0
        if method == "log10":
            # map zero to a large negative
            def log(x):
                return np.log10(x) if x > 0 else -1e10

            t, lo, hi = log(t), log(bounds[0]), log(bounds[1])
        else:
            lo, hi = bounds
        return lower_val + (upper_val - lower_val) * (t - lo) / (hi - lo)

    def compute_stdh(self) -> pd.DataFrame:
        """Builds the STDH table for this series."""
        cols = [
            "DH(Watts/assy.)",
            "FN(n/s/assy.)",
            "HG(r/s/kgSS304/MTU)",
            "FG(r/s/assy.)",
        ]
        df_STDH = pd.DataFrame(columns=cols)
        result_ls = []
        for component in cols:
            component = component.split("(")[0]
            lo = self.df_STDH_filtered[f"{component}_{self.decay_bounds[0]}y"].values[0]
            hi = self.df_STDH_filtered[f"{component}_{self.decay_bounds[1]}y"].values[0]
            val = self.interpolate(self.year, lo, hi, self.decay_bounds, self.method)
            # convert MTU→assy unless HG
            if component != "HG":
                val = Converter_MTU2ASSY(val, self.df_STDH_filtered)
            result_ls.append(f"{val:.3e}")
        df_STDH.loc[len(df_STDH)] = result_ls

        return df_STDH

    def compute_concentration(self) -> pd.DataFrame:
        """Interpolates concentrations for each nuclide."""
        rows = []
        for nuclide, row in self.data_conc.iterrows():
            lo = row[f"{self.decay_bounds[0]}y"].item()
            hi = row[f"{self.decay_bounds[1]}y"].item()
            conc = round(
                self.interpolate(self.year, lo, hi, self.decay_bounds, self.method),
                self.round_num,
            )
            conc_assy = round(
                Converter_MTU2ASSY(conc, self.df_STDH_filtered), self.round_num
            )
            rows.append((nuclide, conc, conc_assy))
        df = pd.DataFrame(rows, columns=["nuclide", "gram/MTU", "gram/assy."])
        df["gram/MTU"] = df["gram/MTU"].map(lambda x: f"{x:.2e}")
        df["gram/assy."] = df["gram/assy."].map(lambda x: f"{x:.2e}")
        return df

    def compute_activity(self) -> pd.DataFrame:
        """Interpolates activities for each nuclide."""
        rows = []
        for nuclide, row in self.data_Ci.iterrows():
            lo = row[f"{self.decay_bounds[0]}y"].item()
            hi = row[f"{self.decay_bounds[1]}y"].item()
            act = round(
                self.interpolate(self.year, lo, hi, self.decay_bounds, self.method),
                self.round_num,
            )
            act_assy = round(
                Converter_MTU2ASSY(act, self.df_STDH_filtered), self.round_num
            )
            rows.append((nuclide, act, act_assy))
        df = pd.DataFrame(rows, columns=["nuclide", "Ci/MTU", "Ci/assy."])
        df = df[df["Ci/MTU"] > 0].sort_values("Ci/MTU", ascending=False)
        # move total & subtotal to bottom
        total, subtotal, rest = df.iloc[:1], df.iloc[1:2], df.iloc[2:]
        df = pd.concat([rest, subtotal, total])
        df["Ci/MTU"] = df["Ci/MTU"].map(lambda x: f"{x:.2e}")
        df["Ci/assy."] = df["Ci/assy."].map(lambda x: f"{x:.2e}")
        return df
    @staticmethod
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

    def plot_single_SNF(self, output_dir: Union[str, Path]) -> None:
        plot_Gram_Ci(
            self.data_conc,
            self.SNF_id,
            f"Weight",
            "Weight (g/assy.)",
            self.df_STDH_filtered,
            output_dir,
        )
        plot_Gram_Ci(
            self.data_Ci,
            self.SNF_id,
            f"Activity",
            "Activity (Ci/assy.)",
            self.df_STDH_filtered,
            output_dir,
        )

    def run(self):
        start = time.time()

        st = self.compute_stdh()
        conc = self.compute_concentration()
        act = self.compute_activity()

        # Prepare output directory
        output_dir = create_output_dir(parent_folder_name="Results_Single")

        self.write_excel(st, act, conc, output_dir, self.series_name)

        print(f"\nResults for {self.series_name} at year {self.year}:")
        print(f"Elapsed: {time.time() - start:.2f}s")

        # interactive plotting
        plot_Gram_Ci(
            self.data_conc,
            self.series_name,
            f"Weight",
            "Concentration(g)",
            self.df_STDH_filtered,
            output_dir,
        )
        plot_Gram_Ci(
            self.data_Ci,
            self.series_name,
            f"Activity",
            "Activity(Ci)",
            self.df_STDH_filtered,
            output_dir,
        )

