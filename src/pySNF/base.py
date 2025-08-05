import time
from pathlib import Path
from typing import Sequence, Tuple, Union
import numpy as np
import pandas as pd
from itertools import product
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

        self.df_STDH_all = pd.read_csv(st_dataset_path, index_col=False)
        self.SNF_id = series_name
        self.df_STDH_filtered = self.df_STDH_all[
            self.df_STDH_all["SNF_id"] == self.SNF_id
        ]
        self.data_conc = pd.read_csv(
            data_dir / f"{self.SNF_id}_gpMTU.csv",
            encoding="utf-8-sig",
            index_col=0,
        )
        self.data_Ci = pd.read_csv(
            data_dir / f"{self.SNF_id}_CipMTU.csv",
            encoding="utf-8-sig",
            index_col=0,
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

        self.write_excel(st, act, conc, output_dir, self.SNF_id)

        print(f"\nResults for {self.SNF_id} at year {self.year}:")
        print(f"Elapsed: {time.time() - start:.2f}s")

        # interactive plotting
        plot_Gram_Ci(
            self.data_conc,
            self.SNF_id,
            f"Weight",
            "Concentration(g)",
            self.df_STDH_filtered,
            output_dir,
        )
        plot_Gram_Ci(
            self.data_Ci,
            self.SNF_id,
            f"Activity",
            "Activity(Ci)",
            self.df_STDH_filtered,
            output_dir,
        )


class PredictSNFs_interpolate:
    """
    Perform 4D interpolation over grid data for given fuel parameters.
    """

    def __init__(
        self,
        grid_df: pd.DataFrame,
        enrichment_space: np.ndarray,
        specific_power_space: np.ndarray,
        burnup_space: np.ndarray,
        cooling_time_space: np.ndarray,
        output_cols: Sequence[str],
    ) -> None:
        # Copy and rename incoming columns for consistency
        # Assumes the first four columns correspond to the 4 axes
        self.grid = grid_df.copy()
        self.grid.columns = ["Enrich", "SP", "Burnup", "Cool"] + list(output_cols)

        # Normalize dtypes & precision** on all four axes
        for col in ("Enrich", "SP", "Burnup"):
            self.grid[col] = self.grid[col].astype(float)
        # round “Cool” to 6 decimals
        self.grid["Cool"] = self.grid["Cool"].astype(float).round(6)

        # Also round your target and your space arrays

        self.enrichment_space = np.round(enrichment_space.astype(float), 6)
        self.specific_power_space = np.round(specific_power_space.astype(float), 6)
        self.burnup_space = np.round(burnup_space.astype(float), 6)
        self.cooling_time_space = np.round(cooling_time_space.astype(float), 6)

        # Build the index *after* rounding
        self.grid.set_index(["Enrich", "SP", "Burnup", "Cool"], inplace=True)

        # Keep output column names in a list
        self.output_cols = list(output_cols)

    @staticmethod
    def _linear_interpolate(
        x0: float, x1: float, y0: np.ndarray, y1: np.ndarray, x: float
    ) -> np.ndarray:
        """
        Linearly interpolate between y0 @ x0 and y1 @ x1 for target x.
        """
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    @staticmethod
    def _find_bounds(value: float, grid: np.ndarray) -> Tuple[float, float]:
        """
        Locate the two nearest grid points around `value`.
        """
        if value <= grid[0]:
            return grid[0], grid[1]
        if value >= grid[-1]:
            return grid[-2], grid[-1]
        idx = np.searchsorted(grid, value) - 1
        return grid[idx], grid[idx + 1]

    def _get_reduced_grid(self) -> np.ndarray:
        """
        Extract the 16 corner points around our target parameters
        and return an array of shape (2,2,2,2,n_outputs).
        """
        # Find lower & upper bounds for each axis once
        e0, e1 = self._find_bounds(self.enrichment, self.enrichment_space)
        sp0, sp1 = self._find_bounds(self.specific_power, self.specific_power_space)
        b0, b1 = self._find_bounds(self.burnup, self.burnup_space)
        c0, c1 = self._find_bounds(self.cooling_time, self.cooling_time_space)

        # Pre‐allocate the 5D result array
        n_out = len(self.output_cols)
        grid_vals = np.empty((2, 2, 2, 2, n_out), dtype=float)

        # Fill the array by direct MultiIndex lookup
        for i, e in enumerate((e0, e1)):
            for j, sp in enumerate((sp0, sp1)):
                for k, b in enumerate((b0, b1)):
                    for m, c in enumerate((c0, c1)):
                        try:
                            row = self.grid.loc[(e, sp, b, c), self.output_cols]
                            grid_vals[i, j, k, m] = row.values
                        except KeyError:
                            # If missing, fill with NaN
                            grid_vals[i, j, k, m] = np.nan

        return grid_vals

    def interpolate(
        self,
        enrichment: float,
        burnup: float,
        specific_power: float,
        cooling_time: float,
    ) -> pd.Series:
        """
        Perform the 4-step hierarchical interpolation:
        1) along Cool, 2) then Burnup, 3) then SP, 4) then Enrich.
        Returns the final interpolated outputs as a pandas Series.
        """

        self.enrichment = float(enrichment)
        self.burnup = float(burnup)
        self.specific_power = float(specific_power)
        self.cooling_time = round(cooling_time, 6)

        # Extract the 16 corner values
        corner = self._get_reduced_grid()
        
        # Precompute all axis bounds
        e0, e1 = self._find_bounds(self.enrichment, self.enrichment_space)
        sp0, sp1 = self._find_bounds(self.specific_power, self.specific_power_space)
        b0, b1 = self._find_bounds(self.burnup, self.burnup_space)
        c0, c1 = self._find_bounds(self.cooling_time, self.cooling_time_space)

        n_out = len(self.output_cols)

        # 1) Interpolate along the Cool axis → shape (2,2,2,n_out)
        interp_c = np.empty((2, 2, 2, n_out), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    y0 = corner[i, j, k, 0]
                    y1 = corner[i, j, k, 1]
                    interp_c[i, j, k] = self._linear_interpolate(
                        c0, c1, y0, y1, self.cooling_time
                    )

        # 2) Interpolate along the Burnup axis → shape (2,2,n_out)
        interp_b = np.empty((2, 2, n_out), dtype=float)
        for i in range(2):
            for j in range(2):
                y0 = interp_c[i, j, 0]
                y1 = interp_c[i, j, 1]
                interp_b[i, j] = self._linear_interpolate(b0, b1, y0, y1, self.burnup)

        # 3) Interpolate along the Specific Power axis → shape (2,n_out)
        interp_sp = np.empty((2, n_out), dtype=float)
        for i in range(2):
            y0 = interp_b[i, 0]
            y1 = interp_b[i, 1]
            interp_sp[i] = self._linear_interpolate(
                sp0, sp1, y0, y1, self.specific_power
            )

        # 4) Final interpolation along the Enrichment axis → shape (n_out,)
        final = self._linear_interpolate(
            e0, e1, interp_sp[0], interp_sp[1], self.enrichment
        )

        return pd.Series(final, index=self.output_cols)
