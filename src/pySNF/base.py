import time
from pathlib import Path
from typing import Sequence, Tuple, Union, Optional

import numpy as np
import pandas as pd
import math
from utils import plot_Gram_Ci, Converter_MTU2ASSY
from io_file import (
    create_output_dir, get_grid_space, get_snfs_dir_path, get_dhst_path, get_grid_ParqFile_path
)


class SNFProcessor:
    """
    Encapsulates logic for a single SNF:
      • DHST (decay heat & source-term) interpolation across target year
      • Nuclide concentration interpolation (grams per MTU / per assembly)
      • Nuclide activity interpolation (Ci per MTU / per assembly)
      • Writing results to a single Excel workbook
      • Generating per-SNF plots (weight/activity)

    Notes
    -----
    - Behavior is intentionally preserved: same column names/order, formatting,
      plotting calls, and output file naming.
    - DataFrames returned by compute_* methods use scientific notation strings
      where the original did, so UI/table rendering remains identical.
    """

    # Fixed years used to bracket the requested target year (relative to 2022)
    _YEARS_BRACKETS: Sequence[int] = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)

    def __init__(
        self,
        series_name: str,
        target_year: float,
        method: str = "log10",
        data_dir: Path = get_snfs_dir_path(),
        st_dataset_path: Path = get_dhst_path(),
    ) -> None:
        self.year: float = target_year
        self.method: str = method
        self.SNF_id: str = series_name

        # Load DHST master table and per-SNF per-nuclide tables
        self.df_DHST_all: pd.DataFrame = pd.read_csv(st_dataset_path, index_col=False)
        self.df_DHST_filtered: pd.DataFrame = self.df_DHST_all[
            self.df_DHST_all["SNF_id"] == self.SNF_id
        ]
        # Expect one row; keep behavior the same even if >1
        self.data_conc: pd.DataFrame = pd.read_csv(
            data_dir / f"{self.SNF_id}_gpMTU.csv",
            encoding="utf-8-sig",
            index_col=0,
        )
        self.data_Ci: pd.DataFrame = pd.read_csv(
            data_dir / f"{self.SNF_id}_CipMTU.csv",
            encoding="utf-8-sig",
            index_col=0,
        )

        # Interpolation bracket and numeric formatting precision
        self.decay_bounds: Tuple[int, int] = self._find_decay_bounds()
        self.round_num: int = 3

    # ────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────────
    def _find_decay_bounds(self) -> Tuple[int, int]:
        """
        Find the two nearest bracket years (in years since 2022) that enclose
        the target year. Fallback to the last bracket (200–500) if outside.
        """
        span = self.year - 2022.0
        years = list(self._YEARS_BRACKETS)
        for lower, upper in zip(years, years[1:]):
            if lower <= span <= upper:
                return lower, upper
        return years[-2], years[-1]  # fallback (200–500)

    @staticmethod
    def interpolate(
        target: float,
        lower_val: float,
        upper_val: float,
        bounds: Tuple[int, int],
        method: str = "log10",
    ) -> float:
        """
        Linear interpolation of a value between two bracketing points (lower/upper)
        with either a linear or log10 transform applied to the *time axis*.

        Parameters
        ----------
        target : float
            Absolute target year (e.g., 2025.0).
        lower_val, upper_val : float
            Values at the lower/upper time bounds.
        bounds : (int, int)
            Lower and upper bounds in years since 2022 (e.g., (0, 1), (1, 2), ...).
        method : {"log10", "linear"}
            Interpolation performed on the time axis; the values remain untransformed.

        Returns
        -------
        float
            Interpolated value at `target`.
        """
        t = target - 2022.0
        if method == "log10":
            # log(0) → very negative number to avoid -inf and keep monotonicity
            def _log(x: float) -> float:
                return np.log10(x) if x > 0 else -1e10

            t, lo, hi = _log(t), _log(bounds[0]), _log(bounds[1])
        else:
            lo, hi = bounds

        # Linear interpolation on the (possibly transformed) axis
        return lower_val + (upper_val - lower_val) * (t - lo) / (hi - lo)

    # ────────────────────────────────────────────────────────────────────────
    # Public computations
    # ────────────────────────────────────────────────────────────────────────
    def compute_dhst(self) -> pd.DataFrame:
        """
        Build the one-row DHST table for this SNF at the target year.
        Columns are kept identical to the original implementation.

        Returns
        -------
        pd.DataFrame
            Columns:
            ["DH(Watts/assy.)", "FN(n/s/assy.)",
             "HG(r/s/kgSS304/MTU)", "FG(r/s/assy.)"]
            Values are strings in scientific notation (e.g., "1.234e+05").
        """
        cols = [
            "DH(Watts/assy.)",
            "FN(n/s/assy.)",
            "FG(r/s/assy.)",
            "HG(r/s/kgSS304/MTU)",
        ]

        # Expect exactly one row in df_DHST_filtered; preserve original behavior
        df_DHST = pd.DataFrame(columns=cols)
        result_vals: list[str] = []

        for label in cols:
            # Component name before "(" : "DH" | "FN" | "HG" | "FG"
            component = label.split("(")[0]
            lo_col = f"{component}_{self.decay_bounds[0]}y"
            hi_col = f"{component}_{self.decay_bounds[1]}y"

            lo = self.df_DHST_filtered[lo_col].values[0]
            hi = self.df_DHST_filtered[hi_col].values[0]

            val = self.interpolate(self.year, lo, hi, self.decay_bounds, self.method)

            # Convert MTU → assy except for HG (kept identical to original logic)
            if component != "HG":
                val = Converter_MTU2ASSY(val, self.df_DHST_filtered)

            result_vals.append(f"{val:.3e}")

        df_DHST.loc[len(df_DHST)] = result_vals
        return df_DHST

    def compute_concentration(self) -> pd.DataFrame:
        """
        Interpolate per-nuclide concentrations; return MTU and assy values.
        Values are formatted as scientific-notation strings (original behavior).
        """
        rows: list[tuple[str, float, float]] = []
        for nuclide, row in self.data_conc.iterrows():
            lo = row[f"{self.decay_bounds[0]}y"].item()
            hi = row[f"{self.decay_bounds[1]}y"].item()
            conc = round(
                self.interpolate(self.year, lo, hi, self.decay_bounds, self.method),
                self.round_num,
            )
            conc_assy = round(Converter_MTU2ASSY(conc, self.df_DHST_filtered), self.round_num)
            rows.append((str(nuclide), conc, conc_assy))

        df = pd.DataFrame(rows, columns=["nuclide", "gram/MTU", "gram/assy."])
        df["gram/MTU"] = df["gram/MTU"].map(lambda x: f"{x:.2e}")
        df["gram/assy."] = df["gram/assy."].map(lambda x: f"{x:.2e}")
        return df

    def compute_activity(self) -> pd.DataFrame:
        """
        Interpolate per-nuclide activities; return MTU and assy values.
        Keeps the original row reordering: top two rows (after sorting by 'Ci/MTU'
        descending) are considered subtotal/total and moved to the bottom.
        """
        rows: list[tuple[str, float, float]] = []
        for nuclide, row in self.data_Ci.iterrows():
            lo = row[f"{self.decay_bounds[0]}y"].item()
            hi = row[f"{self.decay_bounds[1]}y"].item()
            act = round(
                self.interpolate(self.year, lo, hi, self.decay_bounds, self.method),
                self.round_num,
            )
            act_assy = round(Converter_MTU2ASSY(act, self.df_DHST_filtered), self.round_num)
            rows.append((str(nuclide), act, act_assy))

        df = pd.DataFrame(rows, columns=["nuclide", "Ci/MTU", "Ci/assy."])
        df = df[df["Ci/MTU"] > 0].sort_values("Ci/MTU", ascending=False)

        # Move total & subtotal to bottom (preserves original ordering rule)
        total, subtotal, rest = df.iloc[:1], df.iloc[1:2], df.iloc[2:]
        df = pd.concat([rest, subtotal, total])

        df["Ci/MTU"] = df["Ci/MTU"].map(lambda x: f"{x:.2e}")
        df["Ci/assy."] = df["Ci/assy."].map(lambda x: f"{x:.2e}")
        return df

    # ────────────────────────────────────────────────────────────────────────
    # Output helpers
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def write_excel(
        df_dhst: pd.DataFrame,
        df_act: pd.DataFrame,
        df_conc: pd.DataFrame,
        output_dir: Union[str, Path],
        file_name: str,
    ) -> None:
        """
        Write three sheets (DHST / Concentration / Activity) to a single workbook.
        Overwrites the file if it exists (same as original).
        """
        path = Path(output_dir, f"{file_name}.xlsx")
        if path.exists():
            path.unlink()
        with pd.ExcelWriter(path) as writer:
            df_dhst.to_excel(writer, sheet_name="DHST", index=False)
            df_conc.to_excel(writer, sheet_name="Concentration", index=False)
            df_act.to_excel(writer, sheet_name="Activity", index=False)

    def plot_single_SNF(self, output_dir: Union[str, Path]) -> None:
        """Generate two per-SNF plots (weight and activity)."""
        plot_Gram_Ci(
            self.data_conc,
            self.SNF_id,
            "Weight",
            "Weight (g/assy.)",
            self.df_DHST_filtered,
            output_dir,
        )
        plot_Gram_Ci(
            self.data_Ci,
            self.SNF_id,
            "Activity",
            "Activity (Ci/assy.)",
            self.df_DHST_filtered,
            output_dir,
        )

    def run(self) -> None:
        """
        Convenience runner used elsewhere in the app:
          - compute DHST, concentration, and activity
          - create an output directory
          - write Excel
          - print a short timing message
          - create two interactive plots (same calls as plot_single_SNF but with
            original axis labels preserved)
        """
        start = time.time()

        st = self.compute_dhst()
        conc = self.compute_concentration()
        act = self.compute_activity()

        output_dir = create_output_dir(parent_folder_name="Results_Single")
        self.write_excel(st, act, conc, output_dir, self.SNF_id)

        print(f"\nResults for {self.SNF_id} at year {self.year}:")
        print(f"Elapsed: {time.time() - start:.2f}s")

        # Interactive plotting (kept exactly as before)
        plot_Gram_Ci(
            self.data_conc,
            self.SNF_id,
            "Weight",
            "Concentration(g)",
            self.df_DHST_filtered,
            output_dir,
        )
        plot_Gram_Ci(
            self.data_Ci,
            self.SNF_id,
            "Activity",
            "Activity(Ci)",
            self.df_DHST_filtered,
            output_dir,
        )


class PredictSNFs_interpolate:
    """
    Perform 4D linear interpolation over a rectilinear grid of
    (Enrich, SP, Burnup, Cool) → output columns.

    Assumptions
    -----------
    - `grid_df`'s first four columns are the axes in the order:
      Enrich, SP, Burnup, Cool, followed by output columns.
    - Spaces provided are sorted ascending and cover the interpolation range.
    - Behavior preserved: rounding/normalization is identical to the original.
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
        self.grid: pd.DataFrame = grid_df.copy()
        # Default order according to grid_database.parq 
        self.grid.columns = ["Enrich", "SP", "Burnup", "Cool"] + list([f"{p}_prediction" for p in ("DH", "FN", "HG", "FG")])
        # Reset column order with output_cols at the end
        self.grid.reindex(columns=["Enrich", "SP", "Burnup", "Cool"] + list(output_cols))

        # Normalize dtypes & precision on all four axes
        for col in ("Enrich", "SP", "Burnup"):
            self.grid[col] = self.grid[col].astype(float)
        # Round Cool to 6 decimals (keeps original behavior)
        self.grid["Cool"] = self.grid["Cool"].astype(float).round(6)

        # Round the space arrays equally (important for exact MultiIndex matching)
        self.enrichment_space: np.ndarray = np.round(enrichment_space.astype(float), 6)
        self.specific_power_space: np.ndarray = np.round(specific_power_space.astype(float), 6)
        self.burnup_space: np.ndarray = np.round(burnup_space.astype(float), 6)
        self.cooling_time_space: np.ndarray = np.round(cooling_time_space.astype(float), 6)

        # Build the MultiIndex after rounding
        self.grid.set_index(["Enrich", "SP", "Burnup", "Cool"], inplace=True)

        # Keep output column names in a list for consistent ordering
        self.output_cols: list[str] = list(output_cols)

        # Target coordinates are set during interpolate()
        self.enrichment: Optional[float] = None
        self.burnup: Optional[float] = None
        self.specific_power: Optional[float] = None
        self.cooling_time: Optional[float] = None

    # ────────────────────────────────────────────────────────────────────────
    # Low-level interpolation primitives
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _linear_interpolate(
        x0: float, x1: float, y0: np.ndarray, y1: np.ndarray, x: float
    ) -> np.ndarray:
        """Linear interpolation between y0@x0 and y1@x1 for target x."""
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)

    @staticmethod
    def _find_bounds(value: float, grid: np.ndarray) -> Tuple[float, float]:
        """
        Locate the two nearest grid points that bracket `value`.
        Returns the edge pair when `value` is outside the grid.
        """
        if value <= grid[0]:
            return grid[0], grid[1]
        if value >= grid[-1]:
            return grid[-2], grid[-1]
        idx = np.searchsorted(grid, value) - 1
        return grid[idx], grid[idx + 1]

    def _get_reduced_grid(self) -> np.ndarray:
        """
        Extract the 16 corner points around current (E, SP, BU, Cool)
        and return an array of shape (2, 2, 2, 2, n_outputs).
        Missing corners are filled with NaN (kept as in original).
        """
        # Bounds (computed once per call)
        e0, e1 = self._find_bounds(self.enrichment, self.enrichment_space)          # type: ignore[arg-type]
        sp0, sp1 = self._find_bounds(self.specific_power, self.specific_power_space)  # type: ignore[arg-type]
        b0, b1 = self._find_bounds(self.burnup, self.burnup_space)                  # type: ignore[arg-type]
        c0, c1 = self._find_bounds(self.cooling_time, self.cooling_time_space)      # type: ignore[arg-type]

        n_out = len(self.output_cols)
        grid_vals = np.empty((2, 2, 2, 2, n_out), dtype=float)

        # Direct MultiIndex lookup for each corner
        for i, e in enumerate((e0, e1)):
            for j, sp in enumerate((sp0, sp1)):
                for k, b in enumerate((b0, b1)):
                    for m, c in enumerate((c0, c1)):
                        try:
                            row = self.grid.loc[(e, sp, b, c), self.output_cols]
                            grid_vals[i, j, k, m] = row.values
                        except KeyError:
                            grid_vals[i, j, k, m] = np.nan
        return grid_vals

    # ────────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────────
    def interpolate(
        self,
        enrichment: float,
        burnup: float,
        specific_power: float,
        cooling_time: float,
    ) -> pd.Series:
        """
        Perform hierarchical linear interpolation across the 4 axes in order:
        1) Cool, 2) Burnup, 3) Specific Power, 4) Enrichment.

        Returns
        -------
        pd.Series
            Final interpolated outputs with index = self.output_cols.
        """
        # Target coordinates (rounded to match grid precision)
        self.enrichment = float(enrichment)
        self.burnup = float(burnup)
        self.specific_power = float(specific_power)
        self.cooling_time = round(float(cooling_time), 6)

        # Extract the 16 corner values
        corner = self._get_reduced_grid()

        # Axis bounds (again, once)
        e0, e1 = self._find_bounds(self.enrichment, self.enrichment_space)
        sp0, sp1 = self._find_bounds(self.specific_power, self.specific_power_space)
        b0, b1 = self._find_bounds(self.burnup, self.burnup_space)
        c0, c1 = self._find_bounds(self.cooling_time, self.cooling_time_space)

        n_out = len(self.output_cols)

        # 1) Interpolate along the Cool axis → (2, 2, 2, n_out)
        interp_c = np.empty((2, 2, 2, n_out), dtype=float)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    y0, y1 = corner[i, j, k, 0], corner[i, j, k, 1]
                    interp_c[i, j, k] = self._linear_interpolate(c0, c1, y0, y1, self.cooling_time)

        # 2) Burnup axis → (2, 2, n_out)
        interp_b = np.empty((2, 2, n_out), dtype=float)
        for i in range(2):
            for j in range(2):
                y0, y1 = interp_c[i, j, 0], interp_c[i, j, 1]
                interp_b[i, j] = self._linear_interpolate(b0, b1, y0, y1, self.burnup)

        # 3) Specific Power axis → (2, n_out)
        interp_sp = np.empty((2, n_out), dtype=float)
        for i in range(2):
            y0, y1 = interp_b[i, 0], interp_b[i, 1]
            interp_sp[i] = self._linear_interpolate(sp0, sp1, y0, y1, self.specific_power)

        # 4) Enrichment axis → (n_out,)
        final = self._linear_interpolate(e0, e1, interp_sp[0], interp_sp[1], self.enrichment)

        return pd.Series(final, index=self.output_cols)

def run_PredAssy() -> PredictSNFs_interpolate:

    grid_data: pd.DataFrame = pd.read_parquet(get_grid_ParqFile_path())
    grid_space = get_grid_space()  # e.g. "1412"
    enrich_factor = int(grid_space[0])
    sp_factor = int(grid_space[1])
    bp_factor = int(grid_space[2])
    cool_factor = int(grid_space[3])

    enrich_space = np.arange(1.5, 6.1, 0.5)
    enrich_space = enrich_space[0::enrich_factor]
    sp_space = np.arange(5, 46, 5)
    sp_space = sp_space[0::sp_factor]
    burnup_space = np.arange(5000, 74100, 3000)
    burnup_space = burnup_space[0::bp_factor]
    cool_space_raw = np.logspace(-5.75, 6.215, 150, base=math.e)
    cool_space = cool_space_raw[1::cool_factor]

    out_cols = [f"{p}_prediction" for p in ("DH", "FN", "FG", "HG")]

    # Interpolator
    PredAssy = PredictSNFs_interpolate(
        grid_data,
        enrich_space,
        sp_space,
        burnup_space,
        cool_space,
        out_cols,
    )
    return PredAssy
