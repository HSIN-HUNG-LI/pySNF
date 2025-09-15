import threading
import time
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path
from typing import Literal, Optional, List, Tuple

import numpy as np
import pandas as pd

from base import run_PredAssy
from FrameViewer.BaseFrame import DataFrameViewer
from visualize import plot_4x4_scatterplot, plot_dhst_RelativeError_boxplots
from io_file import (
    load_dataset,
    save_PredData,
    get_dhst_path,
    create_output_dir,
)


class PredictionFrame(tk.Frame):
    """
    Frame to
    (1) predict Decay Heat & Source Terms (DH&STs) from fuel parameters,
    (2) preview the results,
    (3) optionally export data/figures, and
    (4) run verification plots against a reference dataset.
    """

    # Columns shown in the prediction preview table
    cols_all: list[str] = [
        "s/n",
        "DH (W/assy.)",
        "FN (n/s/assy.)",
        "FG (r/s/assy.)",
        "HG (r/s/kgSS304/MTU)",
    ]

    # Required input features/order
    input_required: list[str] = ["Enrich", "SP", "Burnup", "Cool"]

    def __init__(self, parent: tk.Misc, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        # Persistent state
        self.PredAssy = run_PredAssy()
        self.database_dhst = pd.read_csv(get_dhst_path())
        self.save_var = tk.BooleanVar(value=True)
        self._running: bool = False

        # Attributes initialized for safety (avoid AttributeError in edge paths)
        self.df_in: Optional[pd.DataFrame] = None
        self.df_in_copy: Optional[pd.DataFrame] = None
        self.df_verify_result: pd.DataFrame = pd.DataFrame()
        self.df_path: Optional[Path] = None
        self.n_snfs: int = 0
        self._dlg: Optional[tk.Toplevel] = None
        self.elapsed_label: Optional[tk.Label] = None
        self._start_time: float = 0.0

        # Default values to seed the input entries
        self.snf_stats: dict[str, float] = {
            "Enrich": 3.17,
            "SP": 26.21,
            "Burnup": 32806.18,
            "Cool": 24.36,
        }

        # Layout
        self._setup_scrollable_canvas()
        self._build_ui()

    # ────────────────────────────────────────────────────────────────────────
    # UI scaffolding
    # ────────────────────────────────────────────────────────────────────────
    def _setup_scrollable_canvas(self) -> None:
        """Create a scrollable canvas with an inner frame."""
        self.canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner = tk.Frame(self.canvas)
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self.window_id, width=e.width),
        )

    def _build_ui(self) -> None:
        """Set up controls, log areas, and DataFrame viewers."""
        # row0: verification
        row0 = tk.Frame(self.inner)
        row0.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(
            row0,
            text="Verification:",
            font=("Helvetica", 12, "bold"),
        ).pack(side=tk.LEFT)
        tk.Label(
            row0,
            text="A case study by comparing the pySNF predictions with the TRITON calculations for a selected set of SNFs.",
        ).pack(side=tk.LEFT)
        row0_1 = tk.Frame(self.inner)
        row0_1.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(row0_1, text="Run Test Case", command=self.verification).pack(
            side=tk.LEFT
        )
        tk.Label(
            row0_1,
            text="Check the results (SNFs_comparsion.png, SNFs_dataset.png, SNFs_prediction.png) in folder output/Prediction",
        ).pack(side=tk.LEFT)

        # Row: section label
        row = tk.Frame(self.inner)
        row.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(
            row,
            text="Predictions of SNF's Decay Heat & Source Terms:",
            font=("Helvetica", 12, "bold"),
        ).pack(side=tk.LEFT)

        # row1: four numeric inputs + Output + Save toggle
        row1_1 = tk.Frame(self.inner)
        row1_1.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(
            row1_1, text="(1) SNF spec: Enter the following information (Bp & Ct are required, En & Sp are optional.)  OR"
        ).pack(side=tk.LEFT)
        row1_2 = tk.Frame(self.inner)
        row1_2.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(row1_2, text="Bp (MWd/MTU):").pack(side=tk.LEFT)
        self.burnup_entry = tk.Entry(row1_2, width=10)
        self.burnup_entry.insert(0, f"{self.snf_stats['Burnup']}")
        self.burnup_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(row1_2, text="Ct (Year):").pack(side=tk.LEFT)
        self.year_entry = tk.Entry(row1_2, width=10)
        self.year_entry.insert(0, f"{self.snf_stats['Cool']}")
        self.year_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(row1_2, text="En (%U235):").pack(side=tk.LEFT)
        self.enrich_entry = tk.Entry(row1_2, width=10)
        self.enrich_entry.insert(0, f"{self.snf_stats['Enrich']}")
        self.enrich_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(row1_2, text="Sp (MW):").pack(side=tk.LEFT)
        self.sp_entry = tk.Entry(row1_2, width=10)
        self.sp_entry.insert(0, f"{self.snf_stats['SP']}")
        self.sp_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(row1_2, text="Output", command=self._on_output).pack(side=tk.LEFT)
        tk.Checkbutton(row1_2, text="Save output", variable=self.save_var).pack(
            side=tk.LEFT
        )

        # Row2: batch file load
        row2 = tk.Frame(self.inner)
        row2.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(
            row2,
            text="(2) SNFs Specs: Load a csv file, e.g., BatchPrediction_TSC01.csv",
        ).pack(side=tk.LEFT)
        tk.Button(row2, text="Load & Output", command=self.load_list).pack(side=tk.LEFT)

        # Prediction preview viewer
        self.DHST_viewer = self._make_viewer(
            parent=self.inner,
            height=300,
            columns=self.cols_all,
            title="Decay Heat & Source Terms",
        )

    def _make_viewer(
        self,
        parent: tk.Misc,
        height: int,
        columns: list[str],
        title: str,
        side: Literal["left", "right", "top", "bottom"] = tk.TOP,
        expand: bool = False,
    ) -> DataFrameViewer:
        """Instantiate a DataFrameViewer with fixed height."""
        frame = tk.Frame(parent, height=height)
        frame.pack(fill=tk.X, side=side, expand=expand, padx=5, pady=5)
        frame.pack_propagate(False)
        df_empty = pd.DataFrame(columns=columns)
        viewer = DataFrameViewer(frame, df_empty, title=title)
        viewer.pack(fill=tk.BOTH, expand=True)
        return viewer

    # ────────────────────────────────────────────────────────────────────────
    # Small helpers
    # ────────────────────────────────────────────────────────────────────────
    def _clear_viewers(self) -> None:
        """Clear all rows in the prediction preview tree."""
        for iid in self.DHST_viewer.tree.get_children():
            self.DHST_viewer.tree.delete(iid)

    def _insert_rows(self, viewer: DataFrameViewer, df: pd.DataFrame) -> None:
        """Insert DataFrame rows into a viewer."""
        for vals in df.itertuples(index=False, name=None):
            viewer.tree.insert("", "end", values=vals)

    def _show_running_dialog(self) -> None:
        """Modal dialog showing elapsed time and an option to cancel."""
        dlg = tk.Toplevel(self)
        dlg.title("Processing...")
        dlg.geometry("350x120")
        self.elapsed_label = tk.Label(dlg, text="Starting...")
        self.elapsed_label.pack(pady=10)
        tk.Button(
            dlg, text="Stop Running", command=lambda: setattr(self, "_running", False)
        ).pack()
        self._dlg = dlg
        self._start_time = time.time()
        self._update_timer()

    def _update_timer(self) -> None:
        """Update elapsed time every second until stopped."""
        if not getattr(self, "_running", False):
            # Be defensive in case the dialog was already closed
            if self._dlg is not None and self._dlg.winfo_exists():
                self._dlg.destroy()
            return
        elapsed = int(time.time() - self._start_time)
        if self.elapsed_label is not None:
            self.elapsed_label.config(
                text=f"Running {self.n_snfs} SNFs... \nElapsed: {elapsed}s\nIf compute ~6800 fuels, wait >360s,\n"
            )
        self.after(1000, self._update_timer)

    @staticmethod
    def _format(val) -> str:
        """Format numeric values in scientific notation; pass through non-numerics."""
        try:
            return f"{val:.2e}"
        except (ValueError, TypeError):
            return str(val)

    # ────────────────────────────────────────────────────────────────────────
    # Actions
    # ────────────────────────────────────────────────────────────────────────
    def get_parameters_dataframe(self) -> pd.DataFrame | None:
        """
        Read and validate the four Entry fields as floats.
        Returns a one-row DataFrame with columns ['Enrich','SP','Burnup','Cool'],
        or None (and shows an error) if any value is invalid.
        """
        fields = {
            "Burnup": self.burnup_entry,
            "Cool": self.year_entry,
            "Enrich": self.enrich_entry,
            "SP": self.sp_entry,
        }
        data: dict[str, float] = {}
        for name, widget in fields.items():
            text = widget.get().strip()
            try:
                data[name] = float(text)
            except ValueError:
                messagebox.showerror("Error", f"{name} must be a float.")
                return None
        return pd.DataFrame([data])

    def _on_output(self) -> None:
        """
        Handler for the 'Output' button: validate inputs and run a one-row prediction.
        """
        df_in = self.get_parameters_dataframe()
        if df_in is None:
            return  # error already shown
        self.df_in = df_in
        self.run_prediction()

    # ────────────────────────────────────────────────────────────────────────
    # Verification & Prediction logic
    # ────────────────────────────────────────────────────────────────────────
    def _validate_required_columns(self, df: pd.DataFrame, required_cols: List[str]) -> pd.DataFrame:
        """
        Ensure `df` contains every column in `required_cols`. Raise a clear error if not.
        Returns a *column-filtered* copy with only the required columns, preserving order.
        """
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"The following required columns are missing from DataFrame: {missing}")
        return df[required_cols].copy()


    def _build_verification_results(
        self,
        df_dhst_req: pd.DataFrame,
        interpolate_fn,
    ) -> pd.DataFrame:
        """
        Run the provided `interpolate_fn(enrich, burnup, sp, cool)` for each row in `df_dhst_req`
        and return a DataFrame of prediction series.
        """
        # Use itertuples for readability + speed; column names accessed as attributes.
        ver_series_list: list[pd.Series] = []
        for row in df_dhst_req.itertuples(index=False):
            ver_series_list.append(
                interpolate_fn(
                    row.Enrich,   # type: ignore[attr-defined]
                    row.Burnup,   # type: ignore[attr-defined]
                    row.SP,       # type: ignore[attr-defined]
                    row.Cool,     # type: ignore[attr-defined]
                )
            )
        return pd.DataFrame(ver_series_list)


    def _make_scatter_plots(
        self,
        df_for_plots: pd.DataFrame,
        y_vars_dataset: List[str],
        y_vars_prediction: List[str],
        out_dir: Path,
    ) -> None:
        """
        Produce the dataset and prediction scatter plots (4×4).
        """
        # [SNFs dataset]
        plot_title_dataset = "[SNFs dataset] Targets vs. Fuel Parameters — Colored by Type"
        output_fig_dataset = out_dir / "SNFs_dataset.png"
        plot_4x4_scatterplot(output_fig_dataset, df_for_plots, y_vars_dataset, plot_title_dataset)

        # [Prediction]
        plot_title_pred = "[Prediction] Targets vs. Fuel Parameters — Colored by Type"
        output_fig_pred = out_dir / "SNFs_prediction.png"
        plot_4x4_scatterplot(output_fig_pred, df_for_plots, y_vars_prediction, plot_title_pred)


    def _make_error_boxplots(
        self,
        df_for_errors: pd.DataFrame,
        error_metrics_colname: List[Tuple[str, str, str]],
        out_dir: Path,
    ) -> None:
        """
        Produce relative error boxplots comparing dataset vs. prediction.
        """
        title_boxplot = "[Prediction / Dataset] Error of Decay Heat & Source Terms"
        save_path = out_dir / "SNFs_comparsion.png"
        plot_dhst_RelativeError_boxplots(df_for_errors, error_metrics_colname, title_boxplot, save_path)


    def _verify_test_case(self) -> None:
        """
        Run a test case with known inputs and outputs to verify the prediction model.
        """
        # --- Paths & constants 
        output_dir = create_output_dir("Prediction/verification")

        X_VARS = ["Burnup", "Cool", "Enrich", "SP"]
        Y_VARS = ["DH_0y", "FN_0y", "FG_0y", "HG_0y"]
        PLOT_VARS = ["Type"]  # for coloring
        REQUIRED_COLS = X_VARS + Y_VARS + PLOT_VARS

        # Validate inputs and subset columns 
        df_dhst = self._validate_required_columns(self.database_dhst.copy(), REQUIRED_COLS)

        # Dataset plot (ground truth vs parameters) 
        #    (This uses only the validated subset.)
        plot_title_dataset = "[SNFs dataset] Targets vs. Fuel Parameters — Colored by Type"
        output_fig = output_dir / "SNFs_dataset.png"
        plot_4x4_scatterplot(output_fig, df_dhst, Y_VARS, plot_title_dataset)

        # Run verification predictions row-by-row 
        df_verify_result = self._build_verification_results(df_dhst, self.PredAssy.interpolate)

        # Concatenate side-by-side keeping column names (axis=1).  The original code
        df_merged = pd.concat([df_dhst.reset_index(drop=True), df_verify_result.reset_index(drop=True)], axis=1)

        # Prediction scatter plots 
        self._make_scatter_plots(
            df_for_plots=df_merged,
            y_vars_dataset=Y_VARS,
            y_vars_prediction=["DH_prediction", "FN_prediction", "FG_prediction", "HG_prediction"],
            out_dir=output_dir,
        )

        # Relative error boxplots 
        self.error_metrics_ColName = [
            ("DH", "DH_0y", "DH_prediction"),
            ("FN", "FN_0y", "FN_prediction"),
            ("FG", "FG_0y", "FG_prediction"),
            ("HG", "HG_0y", "HG_prediction"),
        ]
        self._make_error_boxplots(df_merged, self.error_metrics_ColName, output_dir)

        # Cleanup dialog / flags 
        self._running = False
        if self._dlg is not None and self._dlg.winfo_exists():
            self._dlg.destroy()


    def verification(self) -> None:
        """
        Public entrypoint to run the verification workflow in a background thread.
        Spawns a daemon thread and shows a simple running dialog (same behavior as before).
        """
        self._running = True
        self._show_running_dialog()
        self.n_snfs = len(self.database_dhst)

        # Keep threading behavior identical to original.
        threading.Thread(target=self._verify_test_case, args=(), daemon=True).start()

    # ────────────────────────────────────────────────────────────────────────
    # Actions for batch prediction
    # ────────────────────────────────────────────────────────────────────────
    def load_list(self) -> None:
        """Load a batch of SNF specs from an Excel/CSV file and start prediction."""
        # Capture the raw path first to correctly detect cancellation
        path_str = filedialog.askopenfilename(
            filetypes=[("Excel or CSV", "*.xlsx *.csv")]
        )
        if not path_str:
            messagebox.showerror("Error", "No valid Path.")
            return
        self.df_path = Path(path_str)
        self.df_in = load_dataset(self.df_path)
        if self.df_in.empty:
            messagebox.showerror("Error", "Dataset empty.")
            return

        self.n_snfs = len(self.df_in)
        self._running = True
        self._show_running_dialog()
        # Background computation
        threading.Thread(target=self.run_prediction, args=(), daemon=True).start()

    def run_prediction(self) -> None:
        """
        Perform per-row interpolation, update the UI, and save results if requested.
        For batch mode, shows the first 100 rows in the preview table.
        """
        if self.df_in is None or self.df_in.empty:
            messagebox.showerror("Error", "Dataset empty.")
            return
        series_list: list[pd.Series] = []

        # Copy and select required columns (order matters for downstream code)
        self.df_in_copy = self.df_in.copy()
        desired_cols = ["Enrich", "SP", "Burnup", "Cool"]
        self.df_in_copy = self.df_in_copy[desired_cols]

        # Predict each row
        for _, row in self.df_in_copy.iterrows():
            series_list.append(
                self.PredAssy.interpolate(
                    row["Enrich"],
                    row["Burnup"],
                    row["SP"],
                    row["Cool"],
                )
            )

        # Build predictions DataFrame and the 100-row preview (formatted)
        df_preds = pd.DataFrame(series_list)
        df_display = pd.DataFrame([ser.map(self._format) for ser in series_list[:100]])
        df_display.insert(0, "S/n", range(1, len(df_display) + 1))

        # Update preview table
        self._clear_viewers()
        self._insert_rows(self.DHST_viewer, df_display)

        # Stop progress dialog (if shown)
        self._running = False
        if self._dlg is not None and self._dlg.winfo_exists():
            self._dlg.destroy()

        # Save outputs if requested
        if self.save_var.get():
            df_out = pd.concat(
                [self.df_in.reset_index(drop=True), df_preds.reset_index(drop=True)],
                axis=1,
            )
            save_PredData(df_out)
