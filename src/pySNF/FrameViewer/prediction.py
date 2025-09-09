import threading
import time
import math
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from base import PredictSNFs_interpolate
from FrameViewer.BaseFrame import DataFrameViewer
from visualize import plot_4x4_scatterplot, plot_stdh_RelativeError_boxplots
from io_file import (
    load_dataset,
    save_PredData,
    get_grid_ParqFile_path,
    get_stdh_path,
    create_output_dir,
    get_grid_space,
)


class PredictionFrame(tk.Frame):
    """
    Frame to
    (1) predict Source Term & Decay Heat (ST&DH) from fuel parameters,
    (2) preview the results,
    (3) optionally export data/figures, and
    (4) run verification plots against a reference dataset.
    """

    # Columns shown in the prediction preview table
    cols_all: list[str] = [
        "s/n",
        "DH (W/assy.)",
        "FN (n/s/assy.)",
        "HG (r/s/kgSS304/MTU)",
        "FG (r/s/assy.)",
    ]

    # Required input features/order
    input_required: list[str] = ["Enrich", "SP", "Burnup", "Cool"]

    def __init__(self, parent: tk.Misc, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        # Persistent state
        self.grid_data: pd.DataFrame = pd.read_parquet(get_grid_ParqFile_path())
        self.save_var = tk.BooleanVar(value=True)
        self._running: bool = False

        # Attributes initialized for safety (avoid AttributeError in edge paths)
        self.df_in: Optional[pd.DataFrame] = None
        self.df_in_copy: Optional[pd.DataFrame] = None
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
            text="A case study by comparing the pySNF predictions with the TRITON calculations for a large numbers of SNFs",
        ).pack(side=tk.LEFT)
        row0_1 = tk.Frame(self.inner)
        row0_1.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(row0_1, text="Run Test Case", command=self._verification).pack(
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
            text="Predictions of SNF's Decay Heat & Source Terms",
            font=("Helvetica", 12, "bold"),
        ).pack(side=tk.LEFT)

        # row1: four numeric inputs + Output + Save toggle
        row1_1 = tk.Frame(self.inner)
        row1_1.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(
            row1_1, text="(1) SNF spec: Bp & Ct are required, En & Sp are optional"
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

        tk.Label(row1_2, text="SP (MW):").pack(side=tk.LEFT)
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
            text="(2) SNFs Specs: Load a csv file, e.g., Prediction_tsc01_batch.csv",
        ).pack(side=tk.LEFT)
        tk.Button(row2, text="Load & Output", command=self.load_list).pack(side=tk.LEFT)

        # Prediction preview viewer
        self.STDH_viewer = self._make_viewer(
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
        for iid in self.STDH_viewer.tree.get_children():
            self.STDH_viewer.tree.delete(iid)

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
            "Enrich": self.enrich_entry,
            "SP": self.sp_entry,
            "Burnup": self.burnup_entry,
            "Cool": self.year_entry,
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

    def _verification(self) -> None:
        """
        Generate verification plots (dataset/prediction distributions and
        relative error boxplots) into Prediction/verification.
        """
        output_pred = create_output_dir("Prediction/verification")
        df_stdh = pd.read_csv(get_stdh_path())

        # Dataset plots
        plot_title_dataset = (
            "[SNFs dataset] Targets vs. Fuel Parameters — Colored by Type"
        )
        output_fig = output_pred / "SNFs_dataset.png"
        y_vars = ["DH_0y", "FN_0y", "HG_0y", "FG_0y"]
        plot_4x4_scatterplot(output_fig, df_stdh, y_vars, plot_title_dataset)

        # Prediction plots
        plot_title_pred = "[Prediction] Targets vs. Fuel Parameters — Colored by Type"
        output_fig_pred = output_pred / "SNFs_prediction.png"
        y_vars_pred = [
            "DH_prediction",
            "FN_prediction",
            "HG_prediction",
            "FG_prediction",
        ]
        plot_4x4_scatterplot(output_fig_pred, df_stdh, y_vars_pred, plot_title_pred)

        # Relative error boxplots
        self.error_metrics_ColName = [
            ("FN", "FN_0y", "FN_prediction"),
            ("FG", "FG_0y", "FG_prediction"),
            ("HG", "HG_0y", "HG_prediction"),
            ("DH", "DH_0y", "DH_prediction"),
        ]
        title_boxplot = "[Prediction / Dataset] Error of Decay Heat & Source Terms"
        RelativeError_save_path = output_pred / "SNFs_comparsion.png"
        plot_stdh_RelativeError_boxplots(
            df_stdh, self.error_metrics_ColName, title_boxplot, RelativeError_save_path
        )

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
        # Define interpolation spaces
        grid_data = self.grid_data
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

        out_cols = [f"{p}_prediction" for p in ("DH", "FN", "HG", "FG")]
        series_list: list[pd.Series] = []

        # Copy and select required columns (order matters for downstream code)
        self.df_in_copy = self.df_in.copy()
        desired_cols = ["Enrich", "SP", "Burnup", "Cool"]
        self.df_in_copy = self.df_in_copy.loc[:, desired_cols].copy()

        # Interpolator
        PredAssy = PredictSNFs_interpolate(
            grid_data,
            enrich_space,
            sp_space,
            burnup_space,
            cool_space,
            out_cols,
        )

        # Predict each row
        for _, row in self.df_in_copy.iterrows():
            series_list.append(
                PredAssy.interpolate(
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
        self._insert_rows(self.STDH_viewer, df_display)

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
