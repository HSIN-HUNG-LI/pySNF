import threading
import time
import math
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import numpy as np
from typing import Literal
from pathlib import Path

from base import PredictSNFs_interpolate
from FrameViewer.BaseFrame import DataFrameViewer
from visualize import plot_4x4_scatterplot, plot_stdh_RelativeError_boxplots
from io_file import (
    load_dataset,
    save_PredData,
    get_grid_ParqFile_path,
    get_stdh_path,
    create_output_dir,
)


class PredictionFrame(tk.Frame):
    """
    Frame to load a dataset of SNFs, compute aggregate STDH, mass, and activity,
    and display results in scrollable panels.
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.grid_data = pd.read_parquet(get_grid_ParqFile_path())
        self.save_var = tk.BooleanVar(value=True)
        self._running = False
        self.input_required = ["Enrich", "SP", "Burnup", "Cool"]
        self.cols_all = [
            "S/n",
            "DH(Watts)",
            "FN(n/s)",
            "FG(r/s)",
            "HG(r/s/kgSS304)",
        ]
        self.snf_stats = {
            "Enrich": 3.17,
            "SP": 26.21,
            "Burnup": 32806.18,
            "Cool": 24.36,
        }
        self._setup_scrollable_canvas()
        self._build_ui()
        self._log_initial_message()

    def _setup_scrollable_canvas(self):
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

    def _build_ui(self):
        """Set up controls, log area, and DataFrame viewers."""
        # Controls: dataset label, path, year entry, output button, save checkbox

        row = tk.Frame(self.inner)
        row.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(row, text="SNF Spec:").pack(side=tk.LEFT)

        row1 = tk.Frame(self.inner)
        row1.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(row1, text="Burnup (MWd/MTU):").pack(side=tk.LEFT)
        self.burnup_entry = tk.Entry(row1, width=10)
        self.burnup_entry.insert(0, f"{self.snf_stats['Burnup']}")
        self.burnup_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(row1, text="Cooling time (Year):").pack(side=tk.LEFT)
        self.year_entry = tk.Entry(row1, width=10)
        self.year_entry.insert(0, f"{self.snf_stats['Cool']}")
        self.year_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(row1, text="Enrichment (%U235):").pack(side=tk.LEFT)
        self.enrich_entry = tk.Entry(row1, width=10)
        self.enrich_entry.insert(0, f"{self.snf_stats['Enrich']}")
        self.enrich_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(row1, text="Specific Power (MW):").pack(side=tk.LEFT)
        self.sp_entry = tk.Entry(row1, width=10)
        self.sp_entry.insert(0, f"{self.snf_stats['SP']}")
        self.sp_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(row1, text="Output", command=self._on_output).pack(side=tk.LEFT)
        tk.Checkbutton(row1, text="Save output", variable=self.save_var).pack(
            side=tk.LEFT
        )

        # Text log
        self.multi_text = tk.Text(self.inner, height=6, wrap=tk.WORD)
        self.multi_text.pack(fill=tk.X, padx=10)

        # Data viewers
        self.STDH_viewer = self._make_viewer(
            self.inner,
            200,
            self.cols_all,
            "Predictions in Source Term & Decay Heat",
        )
        row2 = tk.Frame(self.inner)
        row2.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(row2, text="SNFs Specs by a batch file:").pack(side=tk.LEFT)
        entry = tk.Entry(row2, width=40)
        entry.insert(0, f"Use right button to load file and Output")
        entry.config(state="disabled")
        entry.pack(side=tk.LEFT, padx=5)
        tk.Button(row2, text="Load and Output", command=self.load_list).pack(
            side=tk.LEFT
        )
        row3 = tk.Frame(self.inner)
        row3.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(row3, text="Verification:").pack(side=tk.LEFT)
        entry = tk.Entry(row3, width=40)
        entry.insert(0, f"Use right button to load file and Output")
        entry.config(state="disabled")
        entry.pack(side=tk.LEFT, padx=5)
        tk.Button(row3, text="Output", command=self._verification).pack(
            side=tk.LEFT
        )
        # Text log
        self.multi_text2 = tk.Text(self.inner, height=5, wrap=tk.WORD)
        self.multi_text2.pack(fill=tk.X, padx=10)
        self.multi_text2.delete("1.0", tk.END)
        self.multi_text2.insert(
            "1.0",
            (
                "Verify model predictions for Source Term and Decay Heat (ST&DH).\n"
                "This routine exports three diagnostic PNG figures (saved to the output directory):\n"
                "  1) Reference dataset distribution — 'SNFs_dataset.png'\n"
                "  2) Prediction distribution — 'SNFs_prediction.png'\n"
                "  3) Relative error distribution (prediction vs. reference) — 'SNFs_comparison.png'\n"
                "\n"
                "Use these plots to quickly assess data coverage, model behavior, and error characteristics.\n"
            )
        )
    def _make_viewer(
        self,
        parent,
        height: int,
        columns: list[str],
        title: str,
        side: Literal["left", "right", "top", "bottom"] = tk.TOP,
        expand: bool = False,
    ) -> DataFrameViewer:
        """Instantiates a DataFrameViewer with fixed height."""
        frame = tk.Frame(parent, height=height)
        frame.pack(fill=tk.X, side=side, expand=expand, padx=5, pady=5)
        frame.pack_propagate(False)
        df_empty = pd.DataFrame(columns=columns)
        viewer = DataFrameViewer(frame, df_empty, title=title)
        viewer.pack(fill=tk.BOTH, expand=True)
        return viewer

    def _clear_viewers(self):
        # Clear all rows in viewer's tree
        for iid in self.STDH_viewer.tree.get_children():
            self.STDH_viewer.tree.delete(iid)

    def _insert_rows(self, viewer: DataFrameViewer, df: pd.DataFrame):
        # Insert DataFrame rows into viewer
        for vals in df.itertuples(index=False, name=None):
            viewer.tree.insert("", "end", values=vals)

    def _log_initial_message(self):
        self.multi_text.delete("1.0", tk.END)
        self.multi_text.insert(
            "1.0",
            (
                "Source Term & Decay Heat (ST&DH) — Prediction Modes\n"
                "Choose one of the following:\n"
                "  1) Load dataset: provide a table with columns [Enrich, SP, Burnup, Cool],\n"
                "     or select a file from the 'TEST_prediction' folder to compute ST&DH.\n"
                "  2) Manual input: enter the parameters above and click 'Output'.\n"
                "\n"
                "Notes:\n"
                "  • Only the first 100 rows are shown in the preview.\n"
                "  • Results are saved to the output directory.\n"
            )
        )

    def _log_file_message(self):
        """Display initial dataset summary in the log with highlighted SNF count."""
        self.multi_text.delete("1.0", tk.END)
        self.multi_text.tag_configure(
            "highlight", font=("TkDefaultFont", 10, "bold", "underline")
        )
        self.multi_text.insert("1.0", f"From ~/{self.df_path} \nRead file and Load ")
        self.multi_text.insert(tk.END, str(self.n_snfs), "highlight")
        self.multi_text.insert(
            tk.END,
            (
                " SNFs.\n"
                "If compute ~6800 fuels, wait >360s,\n"
                "processing 14,000 files (~5.5M rows).\n"
            ),
        )

    def _show_running_dialog(self):
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

    def _update_timer(self):
        """Update elapsed time every second until stopped."""
        if not getattr(self, "_running", False):
            self._dlg.destroy()
            return
        elapsed = int(time.time() - self._start_time)
        self.elapsed_label.config(
            text=f"Running {self.n_snfs} SNFs... \nElapsed: {elapsed}s\nIf compute ~6800 fuels, wait >360s,\n"
        )
        self.after(1000, self._update_timer)

    @staticmethod
    def _format(val):
        try:
            return f"{val:.2e}"
        except (ValueError, TypeError):
            return val

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

        # Build a one-row DataFrame
        return pd.DataFrame([data])

    def _on_output(self):
        """
        Handler for the "Output" button.
        Clears the log, validates inputs, then kicks off prediction.
        """
        # Reset log
        self.multi_text.delete("1.0", tk.END)
        self.multi_text.insert("1.0", "Compute by input parameters… ")

        # Validate & fetch inputs
        df_in = self.get_parameters_dataframe()
        if df_in is None:
            return  # error dialog already shown

        self.df_in = df_in
        self.run_prediction()
    def _verification(self):
        """
        Placeholder for verification logic.
        Currently does nothing but can be extended to handle verification tasks.
        """
        output_pred = create_output_dir("Prediction/verification")
        df_stdh = pd.read_csv(get_stdh_path())

        # Dataset plots
        plot_title_dataset = "[SNFs dataset] Targets vs. Fuel Parameters — Colored by Type"
        output_fig = output_pred / "SNFs_dataset.png"
        y_vars = ["DH_0y", "FN_0y", "HG_0y", "FG_0y"]
        plot_4x4_scatterplot(output_fig, df_stdh, y_vars, plot_title_dataset)

        # Prediction plots
        plot_title_pred = "[Prediction] Targets vs. Fuel Parameters — Colored by Type"
        output_fig_pred = output_pred / "SNFs_prediction.png"
        y_vars_pred = ["DH_prediction", "FN_prediction", "HG_prediction", "FG_prediction"]
        plot_4x4_scatterplot(output_fig_pred, df_stdh, y_vars_pred, plot_title_pred)

        self.error_metrics_ColName = [
            ("FN", "FN_0y", "FN_prediction"),
            ("FG", "FG_0y", "FG_prediction"),
            ("HG", "HG_0y", "HG_prediction"),
            ("DH", "DH_0y", "DH_prediction"),
        ]
        title_boxplot = f"[Prediction / Dataset] Relative Error across Source term and Decay heat"
        RelativeError_save_path = output_pred / "SNFs_comparsion.png"
        plot_stdh_RelativeError_boxplots(df_stdh, self.error_metrics_ColName, title_boxplot, RelativeError_save_path)

    def load_list(self):
        """Load SNF names from a text/CSV file into the selection list."""

        self.df_path = Path(
            filedialog.askopenfilename(filetypes=[("Excel or CSV", "*.xlsx *.csv")])
        )
        if not self.df_path:
            messagebox.showerror("Error", "No valid Path.")
            return
        else:
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
        Perform per-row interpolation, update the UI,
        and save results if requested.
        """
        # Pre-bind for speed & clarity
        grid_data = self.grid_data
        cross_times = 1
        enrich_step = 0.5 * cross_times
        sp_step = 5 * cross_times
        burnup_step = 3000 * cross_times
        cool_step = cross_times

        enrich_space = np.arange(1.5, 6.1, enrich_step)
        sp_space = np.arange(5, 46, sp_step)
        burnup_space = np.arange(5000, 74100, burnup_step)
        cool_space = np.logspace(-5.75, 6.215, 150, base=math.e)
        cool_space = cool_space[1::cool_step]
        out_cols = [f"{p}_prediction" for p in ("DH", "FN", "HG", "FG")]
        series_list: list[pd.Series] = []

        # Copy dataframe
        self.df_in_copy = self.df_in.copy()
        desired_cols = ["Enrich", "SP", "Burnup", "Cool"]
        self.df_in_copy = self.df_in_copy.loc[:, desired_cols].copy()

        PredAssy = PredictSNFs_interpolate(
            grid_data,
            enrich_space,
            sp_space,
            burnup_space,
            cool_space,
            out_cols,
        )
        for i, (_, row) in enumerate(self.df_in_copy.iterrows()):
            series_list.append(
                PredAssy.interpolate(
                    row["Enrich"],
                    row["Burnup"],
                    row["SP"],
                    row["Cool"],
                )
            )

        # Build DataFrame and format every numeric cell in scientific notation
        df_preds = pd.DataFrame(series_list)
        df_display = pd.DataFrame([ser.map(self._format) for ser in series_list[:100]])
        df_display.insert(0, "S/n", range(1, len(df_display) + 1))
        self._clear_viewers()
        self._insert_rows(self.STDH_viewer, df_display)

        # Stop timer dialog
        self._running = False
        # Save if checkbox is checked
        if self.save_var.get():
            df_out = pd.concat(
                [self.df_in.reset_index(drop=True), df_preds.reset_index(drop=True)],
                axis=1,
            )
            save_PredData(df_out)

    @staticmethod
    def run_experiment_GridResolution(
        enrich_factor: float = 1.0,
        sp_factor: float = 1.0,
        bp_factor: float = 1.0,
        cool_factor: float = 1.0,
        Exp_FolderName: str = "1111",
    ):
        """
        Start the prediction process in varying grid resolutions.
        This method is called by Notebook to run the experiment.
        """
        grid_data = pd.read_parquet(get_grid_ParqFile_path())
        # ============ Start Exp ============
        enrich_step = 0.5 * enrich_factor
        sp_step = 5 * sp_factor
        burnup_step = 3000 * bp_factor
        cool_step = cool_factor
        # ============ End Exp ============

        enrich_space = np.arange(1.5, 6.1, enrich_step)
        sp_space = np.arange(5, 46, sp_step)
        burnup_space = np.arange(5000, 74100, burnup_step)
        # cool_space = np.logspace(-5.75, 6.215, cool_step, base=math.e)
        cool_space = np.logspace(-5.75, 6.215, 150, base=math.e)
        cool_space = cool_space[1::cool_step]
        out_cols = [f"{p}_prediction" for p in ("DH", "FN", "HG", "FG")]
        series_list: list[pd.Series] = []

        # Load dataframe
        df_in = load_dataset(get_stdh_path())

        df_in_copy = df_in.copy().head(50)
        desired_cols = ["Enrich", "SP", "Burnup", "Cool"]
        df_in_copy = df_in_copy.loc[:, desired_cols].copy()

        PredAssy = PredictSNFs_interpolate(
            grid_data,
            enrich_space,
            sp_space,
            burnup_space,
            cool_space,
            out_cols,
        )
        for i, (_, row) in enumerate(df_in_copy.iterrows()):
            series_list.append(
                PredAssy.interpolate(
                    row["Enrich"],
                    row["Burnup"],
                    row["SP"],
                    row["Cool"],
                )
            )

        df_preds = pd.DataFrame(series_list)
        df_out = pd.concat(
            [df_in.reset_index(drop=True), df_preds.reset_index(drop=True)],
            axis=1,
        )
        output_folder = create_output_dir(f"Exp_Prediction_{Exp_FolderName}")
        df_out.to_csv(output_folder / "Prediction.csv", index=False)
