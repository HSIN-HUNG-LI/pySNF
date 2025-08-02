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
from io_file import (
    load_dataset,
    save_PredData,
    get_grid_ParqFile_path,
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
            "DH(Watts)",
            "FN(n/s)",
            "FG(r/s)",
            "HG(r/s)",
        ]

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
        row1 = tk.Frame(self.inner)
        row1.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(row1, text="SNFs dataset:").pack(side=tk.LEFT)
        entry = tk.Entry(row1, width=40)
        entry.insert(0, f"Use right button to load Excel file and Output")
        entry.config(state="disabled")
        entry.pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="Load and Output", command=self.load_list).pack(
            side=tk.LEFT
        )
        row2 = tk.Frame(self.inner)
        row2.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(row2, text="Cool Year:").pack(side=tk.LEFT)
        self.year_entry = tk.Entry(row2, width=10)
        self.year_entry.insert(0, "42.133")  # Default cool year
        self.year_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(row2, text="Burnup(MWD/MTU):").pack(side=tk.LEFT)
        self.burnup_entry = tk.Entry(row2, width=10)
        self.burnup_entry.insert(0, "11970.27")  # Default cool year
        self.burnup_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(row2, text="Average Power (MW)").pack(side=tk.LEFT)
        self.sp_entry = tk.Entry(row2, width=10)
        self.sp_entry.insert(0, "11.59")  # Default cool year
        self.sp_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(row2, text="Enrich(%U235)").pack(side=tk.LEFT)
        self.enrich_entry = tk.Entry(row2, width=10)
        self.enrich_entry.insert(0, "1.9")  # Default cool year
        self.enrich_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(row2, text="Output", command=self._on_output).pack(side=tk.LEFT)
        tk.Checkbutton(row2, text="Save output", variable=self.save_var).pack(
            side=tk.LEFT
        )

        # Text log
        self.multi_text = tk.Text(self.inner, height=4, wrap=tk.WORD)
        self.multi_text.pack(fill=tk.X, padx=10)

        # Data viewers
        self.STDH_viewer = self._make_viewer(
            self.inner,
            300,
            self.cols_all,
            "Predictions in Source Term & Decay Heat",
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
            "Two ways to predict source term and decay heat (ST&DH)\n First: Read dataset (Contain [Enrich, SP, Burnup, Cool] column ) or use file in [test_files] folder to compute ST&DH \n Second: Enter information above and click output \n (Only display first 100 row)",
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
                "If compute ~6800 fuels, wait >720s,\n"
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
            text=f"Running {self.n_snfs} SNFs... \nElapsed: {elapsed}s\nIf compute ~6800 fuels, wait >720s,\n"
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
            "Cool": self.year_entry,  # cooling time
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

    def load_list(self):
        """Load SNF names from a text/CSV file into the selection list."""

        self.df_path = Path(
            filedialog.askopenfilename(filetypes=[("Excel or CSV", "*.txt *.csv")])
        )
        if not self.df_path:
            messagebox.showerror("Error", "No valid Path.")
            return
        try:
            self.df_in = load_dataset(
                self.df_path
            )  # Change for input from button "load file"

            # ================================================================
            # self.df_in = self.df_in.iloc[969:975]
            # self.df_in.columns = ["Enrich", "SP", "Burnup", "Cool"]
            # ================================================================
            self.n_snfs = len(self.df_in)
            self._running = True
            # Show the elapsed-time dialog
            self._show_running_dialog()
            # Background computation
            threading.Thread(target=self.run_prediction, args=(), daemon=True).start()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_prediction(self) -> None:
        """
        Perform per-row interpolation, update the UI,
        and save results if requested.
        """
        # Pre-bind for speed & clarity
        grid_data = self.grid_data
        self._running = True

        # Define parameter grids once
        enrich_space = np.arange(1.5, 6.1, 0.5)
        sp_space = np.arange(5, 46, 5)
        burnup_space = np.arange(5000, 74100, 3000)
        cool_space = np.logspace(-5.75, 6.215, 150, base=math.e)
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
            if i % 1000 == 0:
                print(f"Calculated {i}/{self.n_snfs} cases…")
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
        # Update the treeview
        self._clear_viewers()
        self._insert_rows(
            self.STDH_viewer,
            pd.DataFrame([ser.map(self._format) for ser in series_list[:100]]),
        )
        # Stop timer dialog
        self._running = False
        # Save if checkbox is checked
        if self.save_var.get():
            df_out = pd.concat(
                [self.df_in.reset_index(drop=True), df_preds.reset_index(drop=True)],
                axis=1,
            )
            save_PredData(df_out)
