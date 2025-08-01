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
from io_file import load_dataset, create_output_dir, write_excel, store_data


class PredictionFrame(tk.Frame):
    """
    Frame to load a dataset of SNFs, compute aggregate STDH, mass, and activity,
    and display results in scrollable panels.
    """

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.save_var = tk.BooleanVar(value=True)
        self._running = False
        self.cols_all = [
            "DH(Watts/all)",
            "FN(n/s/all)",
            "FG(r/s/all)",
            "HG(r/s/all)",
        ]

        self._setup_scrollable_canvas()
        self._build_ui()
        self.multi_text.insert(
            "1.0",
            "Read all stdh dataset from [test_files] folder to compute sum of source term and decay heat with nuclide weight and activity",
        )

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
        controls = tk.Frame(self.inner)
        controls.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(controls, text="SNFs dataset:").pack(side=tk.LEFT)
        entry = tk.Entry(controls, width=40)
        entry.insert(0, f"Use right button to load Excel file")
        entry.config(state="disabled")
        entry.pack(side=tk.LEFT, padx=5)
        tk.Button(controls, text="Load (file)", command=self.load_list).pack(
            side=tk.LEFT
        )

        tk.Button(controls, text="Output", command=self._on_output).pack(side=tk.LEFT)
        tk.Checkbutton(controls, text="Save output", variable=self.save_var).pack(
            side=tk.LEFT
        )

        # Text log
        self.multi_text = tk.Text(self.inner, height=4, wrap=tk.WORD)
        self.multi_text.pack(fill=tk.X, padx=10)

        # Data viewers
        self.STDH_viewer = self._make_viewer(
            self.inner,
            150,
            self.cols_all,
            "Source Term & Decay Heat",
        )

        container = tk.Frame(self.inner)
        container.pack(fill=tk.X, padx=10, pady=5)

        self.Gram_viewer = self._make_viewer(
            container,
            200,
            ["nuclide", "gram/all"],
            "Weight (gram)",
            side=tk.LEFT,
            expand=True,
        )
        self.Ci_viewer = self._make_viewer(
            container,
            200,
            ["nuclide", "Ci/all"],
            "Activity (Ci)",
            side=tk.LEFT,
            expand=True,
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

    def _log_initial_message(self):
        """Display initial dataset summary in the log with highlighted SNF count."""
        self.multi_text.delete("1.0", tk.END)
        self.multi_text.tag_configure(
            "highlight", font=("TkDefaultFont", 10, "bold", "underline")
        )
        self.multi_text.insert(
            "1.0", f"From ~/{self.df_path} \nRead excel file and Load "
        )
        self.multi_text.insert(tk.END, str(self.n_snfs), "highlight")
        self.multi_text.insert(
            tk.END,
            (
                " SNFs.\n"
                "If compute ~6800 fuels, wait >720s,\n"
                "processing 14,000 files (~5.5M rows).\n"
            ),
        )

    def _on_output(self):
        """Start processing if inputs are valid."""
        if self.df.empty:
            messagebox.showerror("Error", "Dataset empty.")
            return

        # Reset log and show initial message in bold
        self.multi_text.delete("1.0", tk.END)
        self.multi_text.tag_configure(
            "highlight", font=("TkDefaultFont", 15, "bold", "underline")
        )
        self._log_initial_message()

        # Show progress dialog
        self._running = True
        self._show_running_dialog()

        # Background computation
        threading.Thread(target=self._run_search_all, args=(), daemon=True).start()

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

    def load_list(self):
        """Load SNF names from a text/CSV file into the selection list."""

        self.df_path = Path(filedialog.askopenfilename(
            filetypes=[("Excel or CSV", "*.txt *.csv")]
        ))
        if not self.df_path:
            messagebox.showerror("Error", "No valid Path.")
            return
        try:
            self.df = load_dataset(
                self.df_path
            )  # Change for input from button "load file"
            self.n_snfs = len(self.df)
            self._log_initial_message()
        except Exception as e:
            messagebox.showerror("Error", str(e))


    def run_prediction(
        self,
        data_dir: Path,
        grid_data: pd.DataFrame,
    ) -> None:
        """
        Load input data, perform 4D interpolation for each row, and save results.
        """
        # Define parameter spaces
        enrichment_space = np.arange(1.5, 6.1, 0.5)
        specific_power_space = np.arange(5, 46, 5)
        burnup_space = np.arange(5000, 74100, 3000)
        cooling_time_space = np.logspace(-5.75, 6.215, 150, base=math.e)

        output_cols = [f"{param}_prediction" for param in ['DH', 'FN', 'HG', 'FG']]

        # Read original data
        df_in = load_dataset(data_dir /"all_stdh_dataset.csv")
        df_in = df_in.head(200)
        # Collect predictions
        predictions: list[pd.Series] = []
        for i, (_, row) in enumerate(df_in.iterrows()):
            if i % 100 == 0:
                print(f"Calculated {i} cases…")

            assembler = PredictSNFs_interpolate(
                grid_data,
                row['Enrich'],
                row['Burnup'],
                row['SP'],
                row['Cool'],
                enrichment_space,
                specific_power_space,
                burnup_space,
                cooling_time_space,
                output_cols
            )
            predictions.append(assembler.interpolate())

        df_preds = pd.DataFrame(predictions)
        df_out = pd.concat([df_in.reset_index(drop=True), df_preds.reset_index(drop=True)], axis=1)

        # Save results
        output_filename = f"{pd.Timestamp.now().strftime('%Y%m%d')}_output.csv"
        store_data(df_out, output_filename, data_dir / 'output')