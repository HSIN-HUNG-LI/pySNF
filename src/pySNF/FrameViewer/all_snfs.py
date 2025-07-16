import threading
import time
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from collections import defaultdict
from typing import Literal
from operator import itemgetter

from base import SNFProcessor
from FrameViewer.BaseFrame import DataFrameViewer
from io_file import load_dataset, create_output_dir, write_excel


class AllSNFsFrame(tk.Frame):
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
        tk.Label(controls, text="Year (2022-2522):").pack(side=tk.LEFT)
        self.year_entry = tk.Entry(controls, width=10)
        self.year_entry.insert(0, "2025")
        self.year_entry.pack(side=tk.LEFT, padx=5)

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

    def _validate_year(self):
        """Check year input is an integer in the allowed range."""
        try:
            year = int(self.year_entry.get())
            if 2022 <= year <= 2522:
                return year
        except ValueError:
            pass
        messagebox.showerror("Error", "Enter valid year 2022–2522.")
        return None

    def _on_output(self):
        """Start processing if inputs are valid."""
        if self.df.empty:
            messagebox.showerror("Error", "Dataset empty.")
            return
        year = self._validate_year()
        if year is None:
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
        threading.Thread(target=self._run_search_all, args=(year,), daemon=True).start()

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

        self.df_path = filedialog.askopenfilename(
            filetypes=[("Text & CSV", "*.txt *.csv")]
        )
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

    def _run_search_all(self, year):
        """Compute aggregates for STDH, grams, and Ci across all SNFs."""
        stdh_totals = dict.fromkeys(self.cols_all, 0.0)
        gram_totals = defaultdict(float)
        ci_totals = defaultdict(float)

        for name in self.df["SNF_id"]:
            if not self._running:
                break
            proc = SNFProcessor(series_name=name, target_year=year)
            df_stdh = proc.compute_stdh()
            df_stdh.columns = self.cols_all
            # Sum STDH columns
            for key in self.cols_all:
                stdh_totals[key] += (
                    pd.to_numeric(df_stdh.get(key, pd.Series()), errors="coerce")
                    .fillna(0)
                    .sum()
                )

            # Aggregate top nuclides for mass and activity
            for func, col, totals in (
                (proc.compute_concentration, "gram/assy.", gram_totals),
                (proc.compute_activity, "Ci/assy.", ci_totals),
            ):
                for nuc, val in func().head(20)[["nuclide", col]].values:
                    totals[nuc] += float(pd.to_numeric(val, errors="coerce") or 0)

        # If saving is enabled, write out to Excel
        if self.save_var.get():  #  only save when the checkbox/variable is True
            #  create a timestamped directory under "Results_All_SNFs"
            output_dir = create_output_dir(parent_folder_name="Results_All_SNFs")
            #  build DataFrames for each sheet
            df_stdh_tot = pd.DataFrame([stdh_totals], columns=self.cols_all)
            df_gram_tot = pd.DataFrame(
                sorted(gram_totals.items(), key=itemgetter(1), reverse=True),
                columns=["nuclide", "gram/all"],
            )
            df_ci_tot = pd.DataFrame(
                sorted(ci_totals.items(), key=itemgetter(1), reverse=True),
                columns=["nuclide", "Ci/all"],
            )
            #  choose a meaningful file name
            file_name = f"All_SNFs_{year}"
            #  use the same processor instance (or any) to call the static method
            write_excel(df_stdh_tot, df_ci_tot, df_gram_tot, output_dir, file_name)

        # Display results on the main thread
        self.after(
            0, lambda: self._display_results(stdh_totals, gram_totals, ci_totals)
        )

    def _display_results(self, stdh_totals, gram_totals, ci_totals):
        """Load computed totals into DataFrame viewers."""
        if hasattr(self, "_dlg"):
            self._dlg.destroy()
        self.STDH_viewer.load_dataframe(pd.DataFrame([stdh_totals]))
        self.Gram_viewer.load_dataframe(
            pd.DataFrame(
                sorted(gram_totals.items(), key=itemgetter(1), reverse=True),
                columns=["nuclide", "gram/all"],
            )
        )
        self.Ci_viewer.load_dataframe(
            pd.DataFrame(
                sorted(ci_totals.items(), key=itemgetter(1), reverse=True),
                columns=["nuclide", "Ci/all"],
            )
        )
        self._running = False
