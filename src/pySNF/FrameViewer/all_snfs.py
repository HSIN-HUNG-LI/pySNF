import threading
import time
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from collections import defaultdict
from typing import Literal, Optional
from operator import itemgetter
from pathlib import Path

from base import SNFProcessor
from FrameViewer.BaseFrame import DataFrameViewer
from io_file import load_dataset, create_output_dir, write_excel


class AllSNFsFrame(tk.Frame):
    """
    Load a list of SNF IDs, compute aggregate Source Term & Decay Heat (STDH),
    nuclide mass (grams), and activity (Ci), and display the results
    in scrollable panels. Allows optional export to an Excel workbook.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────────────────
    def __init__(self, parent: tk.Misc, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        # Persistent UI/compute state
        self.save_var = tk.BooleanVar(value=True)
        self._running: bool = False
        self._dlg: Optional[tk.Toplevel] = None
        self._start_time: float = 0.0
        self.elapsed_label: Optional[tk.Label] = None

        # Input/loaded dataset state
        self.df: pd.DataFrame = pd.DataFrame()  # set initially empty
        self.df_path: Optional[Path] = None
        self.n_snfs: int = 0

        # Column names for the STDH totals table
        self.cols_all = [
            "DH(Watts/all)",
            "FN(n/s/all)",
            "FG(r/s/all)",
            "HG(r/s/all)",
        ]

        # Layout and initial message
        self._setup_scrollable_canvas()
        self._build_ui()
        self.multi_text.insert(
            "1.0",
            "Read all stdh dataset from [test_files] folder to compute "
            "sum of source term and decay heat with nuclide weight and activity",
        )

    # ──────────────────────────────────────────────────────────────────────────
    # UI building
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        """Set up controls, the log text area, and three DataFrame viewers."""
        # Controls (dataset path, year, run/save)
        controls = tk.Frame(self.inner)
        controls.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(controls, text="SNFs dataset:").pack(side=tk.LEFT)
        entry = tk.Entry(controls, width=40)
        entry.insert(0, "Use right button to load Excel file")
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
            parent=self.inner,
            height=150,
            columns=self.cols_all,
            title="Decay Heat & Source Terms",
        )

        # Weight/Activity side-by-side container
        container = tk.Frame(self.inner)
        container.pack(fill=tk.X, padx=10, pady=5)

        self.Gram_viewer = self._make_viewer(
            parent=container,
            height=200,
            columns=["nuclide", "gram/all"],
            title="Weight (gram)",
            side=tk.LEFT,
            expand=True,
        )
        self.Ci_viewer = self._make_viewer(
            parent=container,
            height=200,
            columns=["nuclide", "Ci/all"],
            title="Activity (Ci)",
            side=tk.LEFT,
            expand=True,
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
        """
        Instantiate a DataFrameViewer with a fixed-height container to avoid UI jumps.
        """
        frame = tk.Frame(parent, height=height)
        frame.pack(fill=tk.X, side=side, expand=expand, padx=5, pady=5)
        frame.pack_propagate(False)  # preserve the fixed height

        df_empty = pd.DataFrame(columns=columns)
        viewer = DataFrameViewer(frame, df_empty, title=title)
        viewer.pack(fill=tk.BOTH, expand=True)
        return viewer

    def _setup_scrollable_canvas(self) -> None:
        """Create a vertical-scrollable canvas; put all content inside `self.inner`."""
        self.canvas = tk.Canvas(self)
        scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner = tk.Frame(self.canvas)
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        # Keep scrollregion and inner width in sync
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self.window_id, width=e.width),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Logging & dialogs
    # ──────────────────────────────────────────────────────────────────────────
    def _log_initial_message(self) -> None:
        """Display initial dataset summary in the log with a highlighted SNF count."""
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

    def _show_running_dialog(self) -> None:
        """Open a small modal dialog showing elapsed time and a Stop button."""
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
        """Update elapsed time once per second until the computation stops."""
        if not getattr(self, "_running", False):
            # Close dialog if still open
            if self._dlg is not None and self._dlg.winfo_exists():
                self._dlg.destroy()
            return

        elapsed = int(time.time() - self._start_time)
        if self.elapsed_label is not None:
            self.elapsed_label.config(
                text=(
                    f"Running {self.n_snfs} SNFs...\n"
                    f"Elapsed: {elapsed}s\n"
                    f"If compute ~6800 fuels, wait >720s,\n"
                )
            )
        # Re-arm the timer
        self.after(1000, self._update_timer)

    # ──────────────────────────────────────────────────────────────────────────
    # Input validation & actions
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_year(self) -> Optional[int]:
        """Validate that the year is an integer in [2022, 2522]."""
        try:
            year = int(self.year_entry.get())
            if 2022 <= year <= 2522:
                return year
        except ValueError:
            pass
        messagebox.showerror("Error", "Enter valid year 2022–2522.")
        return None

    def _on_output(self) -> None:
        """Prepare state and dispatch the background computation."""
        if self.df.empty:
            messagebox.showerror("Error", "Dataset empty.")
            return

        year = self._validate_year()
        if year is None:
            return

        # Reset log and print the initial summary
        self.multi_text.delete("1.0", tk.END)
        self.multi_text.tag_configure(
            "highlight", font=("TkDefaultFont", 15, "bold", "underline")
        )
        self._log_initial_message()

        # Show progress dialog and launch the worker thread
        self._running = True
        self._show_running_dialog()
        threading.Thread(target=self._run_search_all, args=(year,), daemon=True).start()

    def load_list(self) -> None:
        """Let user select a text/CSV file, load it with `load_dataset`, and summarize."""
        # Ask for path; bail early if user cancels
        path_str = filedialog.askopenfilename(filetypes=[("Text & CSV", "*.txt *.csv")])
        if not path_str:
            messagebox.showerror("Error", "No valid Path.")
            return

        self.df_path = Path(path_str)
        try:
            # Use provided helper to read the dataset
            self.df = load_dataset(self.df_path)
            self.n_snfs = len(self.df)
            self._log_initial_message()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # Core computation (runs on a background thread)
    # ──────────────────────────────────────────────────────────────────────────
    def _run_search_all(self, year: int) -> None:
        """
        Compute totals for STDH and the top-20 nuclides by mass (grams) and activity (Ci),
        aggregated across all SNFs in `self.df`.
        """
        # Accumulators
        stdh_totals: dict[str, float] = dict.fromkeys(self.cols_all, 0.0)
        gram_totals: defaultdict[str, float] = defaultdict(float)
        ci_totals: defaultdict[str, float] = defaultdict(float)

        # Iterate SNFs; allow user to stop via _running flag
        for name in self.df["SNF_id"]:
            if not self._running:
                break

            proc = SNFProcessor(series_name=name, target_year=year)

            # STDH totals
            df_stdh = proc.compute_stdh()
            df_stdh.columns = self.cols_all
            for key in self.cols_all:
                stdh_totals[key] += (
                    pd.to_numeric(df_stdh.get(key, pd.Series()), errors="coerce")
                    .fillna(0)
                    .sum()
                )

            # Top-20 nuclides: mass (grams) and activity (Ci), aggregated
            for func, col, totals in (
                (proc.compute_concentration, "gram/assy.", gram_totals),
                (proc.compute_activity, "Ci/assy.", ci_totals),
            ):
                for nuc, val in func().head(20)[["nuclide", col]].values:
                    # Robust numeric conversion; keep exact behavior (coerce→NaN→0)
                    totals[nuc] += float(pd.to_numeric(val, errors="coerce") or 0)

        # Optional Excel export
        if self.save_var.get():
            output_dir = create_output_dir(parent_folder_name="Results_All_SNFs")

            df_stdh_tot = pd.DataFrame([stdh_totals], columns=self.cols_all)
            df_gram_tot = pd.DataFrame(
                sorted(gram_totals.items(), key=itemgetter(1), reverse=True),
                columns=["nuclide", "gram/all"],
            )
            df_ci_tot = pd.DataFrame(
                sorted(ci_totals.items(), key=itemgetter(1), reverse=True),
                columns=["nuclide", "Ci/all"],
            )

            file_name = f"All_SNFs_{year}"
            write_excel(df_stdh_tot, df_ci_tot, df_gram_tot, output_dir, file_name)

        # Push results to UI on the main thread
        self.after(
            0, lambda: self._display_results(stdh_totals, gram_totals, ci_totals)
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Results rendering (main thread)
    # ──────────────────────────────────────────────────────────────────────────
    def _display_results(
        self,
        stdh_totals: dict[str, float],
        gram_totals: dict[str, float],
        ci_totals: dict[str, float],
    ) -> None:
        """Populate the three viewers and close any progress dialog."""
        # Close dialog if still present
        if self._dlg is not None and self._dlg.winfo_exists():
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
