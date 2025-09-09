import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Iterable

import pandas as pd

from base import SNFProcessor
from FrameViewer.BaseFrame import DataFrameViewer
from io_file import create_output_dir, load_dataset


class MultipleSearchFrame(tk.Frame):
    """
    Frame to manage multiple SNF searches:
      - Add/Delete names manually or via file (or Clear All)
      - Specify target year
      - Display results in a DataFrameViewer
      - Optionally save output as CSV
    """

    # Labels used to highlight summary rows in the tree
    _SUMMARY_LABELS = ("Sum", "Average", "Min", "Max")

    def __init__(self, parent: tk.Misc, df: pd.DataFrame, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        # Main dataset and UI state
        self.df: pd.DataFrame = df
        self.save_var = tk.BooleanVar(value=True)   # Toggle CSV export
        self.selected_names: list[str] = []         # SNF names for batch search

        # Column definitions shared across methods
        self.viewer_cols: list[str] = [
            "SNF_id",
            "DH (W/assy.)",
            "FN (n/s/assy.)",
            "FG (r/s/assy.)",
            "HG (r/s/kgSS304/MTU)",
        ]
        self.df_cols: list[str] = [
            "SNF_id",
            "DH(Watts/assy.)",
            "FN(n/s/assy.)",
            "FG(r/s/assy.)",
            "HG(r/s/kgSS304/MTU)",
        ]
        # ── Build Input Controls ───────────────────────────────────────────────
        # First row for name entry and action buttons
        row1 = tk.Frame(self)
        row1.pack(fill=tk.X, padx=10, pady=10)
        # SNF_id entry and action buttons
        tk.Label(row1, text="(1) SNFs id:").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(row1, width=20)
        self.name_entry.insert(0, "1C2505")  # Default SNF name
        self.name_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="Add", command=self.add_multiple).pack(side=tk.LEFT)
        tk.Button(row1, text="Delete", command=self.delete_multiple).pack(side=tk.LEFT, padx=5)
        tk.Button(row1, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)
        tk.Label(row1, text="OR").pack(side=tk.LEFT, padx=(10, 0))

        # Second row for file load 
        row2 = tk.Frame(self)
        row2.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(row2, text="(2) Load a csv file, e.g., TSC01_SNFs_Id.csv").pack(side=tk.LEFT, padx=(0, 0))
        tk.Button(row2, text="Load", command=self.load_list).pack(side=tk.LEFT, padx=(10, 0))

        # Third row for year entry and search button
        row3 = tk.Frame(self)
        row3.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(row3, text="Year (2022–2522):").pack(side=tk.LEFT, padx=(0, 0))
        self.year_entry = tk.Entry(row3, width=10)
        self.year_entry.insert(0, "2025")  # Default year
        self.year_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(row3, text="Output", command=self.search_multiple).pack(side=tk.LEFT, padx=(10, 0))

        # Toggle for saving output
        tk.Checkbutton(row3, text="Save output", variable=self.save_var).pack(side=tk.LEFT, padx=5)

        # Text log area
        self.multi_text = tk.Text(self, height=4, wrap=tk.WORD)
        self.multi_text.pack(fill=tk.X, padx=10, pady=(0, 5))

        # Results viewer setup
        result_frame = tk.Frame(self, height=200)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        result_frame.pack_propagate(False)

        empty_df = pd.DataFrame(columns=self.viewer_cols)
        self.STDH_df_viewer = DataFrameViewer(result_frame, empty_df, title="Decay Heat & Source Terms")
        self.STDH_df_viewer.pack(fill=tk.BOTH, expand=True)

        # Treeview visual tweaks
        self._configure_tree_styles()

        # Initialize log
        self._refresh_text()

    # ────────────────────────────────────────────────────────────────────────
    # Small UI helpers
    # ────────────────────────────────────────────────────────────────────────
    def _configure_tree_styles(self) -> None:
        """Configure alternating row styles and a style for summary rows."""
        style = ttk.Style()
        style.configure("Treeview", rowheight=24)  # Optionally adjust row height
        self.STDH_df_viewer.tree.tag_configure("evenrow", background="#ffffff")
        self.STDH_df_viewer.tree.tag_configure("oddrow", background="#f0f0f0")
        self.STDH_df_viewer.tree.tag_configure("summary", foreground="red")

    def _refresh_text(self) -> None:
        """Display current list of names in the text log."""
        self.multi_text.delete("1.0", tk.END)
        msg = (
            "SNFs names: " + ", ".join(self.selected_names)
            if self.selected_names
            else "No SNFs names added."
        )
        self.multi_text.insert(tk.END, msg)

    def _show_error(self, msg: str) -> None:
        """
        Show an error message in the log and a message box,
        then restore the name list display.
        """
        self.multi_text.delete("1.0", tk.END)
        self.multi_text.insert(tk.END, f"Error: {msg}\n")
        messagebox.showerror("Error", f"{msg}.\n")
        self._refresh_text()

    # ────────────────────────────────────────────────────────────────────────
    # Name list management
    # ────────────────────────────────────────────────────────────────────────
    def add_multiple(self) -> None:
        """Add unique names from the entry field to the selection list."""
        # Guard: dataset must have SNF_id column (prevents KeyError)
        if "SNF_id" not in self.df.columns:
            self._show_error("Dataset missing 'SNF_id' column.")
            return

        entries = [n.strip() for n in self.name_entry.get().split(",") if n.strip()]
        for name in entries:
            matches = self.df[self.df["SNF_id"].astype(str).str.contains(name, case=False, na=False)]
            if matches.empty:
                self._show_error(f" No matches for '{name}'.\n")
                return
            if name not in self.selected_names:
                self.selected_names.append(name)
            else:
                self._show_error(f" SNF '{name}' already added.\n")
        self._refresh_text()

    def delete_multiple(self) -> None:
        """Remove specified names from the selection list."""
        entries = [n.strip() for n in self.name_entry.get().split(",") if n.strip()]
        for name in entries:
            if name in self.selected_names:
                self.selected_names.remove(name)
            else:
                self._show_error("Please enter selected names in order to delete.")
        self._refresh_text()

    def clear_all(self) -> None:
        """Clear the entire list of selected SNF names and refresh the log display."""
        self.selected_names.clear()
        self._refresh_text()

    def load_list(self) -> None:
        """Load SNF names from a text/CSV file into the selection list."""
        # Guard: dataset must have SNF_id column (prevents KeyError)
        if "SNF_id" not in self.df.columns:
            self._show_error("Dataset missing 'SNF_id' column.")
            return

        path = Path(filedialog.askopenfilename(filetypes=[("CSV", "*.csv")]))
        if not path:
            self._show_error("No valid Path.")
            return
        try:
            self.df_content = load_dataset(path)
            names = self.df_content["SNF_id"].astype(str).tolist()
            if not names:
                self._show_error("No valid names found.")
                return

            # Determine which names are not present in the dataset (exact match)
            valid_names = set(self.df["SNF_id"].astype(str))
            invalid = [n for n in names if n not in valid_names]
            if invalid:
                invalid_list = ", ".join(invalid)
                self._show_error(f"The following names do not exist: {invalid_list}")
                return

            # Keep file order as provided (do not deduplicate to preserve behavior)
            self.selected_names = names
            self._refresh_text()
        except Exception as e:
            self._show_error(str(e))

    # ────────────────────────────────────────────────────────────────────────
    # Core: compute, render, export
    # ────────────────────────────────────────────────────────────────────────
    def search_multiple(self) -> None:
        """Run batch STDH computation and update the viewer (and CSV if enabled)."""
        # Validate year input (keep float to preserve original behavior)
        try:
            year = float(self.year_entry.get())
            assert 2022 <= year <= 2522
        except Exception:
            self._show_error("Invalid year. Enter an integer value between 2022 and 2522.")
            return

        # Validate dataset and selections
        if self.df.empty:
            self._show_error("Dataset is not loaded or is empty.")
            return
        if not self.selected_names:
            self._show_error("No SNF names selected.")
            return
        if "SNF_id" not in self.df.columns:
            self._show_error("Dataset missing 'SNF_id' column.")
            return

        # Accumulators
        rows: list[tuple] = []
        grand_totals: dict[str, float] = {col: 0.0 for col in self.df_cols if col != "SNF_id"}
        name_summaries: list[dict[str, float]] = []

        # Loop over each SNF series
        for name in self.selected_names:
            # Initialize processor for this series
            try:
                proc = SNFProcessor(series_name=name, target_year=year)
            except Exception as e:
                self._show_error(f"Error during SNF Processor:\n{e}")
                return

            # Compute STDH DataFrame for this series
            df_stdh = proc.compute_stdh()

            # Ensure numeric and accumulate per-series totals
            per_name_totals: dict[str, float] = {}
            for col in self.df_cols[1:]:
                df_stdh[col] = pd.to_numeric(df_stdh[col], errors="coerce")
                sum_val = df_stdh[col].sum()
                grand_totals[col] += sum_val
                per_name_totals[col] = sum_val

            # Store this series' totals for later statistics
            name_summaries.append(per_name_totals)

            # Format each row of df_stdh and prepend the series name
            disp = df_stdh.round(2).map(lambda x: f"{x:.2e}")
            disp.insert(0, "SNF_id", name)  # matches original behavior/casing
            rows.extend(disp.itertuples(index=False, name=None))

        # Append summary rows if any data exists
        if rows:
            rows.extend(self._build_summary_rows(grand_totals, name_summaries))

        # Refresh the tree view with all rows
        self._refresh_tree(rows)

        # Optionally save results to CSV
        if self.save_var.get() and rows:
            try:
                out_dir = create_output_dir(parent_folder_name="Results_Multiple_SNFs")
                pd.DataFrame(rows, columns=self.df_cols).to_csv(
                    out_dir / "Multiple_STDH_results.csv", index=False
                )
            except Exception as e:
                self.multi_text.insert(tk.END, f"Error saving CSV: {e}\n")

    # ────────────────────────────────────────────────────────────────────────
    # Internal helpers (no behavior change)
    # ────────────────────────────────────────────────────────────────────────
    def _build_summary_rows(
        self,
        grand_totals: dict[str, float],
        name_summaries: list[dict[str, float]],
    ) -> list[tuple]:
        """
        Build Sum, Average, Min, Max rows in the same format and precision
        as the original code.
        """
        rows: list[tuple] = []

        # Sum row
        sum_row = tuple(["Sum"] + [f"{grand_totals[col]:.2e}" for col in self.df_cols[1:]])
        rows.append(sum_row)

        # Average row
        n = max(1, len(name_summaries))  # guard against division by zero (not expected)
        avg_row = tuple(["Average"] + [f"{grand_totals[col] / n:.2e}" for col in self.df_cols[1:]])
        rows.append(avg_row)

        # Min row
        min_row = tuple(
            ["Min"] + [f"{min(ns[col] for ns in name_summaries):.2e}" for col in self.df_cols[1:]]
        )
        rows.append(min_row)

        # Max row
        max_row = tuple(
            ["Max"] + [f"{max(ns[col] for ns in name_summaries):.2e}" for col in self.df_cols[1:]]
        )
        rows.append(max_row)

        return rows

    def _refresh_tree(self, rows: Iterable[tuple]) -> None:
        """Clear and repopulate the tree with striped rows and summary highlighting."""
        tree = self.STDH_df_viewer.tree
        for item in tree.get_children():
            tree.delete(item)

        for idx, row in enumerate(rows):
            base_tag = "oddrow" if idx % 2 else "evenrow"
            if row and row[0] in self._SUMMARY_LABELS:
                tags = (base_tag, "summary")
            else:
                tags = (base_tag,)
            tree.insert("", "end", values=row, tags=tags)

        self._refresh_text()
