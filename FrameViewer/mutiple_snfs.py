import tkinter as tk
from tkinter import  filedialog, messagebox, ttk
import pandas as pd
import re
from base import SNFProcessor
from FrameViewer.BaseFrame import DataFrameViewer
from pathlib import Path
from io_file import create_output_dir


class MultipleSearchFrame(tk.Frame):
    """
    Frame to manage multiple SNF searches:
      - Add/Delete names manually or via file (or Clear All)
      - Specify target year
      - Display results in a DataFrameViewer
      - Optionally save output as CSV
    """

    def __init__(self, parent, df: pd.DataFrame, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.df = df  # Main dataset
        self.save_var = tk.BooleanVar(value=True)  # Toggle CSV export
        self.selected_names = []  # List of SNF names for batch search

        # --- Column definitions shared across methods ---
        self.cols = [
            "SNF_id",
            "DH(Watts/assy.)",
            "FN(n/s/assy.)",
            "FG(r/s/assy.)",
            "HG(r/s/kgSS304/MTU)",
        ]

        # --- Build Input Controls ---
        input_frame = tk.Frame(self)
        input_frame.pack(fill=tk.X, padx=10, pady=5)

        # --- SNF_id entry and action buttons ---
        tk.Label(input_frame, text="SNFs id:").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(input_frame, width=20)
        self.name_entry.insert(0, "1C2505")  # Default SNF name
        self.name_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(input_frame, text="Add", command=self.add_multiple).pack(side=tk.LEFT)
        tk.Button(input_frame, text="Delete", command=self.delete_multiple).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(input_frame, text="Clear All", command=self.clear_all).pack(
            side=tk.LEFT, padx=5
        )
        tk.Button(input_frame, text="Load (file)", command=self.load_list).pack(
            side=tk.LEFT
        )

        # --- Year entry and search button ---
        tk.Label(input_frame, text="Year (2022–2522):").pack(side=tk.LEFT, padx=(20, 0))
        self.year_entry = tk.Entry(input_frame, width=10)
        self.year_entry.insert(0, "2025")  # Default year
        self.year_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(input_frame, text="Output", command=self.search_multiple).pack(
            side=tk.LEFT
        )

        # --- Toggle for saving output ---
        tk.Checkbutton(input_frame, text="Save output", variable=self.save_var).pack(
            side=tk.LEFT, padx=5
        )

        # --- Text log area ---
        self.multi_text = tk.Text(self, height=4, wrap=tk.WORD)
        self.multi_text.pack(fill=tk.X, padx=10, pady=(0, 5))

        # --- Results viewer setup ---
        result_frame = tk.Frame(self, height=200)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        result_frame.pack_propagate(False)

        empty_df = pd.DataFrame(columns=self.cols)
        self.STDH_df_viewer = DataFrameViewer(
            result_frame, empty_df, title="Source term & Decay Heat"
        )
        self.STDH_df_viewer.pack(fill=tk.BOTH, expand=True)
        
        # Configure alternating row styles for readability
        style = ttk.Style()
        style.configure("Treeview", rowheight=24)  # optionally adjust row height
        self.STDH_df_viewer.tree.tag_configure("evenrow", background="#ffffff")
        self.STDH_df_viewer.tree.tag_configure("oddrow",  background="#f0f0f0")
        self.STDH_df_viewer.tree.tag_configure("summary", foreground="red")

        # Initialize log
        self._refresh_text()

    def _refresh_text(self):
        """Display current list of names in the text log."""
        self.multi_text.delete("1.0", tk.END)
        msg = (
            "SNFs names: " + ", ".join(self.selected_names)
            if self.selected_names
            else "No SNFs names added."
        )
        self.multi_text.insert(tk.END, msg)

    def _show_error(self, msg: str):
        """Show an error message in the log, then restore name list display."""
        self.multi_text.delete("1.0", tk.END)
        self.multi_text.insert(tk.END, f"Error: {msg}\n")
        messagebox.showerror("Error", f"{msg}.\n")
        self._refresh_text()

    def add_multiple(self):
        """Add unique names from the entry field to the selection list."""
        entries = [n.strip() for n in self.name_entry.get().split(",") if n.strip()]
        for name in entries:
            matches = self.df[
                self.df["SNF_id"].astype(str).str.contains(name, case=False, na=False)
            ]
            if matches.empty:
                self._show_error(f" No matches for '{name}'.\n")
                return
            if name not in self.selected_names:
                self.selected_names.append(name)
            else:
                self._show_error(f" SNF '{name}' already added.\n")
        self._refresh_text()

    def delete_multiple(self):
        """Remove specified names from the selection list."""
        entries = [n.strip() for n in self.name_entry.get().split(",") if n.strip()]
        for name in entries:
            if name in self.selected_names:
                self.selected_names.remove(name)
            else:
                self._show_error("Please enter selected names in order to delete.")
        self._refresh_text()

    def clear_all(self):
        """Clear the entire list of selected SNF names and refresh the log display."""
        self.selected_names.clear()
        self._refresh_text()

    def load_list(self):
        """Load SNF names from a text/CSV file into the selection list."""
        path = filedialog.askopenfilename(filetypes=[("Text & CSV", "*.txt *.csv")])
        if not path:
            self._show_error("No valid Path.")
            return
        try:
            content = Path(path).read_text(encoding="utf-8-sig")
            names = [n for n in re.split(r"[,\s]+", content) if n]
            if not names:
                self._show_error("No valid names found.")
                return
            # Determine which names are not present in the dataset
            valid_names = set(self.df["SNF_id"].astype(str))
            invalid = [n for n in names if n not in valid_names]
            if invalid:
                # Alert the user about each invalid name and abort loading
                invalid_list = ", ".join(invalid)
                self._show_error(f"The following names do not exist: {invalid_list}")
                return
            self.selected_names = names
            self._refresh_text()
        except Exception as e:
            self._show_error(str(e))

    def search_multiple(self):
        """Run batch STDH computation and update the viewer (and CSV if enabled)."""
        # Validate year input
        try:
            year = float(self.year_entry.get())
            assert 2022 <= year <= 2522
        except:
            return self._show_error(
                "Invalid year. Enter an integer value between 2022 and 2522."
            )

        # Validate dataset and selections
        if self.df.empty:
            return self._show_error("Dataset is not loaded or is empty.")
        if not self.selected_names:
            return self._show_error("No SNF names selected.")
        if "SNF_id" not in self.df.columns:
            return self._show_error("Dataset missing 'SNF_id' column.")

        # Prepare accumulators
        rows = []
        grand_totals = {col: 0.0 for col in self.cols if col != "SNF_id"}
        name_summaries = []

        # Loop over each SNF series
        for name in self.selected_names:
            # Initialize processor for this series
            try:
                proc = SNFProcessor(series_name=name, target_year=year)
            except Exception as e:
                return self._show_error(f"Error during SNF Processor:\n{e}")

            # Compute STDH DataFrame for this series
            df_stdh = proc.compute_stdh()

            # Ensure numeric and accumulate per-series total
            per_name_totals = {}
            for col in self.cols[1:]:
                df_stdh[col] = pd.to_numeric(df_stdh[col], errors="coerce")
                sum_val = df_stdh[col].sum()
                grand_totals[col] += sum_val
                per_name_totals[col] = sum_val

            # Store this series' totals for later statistics
            name_summaries.append(per_name_totals)

            # Format each row of df_stdh and prepend the series name
            disp = df_stdh.round(2).map(lambda x: f"{x:.2e}")
            disp.insert(0, "SNF_id", name)
            rows.extend(disp.itertuples(index=False, name=None))

        # If we have any data, append summary rows
        if rows:
            # Sum row
            sum_row = tuple(
                ["Sum"] + [f"{grand_totals[col]:.2e}" for col in self.cols[1:]]
            )
            rows.append(sum_row)

            # Average row
            n = len(name_summaries)
            avg_row = tuple(
                ["Average"] + [f"{grand_totals[col] / n:.2e}" for col in self.cols[1:]]
            )
            rows.append(avg_row)

            # Min row
            min_row = tuple(
                ["Min"]
                + [
                    f"{min(ns[col] for ns in name_summaries):.2e}"
                    for col in self.cols[1:]
                ]
            )
            rows.append(min_row)

            # Max row
            max_row = tuple(
                ["Max"]
                + [
                    f"{max(ns[col] for ns in name_summaries):.2e}"
                    for col in self.cols[1:]
                ]
            )
            rows.append(max_row)

        # Refresh the tree view with all rows
        tree = self.STDH_df_viewer.tree
        for item in tree.get_children():
            tree.delete(item)
        for idx, row in enumerate(rows):
            # choose base stripe tag
            base_tag = "oddrow" if idx % 2 else "evenrow"
            # apply red font to summary rows
            if row[0] in ("Sum", "Average", "Min", "Max"):
                tags = (base_tag, "summary")
            else:
                tags = (base_tag,)
            tree.insert("", "end", values=row, tags=tags)
        self._refresh_text()

        # Optionally save results to CSV
        if self.save_var.get() and rows:
            try:
                out_dir = create_output_dir(parent_folder_name="Results_Multiple")
                pd.DataFrame(rows, columns=self.cols).to_csv(
                    out_dir / "Multiple_STDH_results.csv", index=False
                )
            except Exception as e:
                self.multi_text.insert(tk.END, f"Error saving CSV: {e}\n")
