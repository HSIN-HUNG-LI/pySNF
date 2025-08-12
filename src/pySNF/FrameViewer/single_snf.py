import os
import shutil
import tkinter as tk
from tkinter import messagebox
from typing import Literal, Optional

import pandas as pd
from PIL import Image, ImageTk

from base import SNFProcessor
from FrameViewer.BaseFrame import DataFrameViewer, build_scrollbar_canvas
from io_file import create_output_dir, set_SNFdetail_info, get_SNFdetail_TableUnit


class SingleSearchFrame(tk.Frame):
    """
    A scrollable frame for searching SNF data by name and year.
    Features:
    - Loads and filters a provided DataFrame of SNF metadata.
    - Computes STDH, mass (gram), and activity (Ci) for a given SNF/year.
    - Displays details in a grid plus three tabular viewers.
    - Generates plots and (optionally) keeps or removes the output folder.
    """

    def __init__(self, parent: tk.Misc, df: pd.DataFrame, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        # Main dataset and Save toggle
        self.df: pd.DataFrame = df
        self.save_var = tk.BooleanVar(value=True)

        # Canvas + vertical scrollbar (for full-frame scrolling)
        self.canvas = tk.Canvas(self)  # Canvas for vertical scrolling
        vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame embedded inside the canvas
        self.inner = tk.Frame(self.canvas)
        self.window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # Keep scrollregion and inner width in sync
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self.window_id, width=e.width),
        )

        # Build UI inside the scrollable inner frame
        self._build_ui()

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        """Build input controls, details grid, three viewers, and plot placeholders."""
        # --- Input row ---
        row = tk.Frame(self.inner)
        row.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(row, text="SNF Name:").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(row, width=20)
        self.name_entry.insert(0, "1C2505")  # Default name
        self.name_entry.pack(side=tk.LEFT, padx=5)

        tk.Label(row, text="Year (2022-2522):").pack(side=tk.LEFT)
        self.year_entry = tk.Entry(row, width=10)
        self.year_entry.insert(0, "2025")  # Default year
        self.year_entry.pack(side=tk.LEFT, padx=5)

        tk.Button(row, text="Output", command=self.search_single).pack(side=tk.LEFT)
        tk.Checkbutton(row, text="Save output", variable=self.save_var).pack(
            side=tk.LEFT, padx=(5, 0)
        )  # Save output checkbox

        # --- Details Viewer (above STDH_viewer) ---
        _, self.details_canvas, self.details_frame = build_scrollbar_canvas(
            self.inner, label="SNF Details"
        )

        # Default empty values (ordering/keys provided by external helper)
        self.default_fields = set_SNFdetail_info(option=1)

        # --- STDH Viewer: full width, fixed 150px height ---
        self.STDH_viewer = self._make_viewer(
            parent=self.inner,
            height=150,
            columns=[
                "DH(Watts/assy.)",
                "FN(n/s/assy.)",
                "FG(r/s/assy.)",
                "HG(r/s/kgSS304/MTU)",
            ],
            title="Source term & Decay Heat",
        )

        # --- Weight & Activity side-by-side container ---
        container = tk.Frame(self.inner)
        container.pack(fill=tk.X, padx=10, pady=5)

        # Gram viewer: half width, fixed 200px height
        self.Gram_viewer = self._make_viewer(
            parent=container,
            height=200,
            columns=["nuclide", "gram/MTU", "gram/assy."],
            title="Weight (gram)",
            side=tk.LEFT,
            expand=True,
        )

        # Ci viewer: half width, fixed 200px height
        self.Ci_viewer = self._make_viewer(
            parent=container,
            height=200,
            columns=["nuclide", "Ci/MTU", "Ci/assy."],
            title="Activity (Ci)",
            side=tk.LEFT,
            expand=True,
        )

        # --- Plot display area (two cells in a grid) ---
        self.plot_frame = tk.Frame(self.inner)
        self.plot_frame.pack(fill=tk.X, padx=10, pady=5)

        # Two equal columns
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.columnconfigure(1, weight=1)

        # Left: weight image
        self.weight_label = tk.Label(self.plot_frame, bd=1, relief="sunken")
        self.weight_label.grid(row=0, column=0, sticky="nsew", padx=5)

        # Right: activity image
        self.ci_label = tk.Label(self.plot_frame, bd=1, relief="sunken")
        self.ci_label.grid(row=0, column=1, sticky="nsew", padx=5)

    def _make_viewer(
        self,
        parent: tk.Misc,
        height: int,
        columns: Optional[list[str]] = None,
        title: str = "",
        side: Literal["left", "right", "top", "bottom"] = tk.TOP,
        expand: bool = False,
    ) -> DataFrameViewer:
        """
        Create a DataFrameViewer in a fixed-height frame to prevent layout jumps.

        Parameters
        ----------
        parent : tk.Misc
            Parent container.
        height : int
            Fixed height (px) for the viewer container.
        columns : list[str] | None
            Initial DataFrame columns; empty if None.
        title : str
            Title displayed by the viewer.
        side : Literal["left","right","top","bottom"]
            Pack side for the container.
        expand : bool
            Whether the container expands within its row.
        """
        frame = tk.Frame(parent, height=height)
        frame.pack(fill=tk.X, side=side, expand=expand, padx=5, pady=5)  # Full width
        frame.pack_propagate(False)  # Keep fixed height

        df_empty = pd.DataFrame(columns=columns or [])
        viewer = DataFrameViewer(frame, df_empty, title=title)
        viewer.pack(fill=tk.BOTH, expand=True)
        return viewer

    # ──────────────────────────────────────────────────────────────────────────
    # Viewer utilities
    # ──────────────────────────────────────────────────────────────────────────
    def _clear_viewers(self) -> None:
        """Remove all rows from each DataFrameViewer's underlying treeview."""
        for v in (self.STDH_viewer, self.Gram_viewer, self.Ci_viewer):
            for iid in v.tree.get_children():
                v.tree.delete(iid)

    def _insert_rows(self, viewer: DataFrameViewer, df: pd.DataFrame) -> None:
        """Insert all rows of a DataFrame into a viewer's treeview."""
        for vals in df.itertuples(index=False, name=None):
            viewer.tree.insert("", "end", values=vals)

    # ──────────────────────────────────────────────────────────────────────────
    # Details grid
    # ──────────────────────────────────────────────────────────────────────────
    def _update_details_grid(self, data: dict) -> None:
        """
        Render the details table into `self.details_frame` with bordered cells,
        uniform column sizing, and scientific notation for numeric values.
        """
        import tkinter.font as tkfont

        cell_font = tkfont.Font(size=10)
        pad_x, pad_y = 4, 4

        # Clear existing widgets
        for widget in self.details_frame.winfo_children():
            widget.destroy()

        # Four items per row (key/value pairs)
        n_per_row = 4

        # Configure grid columns to expand equally with uniform width
        total_cols = n_per_row * 2
        for col_idx in range(total_cols):
            self.details_frame.grid_columnconfigure(col_idx, weight=1, uniform="col")

        # Prepare display values, defaulting to "--"
        values = [data.get(key, "--") for key in self.default_fields]
        table_SNF_detail = get_SNFdetail_TableUnit()

        # Populate table; format numbers in scientific notation (2 decimals)
        for idx, (key, val) in enumerate(zip(table_SNF_detail, values)):
            row = idx // n_per_row
            col = (idx % n_per_row) * 2

            try:
                num = float(val)
                text_val = f"{num:.2e}"
            except (TypeError, ValueError):
                text_val = str(val)

            # Field name cell
            tk.Label(
                self.details_frame,
                text=f"{key}:",
                font=cell_font,
                anchor="e",
                borderwidth=1,
                relief="solid",
                padx=pad_x,
                pady=pad_y,
            ).grid(row=row, column=col, sticky="nsew")

            # Field value cell
            tk.Label(
                self.details_frame,
                text=text_val,
                font=cell_font,
                anchor="w",
                borderwidth=1,
                relief="solid",
                padx=pad_x,
                pady=pad_y,
            ).grid(row=row, column=col + 1, sticky="nsew")

    # ──────────────────────────────────────────────────────────────────────────
    # Main action
    # ──────────────────────────────────────────────────────────────────────────
    def search_single(self) -> None:
        """
        Validate inputs, compute STDH/grams/Ci via SNFProcessor, update viewers
        and details, render plots, and optionally delete the output folder if
        'Save output' is unchecked.
        """
        # --- Input validation ---
        name = self.name_entry.get().strip()
        yr = self.year_entry.get().strip()

        if self.df.empty:
            messagebox.showerror("Error", "Dataset in /snfs_details folder empty.\n")
            return
        if not name:
            messagebox.showerror("Error", "Enter a SNF name.\n")
            return
        if not yr:
            messagebox.showerror("Error", "Enter a year.\n")
            return
        if "SNF_id" not in self.df.columns:
            messagebox.showerror("Error", " 'SNF_id' column missing in CSV Dataframe .\n")
            return

        try:
            y = int(yr)
            assert 2022 <= y <= 2522
        except Exception:
            messagebox.showerror("Error", f" Year 2022–2522 only. You entered '{yr}'.\n")
            return

        # --- Data lookup (case-insensitive substring match) ---
        matches = self.df[self.df["SNF_id"].astype(str).str.contains(name, case=False, na=False)]
        if matches.empty:
            messagebox.showerror("Error", f" No matches for '{name}' in {yr}.\n")
            return

        # --- Insert SNF metadata details into details_viewer ---
        snf_row = matches.iloc[0].to_dict()
        self._update_details_grid(snf_row)

        # --- Computation with SNFProcessor ---
        proc = SNFProcessor(series_name=name, target_year=y)
        df_stdh = proc.compute_stdh()
        df_gram = proc.compute_concentration()
        df_ci = proc.compute_activity()

        # --- Update viewers with new data ---
        self._clear_viewers()
        self._insert_rows(self.STDH_viewer, df_stdh)
        self._insert_rows(self.Gram_viewer, df_gram)
        self._insert_rows(self.Ci_viewer, df_ci)

        # --- Generate plots and write Excel ---
        output_dir = create_output_dir(parent_folder_name="Results_Single")
        proc.plot_single_SNF(output_dir)
        proc.write_excel(df_stdh, df_ci, df_gram, output_dir, name)

        # --- Load and display plots ---
        weight_img_path = os.path.join(output_dir, f"{name}_Weight.png")
        ci_img_path = os.path.join(output_dir, f"{name}_Activity.png")

        try:
            w_img = Image.open(weight_img_path)
            c_img = Image.open(ci_img_path)

            self.inner.update_idletasks()
            pfw = self.plot_frame.winfo_width()
            if pfw <= 1:
                pfw = self.winfo_screenwidth() - 20
            half_w = pfw // 2

            def _resize_keep_aspect(img: Image.Image, target_w: int) -> Image.Image:
                w, h = img.size
                ratio = target_w / max(1, w)
                return img.resize((target_w, int(h * ratio)), resample=Image.Resampling.LANCZOS)

            w_img = _resize_keep_aspect(w_img, half_w)
            c_img = _resize_keep_aspect(c_img, half_w)

            self._w_photo = ImageTk.PhotoImage(w_img)
            self._c_photo = ImageTk.PhotoImage(c_img)

            self.weight_label.config(image=self._w_photo)
            self.ci_label.config(image=self._c_photo)

            self.inner.update_idletasks()
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

        except Exception as e:
            messagebox.showerror("Error", f" Error loading plots: {e}.\n")
            return

        # --- Save results or not ---
        if not self.save_var.get():
            shutil.rmtree(output_dir)  # delete whole folder
