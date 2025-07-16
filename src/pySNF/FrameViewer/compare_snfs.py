import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os, shutil
from PIL import Image, ImageTk
from typing import Literal

from base import SNFProcessor
from FrameViewer.BaseFrame import DataFrameViewer, build_scrollbar_canvas
from io_file import create_output_dir, set_SNFdetail_info, get_SNFdetail_TableUnit


class CompareSNFsFrame(tk.Frame):
    """
    A scrollable frame for searching SNF data by name and year
    """

    def __init__(self, parent, df: pd.DataFrame, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.df = df  # Store main dataset
        self.save_var = tk.BooleanVar(value=True)

        # Set up scrolling canvas
        self._photo_refs = {}
        self.canvas = tk.Canvas(self)  # Canvas for vertical scrolling
        vsb = tk.Scrollbar(
            self, orient="vertical", command=self.canvas.yview
        )  # Scrollbar
        self.canvas.configure(yscrollcommand=vsb.set)  # Link scrollbar
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame inside canvas
        self.inner = tk.Frame(self.canvas)
        self.window_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )
        # Update scrollregion when inner frame resizes
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        # Make inner frame width match canvas width
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfigure(self.window_id, width=e.width),
        )

        # Build UI inside scrollable inner frame
        self._build_ui()

    def _build_ui(self):
        # --- Input row ---
        row = tk.Frame(self.inner)
        row.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(row, text="First SNF ID:").pack(side=tk.LEFT)
        self.name_entry = tk.Entry(row, width=20)
        self.name_entry.insert(0, "1C2505")  # Default name
        self.name_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(row, text="First SNF Year (2022-2522):").pack(side=tk.LEFT)
        self.year_entry = tk.Entry(row, width=10)
        self.year_entry.insert(0, "2025")  # Default year
        self.year_entry.pack(side=tk.LEFT, padx=5)

        row2 = tk.Frame(self.inner)
        row2.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Label(row2, text="Second SNF ID:").pack(side=tk.LEFT)
        self.name_entry2 = tk.Entry(row2, width=20)
        self.name_entry2.insert(0, "1A0137")  # Default name
        self.name_entry2.pack(side=tk.LEFT, padx=5)
        tk.Label(row2, text="Second SNF Year (2022-2522):").pack(side=tk.LEFT)
        self.year_entry2 = tk.Entry(row2, width=10)
        self.year_entry2.insert(0, "2025")  # Default year
        self.year_entry2.pack(side=tk.LEFT, padx=5)
        tk.Button(row2, text="Output", command=self.compare_snfs).pack(side=tk.LEFT)
        tk.Checkbutton(row2, text="Save output", variable=self.save_var).pack(
            side=tk.LEFT, padx=(5, 0)
        )  # Save output checkbox
        # --- Details Viewer (above STDH_viewer) ---
        _, self.details_canvas, self.details_frame = build_scrollbar_canvas(
            self.inner, label="SNF Details"
        )

        # Set default empty values
        self.default_fields = set_SNFdetail_info(option=1)

        # --- STDH Viewer: full width, fixed 150px height ---
        self.STDH_viewer = self._make_viewer(
            parent=self.inner,
            height=150,
            columns=[
                "SNF_id",
                "DH(Watts/assy.)",
                "FN(n/s/assy.)",
                "FG(r/s/assy.)",
                "HG(r/s/kgSS304/MTU)",
            ],
            title="Source term & Decay Heat",
        )

        # ===== First container (Weight & Activity) =====
        container = tk.Frame(self.inner)
        container.pack(fill=tk.X, padx=10, pady=5)

        # --- Gram viewer: half width, fixed 200px height ---
        self.Gram_viewer = self._make_viewer(
            parent=container,
            height=200,
            columns=["nuclide", "gram/MTU", "gram/assy."],
            title="SNF1 Weight (gram)",
            side=tk.LEFT,
            expand=True,
        )
        # --- Ci viewer: half width, fixed 200px height ---
        self.Gram_viewer2 = self._make_viewer(
            parent=container,
            height=200,
            columns=["nuclide", "gram/MTU", "gram/assy."],
            title="SNF2 Weight (gram)",
            side=tk.LEFT,
            expand=True,
        )
        # ===== End First container (Weight & Activity) =====

        # ===== Second container2 (Weight & Activity) =====
        container2 = tk.Frame(self.inner)
        container2.pack(fill=tk.X, padx=10, pady=5)
        # --- Gram viewer: half width, fixed 200px height ---
        self.Ci_viewer = self._make_viewer(
            parent=container2,
            height=200,
            columns=["nuclide", "Ci/MTU", "Ci/assy."],
            title="SNF1 Activity (Ci)",
            side=tk.LEFT,
            expand=True,
        )
        # --- Ci viewer: half width, fixed 200px height ---
        self.Ci_viewer2 = self._make_viewer(
            parent=container2,
            height=200,
            columns=["nuclide", "Ci/MTU", "Ci/assy."],
            title="SNF2 Activity (Ci)",
            side=tk.LEFT,
            expand=True,
        )
        # ===== End Second container2 (Weight & Activity) =====

        # ===== First container Plot display (Weight) =====
        # --- Plot display area (grid, dynamic) ---
        self.plot_frame = tk.Frame(self.inner)
        self.plot_frame.pack(fill=tk.X, padx=10, pady=5)
        # configure two equal columns
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.columnconfigure(1, weight=1)
        self.weight_label = tk.Label(self.plot_frame, bd=1, relief="sunken")
        self.weight_label.grid(row=0, column=0, sticky="nsew", padx=5)
        self.weight_label2 = tk.Label(self.plot_frame, bd=1, relief="sunken")
        self.weight_label2.grid(row=0, column=1, sticky="nsew", padx=5)
        # ===== End First container Plot display (Weight) =====

        # ===== Second container Plot display (Activity) =====
        # --- Plot display area (grid, dynamic) ---
        self.plot_frame2 = tk.Frame(self.inner)
        self.plot_frame2.pack(fill=tk.X, padx=10, pady=5)
        # configure two equal columns
        self.plot_frame2.columnconfigure(0, weight=1)
        self.plot_frame2.columnconfigure(1, weight=1)
        # left: weight image
        self.ci_label = tk.Label(self.plot_frame2, bd=1, relief="sunken")
        self.ci_label.grid(row=0, column=0, sticky="nsew", padx=5)
        # right: activity image
        self.ci_label2 = tk.Label(self.plot_frame2, bd=1, relief="sunken")
        self.ci_label2.grid(row=0, column=1, sticky="nsew", padx=5)
        # ===== End Second container Plot display (Activity) =====

    def _make_viewer(
        self,
        parent,
        height: int,
        columns: list[str] = [],
        title: str = "",
        side: Literal["left", "right", "top", "bottom"] = tk.TOP,
        expand: bool = False,
    ) -> DataFrameViewer:
        """
        Helper to create a DataFrameViewer inside a frame:
        - height: fixed height in px (if provided)
        - side + expand: layout parameters
        """
        # Use full-width pack
        frame = tk.Frame(parent, height=height)
        frame.pack(
            fill=tk.X, side=side, expand=expand, padx=5, pady=5  # Always fill width
        )
        frame.pack_propagate(False)  # Prevent content-based resizing

        # Initialize DataFrameViewer
        df_empty = pd.DataFrame(columns=columns or [])
        viewer = DataFrameViewer(frame, df_empty, title=title)
        viewer.pack(fill=tk.BOTH, expand=True)
        return viewer

    def _clear_viewers(self):
        # Clear all rows in each viewer's tree
        for v in (
            self.STDH_viewer,
            self.Gram_viewer,
            self.Ci_viewer,
            self.Gram_viewer2,
            self.Ci_viewer2,
        ):
            for iid in v.tree.get_children():
                v.tree.delete(iid)

    def _insert_rows(self, viewer: DataFrameViewer, df: pd.DataFrame):
        # Insert DataFrame rows into viewer
        for vals in df.itertuples(index=False, name=None):
            viewer.tree.insert("", "end", values=vals)

    def _update_compare_grid(self, data1: dict, data2: dict):
        """Render a side-by-side comparison of two SNF detail dicts."""
        import tkinter.font as tkfont

        # Prepare font and padding
        cell_font = tkfont.Font(size=10)
        pad_x = 4
        pad_y = 4

        # Clear any existing widgets in the frame
        for widget in self.details_frame.winfo_children():
            widget.destroy()
        table_SNF_detail = get_SNFdetail_TableUnit()

        # Iterate over each field and layout SNF1 and SNF2 side by side
        # for row_idx, key in enumerate(self.default_fields):
        for row_idx, (key, key_unit) in enumerate(
            zip(self.default_fields, table_SNF_detail)
        ):
            # Format SNF1 value in scientific notation if numeric
            val1_raw = data1.get(key, "--")
            try:
                val1_num = float(val1_raw)
                val1_text = f"{val1_num:.2e}"
            except (TypeError, ValueError):
                val1_text = str(val1_raw)

            # Format SNF2 value in scientific notation if numeric
            val2_raw = data2.get(key, "--")
            try:
                val2_num = float(val2_raw)
                val2_text = f"{val2_num:.2e}"
            except (TypeError, ValueError):
                val2_text = str(val2_raw)

            # SNF1 field name label
            tk.Label(
                self.details_frame,
                text=f"{key_unit}:",
                font=cell_font,
                anchor="e",
                borderwidth=1,
                relief="solid",
                padx=pad_x,
                pady=pad_y,
            ).grid(row=row_idx, column=0, sticky="nsew")

            # SNF1 value label
            tk.Label(
                self.details_frame,
                text=val1_text,
                font=cell_font,
                anchor="w",
                borderwidth=1,
                relief="solid",
                padx=pad_x,
                pady=pad_y,
            ).grid(row=row_idx, column=1, sticky="nsew")

            # Vertical separator between SNF blocks
            tk.Frame(self.details_frame, bg="black", width=4).grid(
                row=row_idx, column=2, sticky="ns", padx=(2, 2)
            )

            # SNF2 field name label
            tk.Label(
                self.details_frame,
                text=f"{key_unit}:",
                font=cell_font,
                anchor="e",
                borderwidth=1,
                relief="solid",
                padx=pad_x,
                pady=pad_y,
            ).grid(row=row_idx, column=3, sticky="nsew")

            # SNF2 value label
            tk.Label(
                self.details_frame,
                text=val2_text,
                font=cell_font,
                anchor="w",
                borderwidth=1,
                relief="solid",
                padx=pad_x,
                pady=pad_y,
            ).grid(row=row_idx, column=4, sticky="nsew")

        # Configure column weights for equal expansion of SNF blocks
        for col in (0, 1, 3, 4):
            self.details_frame.grid_columnconfigure(col, weight=1, uniform="col")
        # Keep separator column fixed width
        self.details_frame.grid_columnconfigure(2, weight=0)

    def proc_img_layout(
        self, img_path1: str, img_path2: str, lbl_left: tk.Label, lbl_right: tk.Label
    ):
        # Load and resize both images
        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)

        self.inner.update_idletasks()
        pfw = self.plot_frame.winfo_width()
        if pfw <= 1:
            pfw = self.winfo_screenwidth() - 20  # fallback
        half_w = pfw // 2

        def resize(img):
            w, h = img.size
            ratio = half_w / w
            return img.resize((half_w, int(h * ratio)), Image.Resampling.LANCZOS)

        img1 = resize(img1)
        img2 = resize(img2)

        # Create PhotoImage instances
        photo1 = ImageTk.PhotoImage(img1)
        photo2 = ImageTk.PhotoImage(img2)

        # Assign to the labels
        lbl_left.config(image=photo1)
        lbl_right.config(image=photo2)

        # Store each image in our dict instead of as an attribute on Label
        self._photo_refs[lbl_left] = photo1
        self._photo_refs[lbl_right] = photo2

        # Update scrollregion
        self.inner.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def compare_snfs(self):
        # --- Input validation ---
        name = self.name_entry.get().strip()
        name2 = self.name_entry2.get().strip()
        yr = self.year_entry.get().strip()
        yr2 = self.year_entry2.get().strip()

        if self.df.empty:
            return messagebox.showerror(
                "Error", "Dataset in /snfs_details folder empty.\n"
            )
        if not name or not name2:
            return messagebox.showerror("Error", "Enter a SNF name.\n")
        if not yr or not yr2:
            return messagebox.showerror("Error", "Enter a year.\n")
        if "SNF_id" not in self.df.columns:
            return messagebox.showerror(
                "Error", " 'SNF_id' column missing in CSV Dataframe .\n"
            )
        try:
            yr = int(yr)
            yr2 = int(yr2)
            assert 2022 <= yr <= 2522
            assert 2022 <= yr2 <= 2522
        except:
            return messagebox.showerror(
                "Error", f" Year 2022–2522 only. You entered '{yr}'.\n"
            )

        # --- Fetch and validate first SNF ---
        matches = self.df[
            self.df["SNF_id"].astype(str).str.contains(name, case=False, na=False)
        ]
        if matches.empty:
            return messagebox.showerror("Error", f" No matches for '{name}' in {yr}.\n")
        snf_row1 = matches.iloc[0].to_dict()

        # --- Fetch and validate second SNF ---

        matches2 = self.df[
            self.df["SNF_id"].astype(str).str.contains(name2, case=False, na=False)
        ]
        if matches2.empty:
            return messagebox.showerror(
                "Error", f"No matches for second SNF '{name2}'."
            )
        snf_row2 = matches2.iloc[0].to_dict()

        self._update_compare_grid(snf_row1, snf_row2)

        # --- Computation with SNFProcessor ---
        proc = SNFProcessor(series_name=name, target_year=yr)
        df_stdh = proc.compute_stdh()
        df_gram = proc.compute_concentration()
        df_ci = proc.compute_activity()

        proc2 = SNFProcessor(series_name=name2, target_year=yr2)
        df_stdh2 = proc2.compute_stdh()
        df_gram2 = proc2.compute_concentration()
        df_ci2 = proc2.compute_activity()

        # --- Update viewers with new data ---
        self._clear_viewers()

        # Add a "Name" column to each STDH table for identification
        df_stdh.insert(0, "SNF_id", f"{name}_{yr}")
        df_stdh2.insert(0, "SNF_id", f"{name2}_{yr2}")

        # Merge the two STDH DataFrames so users can scroll through both in one table
        df_stdh_merged = pd.concat([df_stdh, df_stdh2], ignore_index=True)

        # Populate STDH viewer with merged results
        for row in df_stdh_merged.itertuples(index=False, name=None):
            self.STDH_viewer.tree.insert("", "end", values=row)

        self.Gram_viewer.label.config(text=f"{name}_{yr} Weight (gram)")
        self.Gram_viewer2.label.config(text=f"{name2}_{yr2} Weight (gram)")
        self.Ci_viewer.label.config(text=f"{name}_{yr} Activity (Ci)")
        self.Ci_viewer2.label.config(text=f"{name2}_{yr2} Activity (Ci)")

        self._insert_rows(self.Gram_viewer, df_gram)
        self._insert_rows(self.Gram_viewer2, df_gram2)

        self._insert_rows(self.Ci_viewer, df_ci)
        self._insert_rows(self.Ci_viewer2, df_ci2)

        # --- Generate plots and get directory ---
        output_dir = create_output_dir(parent_folder_name="Results_Compare_SNFs")
        # For the First SNF
        proc.plot_single_SNF(output_dir)
        proc.write_excel(df_stdh, df_ci, df_gram, output_dir, name)
        weight_img_path = os.path.join(output_dir, f"{name}_Weight.png")
        ci_img_path = os.path.join(output_dir, f"{name}_Activity.png")
        # For the Second SNF
        proc2.plot_single_SNF(output_dir)
        proc2.write_excel(df_stdh2, df_ci2, df_gram2, output_dir, name2)
        weight_img_path2 = os.path.join(output_dir, f"{name2}_Weight.png")
        ci_img_path2 = os.path.join(output_dir, f"{name2}_Activity.png")

        try:
            # layout weight plots side-by-side
            self.proc_img_layout(
                weight_img_path, weight_img_path2, self.weight_label, self.weight_label2
            )
            # layout activity plots side-by-side
            self.proc_img_layout(
                ci_img_path, ci_img_path2, self.ci_label, self.ci_label2
            )

        except Exception as e:
            return messagebox.showerror("Error", f" Error loading plots: {e}.\n")

        # --- Save results or not ---
        if not self.save_var.get():  #
            shutil.rmtree(output_dir)  # delete whole folder
