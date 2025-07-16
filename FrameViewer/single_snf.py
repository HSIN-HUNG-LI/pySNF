import tkinter as tk
from tkinter import messagebox
import pandas as pd
import os, shutil
from PIL import Image, ImageTk
from typing import Literal

from base import SNFProcessor
from FrameViewer.BaseFrame import DataFrameViewer, build_scrollbar_canvas
from io_file import create_output_dir, set_SNFdetail_info


class SingleSearchFrame(tk.Frame):
    """
    A scrollable frame for searching SNF data by name and year
    """

    def __init__(self, parent, df: pd.DataFrame, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.df = df  # Store main dataset
        self.save_var = tk.BooleanVar(value=True)

        # Set up scrolling canvas
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

        # Set default empty values
        self.default_fields = set_SNFdetail_info(option=1)
        # self._update_details_grid({key: "--" for key in self.default_fields})

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

        # --- Gram viewer: half width, fixed 200px height ---
        self.Gram_viewer = self._make_viewer(
            parent=container,
            height=200,
            columns=["nuclide", "gram/MTU", "gram/assy."],
            title="Weight (gram)",
            side=tk.LEFT,
            expand=True,
        )
        # --- Ci viewer: half width, fixed 200px height ---
        self.Ci_viewer = self._make_viewer(
            parent=container,
            height=200,
            columns=["nuclide", "Ci/MTU", "Ci/assy."],
            title="Activity (Ci)",
            side=tk.LEFT,
            expand=True,
        )

        # --- Plot display area (grid, dynamic) ---
        self.plot_frame = tk.Frame(self.inner)
        self.plot_frame.pack(fill=tk.X, padx=10, pady=5)

        # configure two equal columns
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.columnconfigure(1, weight=1)

        # left: weight image
        self.weight_label = tk.Label(self.plot_frame, bd=1, relief="sunken")
        self.weight_label.grid(row=0, column=0, sticky="nsew", padx=5)

        # right: activity image
        self.ci_label = tk.Label(self.plot_frame, bd=1, relief="sunken")
        self.ci_label.grid(row=0, column=1, sticky="nsew", padx=5)

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
        for v in (self.STDH_viewer, self.Gram_viewer, self.Ci_viewer):
            for iid in v.tree.get_children():
                v.tree.delete(iid)

    def _insert_rows(self, viewer: DataFrameViewer, df: pd.DataFrame):
        # Insert DataFrame rows into viewer
        for vals in df.itertuples(index=False, name=None):
            viewer.tree.insert("", "end", values=vals)

    def _update_details_grid(self, data: dict):
        # Local import for font support
        import tkinter.font as tkfont

        # Define font size and padding for table cells
        cell_font = tkfont.Font(size=10)
        pad_x = 4
        pad_y = 4

        # Clear existing widgets
        for widget in self.details_frame.winfo_children():
            widget.destroy()

        # Three items per row
        n_per_row = 4

        # Configure grid columns to expand equally with uniform width
        total_cols = n_per_row * 2
        for col_idx in range(total_cols):
            self.details_frame.grid_columnconfigure(col_idx, weight=1, uniform="col")

        # Prepare display values, defaulting to "--"
        values = [data.get(key, "--") for key in self.default_fields]

        # Populate table with bordered cells, larger font, and extra spacing
        for idx, (key, val) in enumerate(zip(self.default_fields, values)):
            row = idx // n_per_row
            col = (idx % n_per_row) * 2

            # Format numeric values in scientific notation with two decimals
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

    def search_single(self):
        # --- Input validation ---
        name = self.name_entry.get().strip()
        yr = self.year_entry.get().strip()

        if self.df.empty:
            return messagebox.showerror(
                "Error", "Dataset in /snfs_details folder empty.\n"
            )
        if not name:
            return messagebox.showerror("Error", "Enter a SNF name.\n")
        if not yr:
            return messagebox.showerror("Error", "Enter a year.\n")
        if "SNF_id" not in self.df.columns:
            return messagebox.showerror(
                "Error", " 'SNF_id' column missing in CSV Dataframe .\n"
            )
        try:
            y = int(yr)
            assert 2022 <= y <= 2522
        except:
            return messagebox.showerror(
                "Error", f" Year 2022–2522 only. You entered '{yr}'.\n"
            )

        # --- Data lookup ---
        matches = self.df[
            self.df["SNF_id"].astype(str).str.contains(name, case=False, na=False)
        ]
        if matches.empty:
            return messagebox.showerror("Error", f" No matches for '{name}' in {yr}.\n")

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

        # --- Generate plots and get directory ---
        output_dir = create_output_dir(parent_folder_name="Results_Single")
        proc.plot_single_SNF(output_dir)
        proc.write_excel(df_stdh, df_ci, df_gram, output_dir, name)

        weight_img_path = os.path.join(output_dir, f"{name}_Weight.png")
        ci_img_path = os.path.join(output_dir, f"{name}_Activity.png")

        try:
            w_img = Image.open(weight_img_path)
            c_img = Image.open(ci_img_path)

            # force layout so plot_frame has correct width
            self.inner.update_idletasks()
            pfw = self.plot_frame.winfo_width()
            if pfw <= 1:
                pfw = self.winfo_screenwidth() - 20  # fallback

            half_w = pfw // 2

            # resize preserving aspect ratio
            def resize(img):
                w, h = img.size
                ratio = half_w / w
                return img.resize(
                    (half_w, int(h * ratio)), resample=Image.Resampling.LANCZOS
                )

            w_img = resize(w_img)
            c_img = resize(c_img)

            # convert and keep refs on self
            self._w_photo = ImageTk.PhotoImage(w_img)
            self._c_photo = ImageTk.PhotoImage(c_img)

            self.weight_label.config(image=self._w_photo)
            self.ci_label.config(image=self._c_photo)

            # update scrollregion so frame height adjusts
            self.inner.update_idletasks()
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

        except Exception as e:
            return messagebox.showerror("Error", f" Error loading plots: {e}.\n")

        # --- Save results or not ---
        if not self.save_var.get():  #
            shutil.rmtree(output_dir)  # delete whole folder
