import tkinter as tk
from tkinter import ttk
import pandas as pd
from typing import Tuple
import tkinter.font as tkfont

import tkinter as tk
from tkinter import ttk
import pandas as pd


class DataFrameViewer(tk.Frame):
    def __init__(self, parent, df: pd.DataFrame, title: str, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # If a title is provided, display it at the top and keep reference
        if title:
            self.label = tk.Label(self, text=title, font=("Arial", 12, "bold"))
            self.label.pack(padx=5, pady=(5, 0))

        # Create a container Frame for the Treeview and scrollbars
        container = tk.Frame(self)
        container.pack(fill="both", expand=True, padx=5, pady=5)

        # Vertical and horizontal scrollbars
        vsb = ttk.Scrollbar(container, orient="vertical")
        hsb = ttk.Scrollbar(container, orient="horizontal")

        self.tree = ttk.Treeview(
            container,
            columns=list(df.columns),
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )

        # Place scrollbars and Treeview in a grid
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self.tree.grid(row=0, column=0, sticky="nsew")

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        vsb.configure(command=self.tree.yview)
        hsb.configure(command=self.tree.xview)

        # Set column headings and alignment
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center")

        # Insert each row of the DataFrame into the Treeview
        for row in df.to_numpy().tolist():
            self.tree.insert("", "end", values=row)

    def load_dataframe(self, df: pd.DataFrame):
        """
        Load and display a new DataFrame, replacing existing content.
        Numeric values are formatted in scientific notation with two decimal places.
        Columns are resized to equally fill the available width.

        Parameters:
        df (pd.DataFrame): The DataFrame to display.
        """
        # Store a copy of the DataFrame
        self.df = df.copy()

        # Clear existing headings and rows
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
        self.tree.delete(*self.tree.get_children())

        # Configure new columns
        cols = list(self.df.columns)
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="center", stretch=True)

        # Insert rows with formatted values
        for _, row in self.df.iterrows():
            values = []
            for col in cols:
                val = row[col]
                try:
                    num = float(val)
                    values.append(f"{num:.2e}")
                except:
                    values.append(str(val))
            self.tree.insert("", "end", values=values)

        # Resize columns equally to fill the container
        self.tree.update_idletasks()
        total_width = self.tree.winfo_width()
        n = max(len(cols), 1)
        col_width = total_width // n
        for col in cols:
            self.tree.column(col, width=col_width, stretch=True)


def build_scrollbar_canvas(
    parent: tk.Widget, label: str = "", height: int = 120
) -> Tuple[tk.LabelFrame, tk.Canvas, tk.Frame]:
    """
    Create a scrollable canvas with a centered, enlarged title and vertical scrollbar.
    Returns: (label_frame, canvas, inner_frame)
    """
    # Bold font for the LabelFrame title
    label_font = tkfont.Font(size=12, weight="bold")
    # LabelFrame container with title centered at the top
    label_frame = tk.LabelFrame(
        parent, text=label, font=label_font, labelanchor="n", bd=2, relief="groove"
    )
    label_frame.pack(fill=tk.BOTH, padx=15, pady=5, expand=True)

    # Main canvas for scrollable content
    canvas = tk.Canvas(label_frame, height=height, highlightthickness=0)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Vertical scrollbar
    vbar = tk.Scrollbar(label_frame, orient=tk.VERTICAL, command=canvas.yview)
    vbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.configure(yscrollcommand=vbar.set)

    # Inner frame inside the canvas
    inner = tk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def update_scrollregion(event=None):
        # Ensure inner frame width matches canvas and update scrollregion
        canvas.itemconfigure(window_id, width=canvas.winfo_width())
        canvas.configure(scrollregion=canvas.bbox("all"))

    # Bind resizing events
    inner.bind("<Configure>", update_scrollregion)
    canvas.bind("<Configure>", update_scrollregion)

    # Enable mousewheel scrolling across platforms
    def _on_mousewheel(event):
        if event.delta:
            canvas.yview_scroll(int(-event.delta / 120), "units")
        elif event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_mousewheel)
    canvas.bind_all("<Button-5>", _on_mousewheel)

    return label_frame, canvas, inner
