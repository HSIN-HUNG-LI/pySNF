import tkinter as tk
import tkinter.font as tkfont


class IntroFrame(tk.Frame):
    """
    Introduction view for SNFs Dataset Search App.
    Displays app purpose and folder/CSV requirements.
    """

    def __init__(self, parent):
        super().__init__(parent)
        # Define and apply a larger, readable font
        font = tkfont.Font(family="Helvetica", size=12)

        # Create a read-only text widget for instructions
        text = tk.Text(self, wrap=tk.WORD, height=10, font=font)
        text.insert(
            tk.END,
            """
This app is used to explore SNFs data by name.

Please ensure there is a folder named ~/snfs_details in your working directory, containing:
1. all_stdh_dataset.csv (the main SNF index)
2. individual weight and activity CSV files for each SNF

Copyright © 2025 Laboratory of Prof. Rong-Jiun Sheu, National Tsing Hua University. All rights reserved.
""",
        )
        text.config(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
