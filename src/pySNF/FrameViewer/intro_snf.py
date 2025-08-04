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
🔍 SNFs Dataset Explorer – Application Overview

📁 Required Data Structure
Please ensure the following directory and file structure is correctly set up within your working environment:

~/data/DataBase_SNFs/
1. all_stdh_dataset.csv
→ Serves as the primary SNF index containing summary information for all fuel entries.
2. Individual SNF files 
→ Each file includes weight and activity data specific to a single SNF unit. Filenames must match SNF identifiers.

~/data/DataBase_SNFs/
grid_database.parq
→ A preprocessed grid dataset in Parquet format used for interpolation, prediction, or comparison tasks.

🔧 Important: All files must be properly named and located as described. The application relies on this structure to function correctly. Missing or misnamed files will result in errors during execution.

Copyright © 2025 Laboratory of Prof. Rong-Jiun Sheu, National Tsing Hua University. All rights reserved.
""",
        )
        text.config(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
