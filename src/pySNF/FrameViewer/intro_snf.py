import tkinter as tk
import tkinter.font as tkfont


class IntroFrame(tk.Frame):
    """
    Introduction view for pySNF: SNF Dataset Explorer App.
    Displays app design and purpose.
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
🔍 pySNF: SNF Dataset Explorer App:
pySNF is a Python App designed to facilitate data exploration and prediction of spent nuclear fuel characteristics
Exploration: data query, analysis, and visualization of the existing spent nuclear fuel dataset
Prediction: estimate decay heat and source terms of a spent nuclear fuel according to its burnup and cooling time

Keywords:
spent nuclear fuel (SNF), composition, nuclide, weight, activity,
decay heat (DH), source term (ST), fuel neutron (FN), fuel gamma (FG), hardware gamma (HG),
burnup (Bp), cooling time (Ct), enrichment (En), specific power (Sp), etc. 

📁 App installation and usage:
(0) Unpacking the download file: pySNF.zip
(1) Installing the latest Python on your platform
(2) Installing the required packages using > pip install -r requirements.txt
(3) Executing the app by entering > py main.py

Please ensure the following directories and files are correctly set up:
pySNF/data/DataBase_SNFs/:
→ all_stdh_dataset.csv: fuel ID and important information for all spent nuclear fuels
→ Individual SNF files: nuclides (weight and activity) in each fuel as a function of colling time 
pySNF/data/DataBase_GridPoint/:
→ grid_database.parq: a pre-processed grid database in Parquet format used for predicting SNF characteristics

Developers: Hsin-Hung Li, Po-Chen Tsai, and Rong-Jiun Sheu at National Tsing Hua University.
""",
        )
        text.config(state=tk.DISABLED)
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
