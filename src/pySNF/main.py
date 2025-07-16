import tkinter as tk
from FrameViewer.intro_snf import IntroFrame
from FrameViewer.single_snf import SingleSearchFrame
from FrameViewer.compare_snfs import CompareSNFsFrame
from FrameViewer.mutiple_snfs import MultipleSearchFrame
from FrameViewer.all_snfs import AllSNFsFrame
from io_file import load_dataset, get_stdh_path

class SearchApp(tk.Tk):
    """Main window for SNFs Dataset Search App."""

    def __init__(self):
        super().__init__()
        self.title(
            "SNFs Explorer App © 2025 Lab of Prof. Rong-Jiun Sheu, NTHU. All rights reserved."
        )
        self.geometry("1200x600")

        # Load dataset once to avoid repeated I/O (MessageBox in io file)
        self.df_path = get_stdh_path()
        self.df = load_dataset(self.df_path)

        # Define available modes
        modes = ["Introduction", "Single SNF", "Compare SNFs", "Multiple SNFs", "All SNFs"]
        self.mode_var = tk.StringVar(value=modes[2])

        # Mode selection dropdown
        tk.OptionMenu(
            self,
            self.mode_var,
            *modes,
            command=lambda _=None: self._show_frame(self.mode_var.get())
        ).pack(anchor="nw", padx=10, pady=5)

        # Initialize frames for each mode
        self.frames = {
            "Introduction": IntroFrame(self),
            "Single SNF": SingleSearchFrame(self, self.df),
            "Compare SNFs": CompareSNFsFrame(self, self.df),
            "Multiple SNFs": MultipleSearchFrame(self, self.df),
            "All SNFs": AllSNFsFrame(self),
        }

        # Display the initial frame
        self._show_frame(self.mode_var.get())

    def _show_frame(self, mode: str) -> None:
        """Hide all frames, then show the selected one."""
        for frame in self.frames.values():
            frame.pack_forget()
        self.frames[mode].pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    SearchApp().mainloop()
