<body>
  <h1>SNFs Dataset Explore App</h1>
  <p>A professional Python project for searching, analyzing, and visualizing spent nuclear fuel (SNF) datasets with a Tkinter GUI.</p>

  <h2>1. Windows Environment Setup</h2>

  <h3>1.1 Install the Latest Python</h3>
  <ol>
    <li>Download the installer from <a href="https://www.python.org/downloads/" target="_blank">python.org/downloads/</a>.</li>
    <li>Run the installer. <span class="highlight">Ensure you select "Add Python to PATH"</span> before installation.</li>
    <li>Open <b>Command Prompt</b> and check your Python version:
      <pre>python --version</pre>
      or
      <pre>py --version</pre>
    </li>
  </ol>

  <h3>1.2 Setup Project Folder</h3>
  <pre>cd C:\path\to\your\project</pre>

  <h3>1.3 Create a Virtual Environment</h3>
  <pre>py -m venv pySNF_project</pre>
  <p>This creates an isolated environment in <code>pySNF_project</code>.</p>

  <h3>1.4 Activate the Virtual Environment</h3>
  <ul>
    <li><b>PowerShell:</b>
      <pre>.\pySNF_project\Scripts\Activate.ps1</pre>
    </li>
    <li><b>Command Prompt:</b>
      <pre>.\pySNF_project\Scripts\activate.bat</pre>
    </li>
  </ul>
  <p>Your prompt will show <code>(pySNF_project)</code> to confirm activation.</p>

  <h3>1.5 Install Dependencies</h3>
  <pre>pip install -r requirements.txt</pre>
  <p>Installs all required libraries as specified by your project.</p>

  <h3>1.6 Run the Application (cd .\src\pySNF)</h3>
  <pre>py main.py</pre>
  <p>This command starts the SNFs Tkinter GUI app.</p>

  <h3>1.7 Top-Level Packages</h3>
  <p>In your <code>requirements.txt</code>, list only the packages your project directly imports. These are called <strong>top-level packages</strong>. Pip will automatically install any sub-dependencies required by them.</p>
  <p>Example of common top-level packages for this project:</p>
  <ul>
    <li><code>numpy==2.2.6</code> (for efficient numerical arrays and linear algebra)</li>
    <li><code>pandas==2.2.3</code> (for data manipulation and analysis)</li>
    <li><code>matplotlib==3.10.3</code> (for creating basic 2D plots and figures)</li>
    <li><code>seaborn==0.13.2</code> (for advanced statistical visualizations)</li>
    <li><code>openpyxl==3.1.5</code> (if exporting data to Excel files)</li>
    <li><code>fsspec==2025.5.1</code> (for unified filesystem abstractions)</li>
    <li><code>python-dateutil==2.9.0.post0</code> (for flexible date parsing and arithmetic)</li>
    <li><code>pytz==2025.2</code> (for accurate timezone definitions)</li>
    <li><code>tzdata==2025.2</code> (to keep timezone data up to date)</li>
    <li><code>colorama==0.4.6</code> (for cross-platform terminal text coloring)</li>
  </ul>

  <h2>2. Project File Structure</h2>
  <pre class="structure">
    pySNF/<br>
    ├── README.md                # High‑level project overview, installation steps, and usage examples<br>  
    ├── LICENSE                  # Open‑source license terms (Apache 2.0)<br>  
    ├── requirements.txt         # Pin‑version Python dependencies for pip installation<br>  
    ├── .gitignore               # Patterns for files/directories to exclude from Git<br>  
    ├── src/                     # source code<br>
    │   ├── Notebook             # Dataset converter or developing project<br>
    │   └── pySNF/               # module name<br>
    │       ├── __init__.py      <br>
    │       ├── main.py          # Entry point of the app<br>
    │       ├── io.py            # File I/O: load/save datasets<br>
    │       ├── base.py          # Shared SNFs processor classes<br>
    │       ├── utils.py         # Utility functions (plotting, conversion, etc.)<br>
    │       └── FrameViewer/            # UI modules for Tkinter<br>
    │           ├── __init__.py         <br>
    │           ├── BaseFrame.py         # Abstract DataFrame viewer<br>
    │           ├── intro_snf.py         # Introduction page<br>
    │           ├── single_snf.py        # Single name/year search<br>
    │           ├── mutiple_snfs.py      # Multi-name search interface<br>
    │           ├── all_snfs.py          # All SNFs search interface<br>
    │           └── compare_snfs.py      # Compare SNFs interface<br>
    ├── data/                    # Data directory for SNF files (*.csv)<br>
    └── output/                  # Exported result files (*.csv, *.png)<br>

  </pre>

  <h3>Directory & File Description</h3>
  <table class="desc-table">
    <tr><th>File/Directory</th><th>Description</th></tr>
    <tr><td><code>main.py</code></td><td>App entry point. Launches the main Tkinter window.</td></tr>
    <tr><td><code>base.py</code></td><td>Core classes for SNF processing logic.</td></tr>
    <tr><td><code>utils.py</code></td><td>Plotting, data transformation, and utility functions.</td></tr>
    <tr><td><code>io_file.py</code></td><td>Handles data loading/saving to/from disk.</td></tr>
    <tr><td><code>FrameViewer/</code></td><td>All Tkinter GUI modules and views.</td></tr>
    <tr><td><code>snfs_details/</code></td><td>Input SNF dataset files (CSV format).</td></tr>
    <tr><td><code>output/</code></td><td>Directory for exported results, e.g., CSVs and PNG plots.</td></tr>
    <tr><td><code>README.md</code></td><td>This step-by-step guide and documentation.</td></tr>
  </table>

  <h2>3. Common Command Reference</h2>
  <pre>
cd C:\path\to\your\project
py -m venv pySNF_project
.\pySNF_project\Scripts\Activate.ps1
pip install -r requirements.txt
py main.py (cd .\src\pySNF)
  </pre>

  <h2>4. Best Practices</h2>
  <ul>
    <li>Use a virtual environment to avoid dependency conflicts.</li>
    <li>Update <code>requirements.txt</code> regularly via <code>pip freeze &gt; requirements.txt</code>.</li>
    <li>Do not commit sensitive data to version control.</li>
    <li>Consult this documentation for setup or troubleshooting.</li>
  </ul>
</body>

