#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reduce an interpolated grid database by sub-sampling along (Enrich, SP, Burnup, Cool)
according to a compact factor string, e.g., exp_folder_name="1412" -> (1,4,1,2).

- Input : grid_database.parq
- Output: grid_database_1412.parq

Assumptions:
- The Parquet file contains columns: 'Enrich', 'SP', 'Burnup', 'Cool'.
- Original grid definitions:
    Enrich:  np.arange(1.5, 6.1, 0.5)          # 1.5, 2.0, ..., 6.0
    SP:      np.arange(5, 46, 5)               # 5, 10, ..., 45
    Burnup:  np.arange(5000, 74100, 3000)      # 5000, 8000, ..., 74000
    CoolRaw: np.logspace(-5.75, 6.215, 150, base=e)
    Cool:    CoolRaw[1::cool_factor]           # skip the first and stride by factor
- Floating point comparisons are handled with a tolerance to avoid precision traps.
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Iterable, cast
from numpy.typing import NDArray
import numpy as np
import pandas as pd


def build_reduced_spaces(exp_folder_name: str):
    """
    Build the reduced value spaces from the factor string.

    Parameters
    ----------
    exp_folder_name : str
        Four digits string (e.g., "1412") meaning:
        (enrich_factor, sp_factor, bp_factor, cool_factor).

    Returns
    -------
    dict
        {'Enrich': np.ndarray, 'SP': np.ndarray, 'Burnup': np.ndarray, 'Cool': np.ndarray}
    """
    if len(exp_folder_name) != 4 or not exp_folder_name.isdigit():
        raise ValueError("exp_folder_name must be a 4-digit string like '1412'.")

    enrich_factor = int(exp_folder_name[0])
    sp_factor = int(exp_folder_name[1])
    bp_factor = int(exp_folder_name[2])
    cool_factor = int(exp_folder_name[3])

    # Original spaces
    enrich_space = np.arange(1.5, 6.1, 0.5)     # [1.5, 2.0, ..., 6.0]
    sp_space = np.arange(5, 46, 5)              # [5, 10, ..., 45]
    burnup_space = np.arange(5000, 74100, 3000) # [5000, 8000, ..., 74000]
    cool_space_raw = np.logspace(-5.75, 6.215, 150, base=math.e)

    # Reduction (note: Enrich/SP/Burnup keep index 0; Cool starts from index 1 by design)
    if enrich_factor < 1 or sp_factor < 1 or bp_factor < 1 or cool_factor < 1:
        raise ValueError("All factors must be >= 1.")
    enrich_space = enrich_space[0::enrich_factor]
    sp_space = sp_space[0::sp_factor]
    burnup_space = burnup_space[0::bp_factor]
    cool_space = cool_space_raw[1::cool_factor]

    return {
        "Enrich": enrich_space,
        "SP": sp_space,
        "Burnup": burnup_space,
        "Cool": cool_space,
    }


def isin_with_tol(
    values: Iterable[float],
    candidates: np.ndarray,
    tol: float = 1e-9
) -> NDArray[np.bool_]:
    """
    Vectorized membership check with tolerance for floating point columns.

    Returns
    -------
    NDArray[np.bool_]
        A 1-D boolean mask of length len(values), guaranteed to be an ndarray
        (never a scalar), to satisfy static type checkers.
    """
    v = np.asarray(values, dtype=float).reshape(-1, 1)     # (N, 1)
    c = np.asarray(candidates, dtype=float).reshape(1, -1) # (1, M)

    # (N, M) boolean array -> reduce along candidates axis -> (N,)
    mask = (np.abs(v - c) <= tol).any(axis=1)

    # Force an ndarray[bool] (not numpy.bool_) for static typing
    mask = np.asarray(mask, dtype=bool).reshape(-1)

    # Help Pylance/Pyright: tell it this is NDArray[np.bool_]
    return cast(NDArray[np.bool_], mask)

def reduce_grid(
    input_path: Path,
    output_path: Path,
    exp_folder_name: str = "1412",
    tol: float = 1e-8,
) -> pd.DataFrame:
    """
    Load the original grid Parquet, filter rows to the reduced grid, and write the result.

    Parameters
    ----------
    input_path : Path
        Path to the original Parquet file (grid_database.parq).
    output_path : Path
        Destination path for the reduced Parquet (e.g., grid_database_1412.parq).
    exp_folder_name : str
        Factor string controlling the stride per dimension (default: "1412").
    tol : float
        Absolute tolerance for float equality checks.

    Returns
    -------
    pd.DataFrame
        The reduced DataFrame that was written to disk.
    """
    spaces = build_reduced_spaces(exp_folder_name)

    # Read once
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    # df = pd.read_parquet(input_path)
    df = pd.read_excel(input_path, engine="openpyxl")
    required_cols = ["Enrich", "SP", "Burnup", "Cool"]
    out_cols = [f"{p}_prediction" for p in ("DH", "FN", "HG", "FG")]
    df.columns = required_cols + list(out_cols)

    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in input Parquet: {missing}")

    # Build boolean masks per dimension with tolerance
    m_en = isin_with_tol(df["Enrich"].to_numpy(), spaces["Enrich"], tol=tol)
    m_sp = isin_with_tol(df["SP"].to_numpy(), spaces["SP"], tol=tol)
    m_bp = isin_with_tol(df["Burnup"].to_numpy(), spaces["Burnup"], tol=tol)
    m_cl = isin_with_tol(df["Cool"].to_numpy(), spaces["Cool"], tol=tol)

    mask = m_en & m_sp & m_bp & m_cl
    reduced = df.loc[mask].copy()

    # Persist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reduced.to_excel(output_path, index=False, engine="openpyxl", header=False)
    # reduced.to_parquet(output_path, index=False)

    # Reporting
    print("== Grid Reduction Summary ==")
    print(f"Factors (E,SP,Bp,Cool): {tuple(int(d) for d in exp_folder_name)}")
    print(f"Input rows : {len(df):,}")
    print(f"Output rows: {len(reduced):,}")
    print("-- Unique levels kept --")
    for k in ["Enrich", "SP", "Burnup", "Cool"]:
        print(f"{k:<7}: {len(np.unique(reduced[k]))} levels")

    return reduced


if __name__ == "__main__":
    # Default I/O: current working directory
    # in_path = Path("grid_database.parq").resolve()
    # out_path = in_path.with_name("grid_database_1412.parq")
    project_root = Path.cwd()
    input_data_file = project_root / "data" / "grid_database_1111.xlsx"
    exp_folder_name = "1412"
    output_data_file = project_root / "data" / f"grid_database_{exp_folder_name}.xlsx"
    # Run reduction
    reduce_grid(input_path=input_data_file, output_path=output_data_file, exp_folder_name=exp_folder_name, tol=1e-8)
