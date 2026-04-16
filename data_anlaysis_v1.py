#!/usr/bin/env python3
"""
Oxford Battery Degradation Dataset 1 — robust MATLAB v5 loader + extractor + metrics

This script auto-detects the correct cell container by testing candidates.

Outputs (in out_dir):
- cycle_metrics.csv        (capacity fade, Tmax, R0 if available)
- ica_features.csv         (ICA features from OCVdc)
- master_table.csv         (merged metrics + ICA)
- traces_long.(parquet or csv.gz)  (optional, can be huge)

Run:
  python data_analysis_v3.py --mat Oxford_Battery_Degradation_Dataset_1.mat --out_dir out
Optional (skip huge long df):
  python data_analysis_v3.py --mat Oxford_Battery_Degradation_Dataset_1.mat --out_dir out --no_long_df
Optional (just inspect structure and exit):
  python data_analysis_v3.py --mat Oxford_Battery_Degradation_Dataset_1.mat --probe
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# Robust MATLAB v5 loader
# ---------------------------

def load_mat_v5_robust(path: str) -> Dict[str, Any]:
    from scipy.io import loadmat

    common_kwargs = dict(squeeze_me=True, struct_as_record=False)

    try:
        return loadmat(path, **common_kwargs)
    except Exception as e1:
        print(f"[ERROR] scipy.io.loadmat failed (first attempt): {type(e1).__name__}: {e1}")

    # Retry with verify_compressed_data_integrity=False if your SciPy supports it
    try:
        return loadmat(path, **common_kwargs, verify_compressed_data_integrity=False)
    except TypeError as e2:
        print(f"[WARN] SciPy does not support verify_compressed_data_integrity=False: {e2}")
        raise RuntimeError(
            "Update SciPy (recommended): conda install -c conda-forge 'scipy>=1.10'"
        ) from e2
    except Exception as e3:
        print(f"[ERROR] scipy.io.loadmat failed (retry): {type(e3).__name__}: {e3}")
        raise


def nonprivate_keys(mat: Dict[str, Any]) -> List[str]:
    return [k for k in mat.keys() if not k.startswith("__")]


# ---------------------------
# SciPy MATLAB-struct helpers
# ---------------------------

def is_mat_struct(x: Any) -> bool:
    return hasattr(x, "_fieldnames")


def fieldnames(x: Any) -> List[str]:
    return list(getattr(x, "_fieldnames", []) or [])


def unwrap_obj(x: Any) -> Any:
    """Unwrap 1-element object arrays."""
    if isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        return x.flat[0]
    return x


def coerce_1d_float(x: Any) -> np.ndarray:
    arr = np.array(x)
    return arr.astype(float).reshape(-1)


def has_cycle_fields(cell_obj: Any) -> bool:
    """A 'cell' struct should have fields like cyc0100, cyc0200, ..."""
    cell_obj = unwrap_obj(cell_obj)
    if not is_mat_struct(cell_obj):
        return False
    fns = fieldnames(cell_obj)
    return any(re.match(r"(?i)^cyc\d+$", f) for f in fns)


def looks_like_cell_array(arr: Any, min_cells: int = 4) -> bool:
    """
    True if arr is an object array whose elements look like cell structs
    (i.e., contain cycXXXX fields).
    """
    if not isinstance(arr, np.ndarray):
        return False
    arr = np.array(arr)
    if arr.dtype != object:
        return False
    flat = arr.reshape(-1)
    if flat.size < min_cells:
        return False

    # Check a few elements
    checks = 0
    good = 0
    for j in range(min(8, flat.size)):
        checks += 1
        if has_cycle_fields(flat[j]):
            good += 1
    return good >= max(1, checks // 2)


# ---------------------------
# Expected segments / signals
# ---------------------------

SEGMENTS = ("C1ch", "C1dc", "OCVch", "OCVdc")
SIGNALS = ("t", "v", "q", "T", "i")  # i often absent in this big dataset


@dataclass
class Trace:
    cell: int
    cyc: int
    segment: str
    t: np.ndarray
    v: np.ndarray
    q: np.ndarray
    T: np.ndarray
    i: Optional[np.ndarray] = None


# ---------------------------
# Cell container detection (fixes your error)
# ---------------------------

def iter_cells_any(obj: Any) -> Optional[List[Tuple[int, Any]]]:
    """
    Try to interpret 'obj' as the cell container.
    Returns list[(cell_id, cell_obj)] or None if not recognized.
    Handles many layouts:
      - struct with fields Cell1..Cell8
      - struct with field 'Cells'/'cells'/'Cell'/'cell' that is an array
      - struct with ANY field that is a 1x8 object array of cell structs
      - direct 1x8 object array at top level
    """
    obj = unwrap_obj(obj)

    # Case 1: direct object array of cells
    if looks_like_cell_array(obj):
        flat = np.array(obj, dtype=object).reshape(-1)
        return [(i + 1, flat[i]) for i in range(flat.size)]

    # Case 2: struct holding cells in fields
    if is_mat_struct(obj):
        fns = fieldnames(obj)

        # 2a) Cell1..Cell8 style
        cell_fields = [f for f in fns if re.match(r"(?i)^cell\d+$", f)]
        if cell_fields:
            cell_fields_sorted = sorted(cell_fields, key=lambda s: int(re.findall(r"\d+", s)[0]))
            out = []
            for f in cell_fields_sorted:
                cid = int(re.findall(r"\d+", f)[0])
                out.append((cid, getattr(obj, f)))
            return out

        # 2b) Common container field names
        for fname in ("Cells", "cells", "Cell", "cell"):
            if fname in fns:
                cand = getattr(obj, fname)
                if looks_like_cell_array(cand):
                    flat = np.array(cand, dtype=object).reshape(-1)
                    return [(i + 1, flat[i]) for i in range(flat.size)]

        # 2c) Search any field for a cell array
        for f in fns:
            cand = getattr(obj, f)
            if looks_like_cell_array(cand):
                flat = np.array(cand, dtype=object).reshape(-1)
                return [(i + 1, flat[i]) for i in range(flat.size)]

    return None


def find_cell_container(mat: Dict[str, Any]) -> Tuple[str, List[Tuple[int, Any]]]:
    """
    Searches ALL top-level variables and returns the one that contains cells.

    Supports the Oxford layout where Cell1..Cell8 are TOP-LEVEL variables.
    """
    keys = nonprivate_keys(mat)
    if not keys:
        raise RuntimeError("No non-private variables found in MAT file.")

    # ✅ CASE 0: top-level keys are Cell1..Cell8
    top_cell_keys = [k for k in keys if re.match(r"(?i)^cell\d+$", k)]
    if len(top_cell_keys) >= 4:
        top_cell_keys = sorted(top_cell_keys, key=lambda s: int(re.findall(r"\d+", s)[0]))
        cells = [(int(re.findall(r"\d+", k)[0]), mat[k]) for k in top_cell_keys]
        return "top_level(Cell1..CellN)", cells

    # CASE 1: a top-level variable IS the container (array/struct)
    for k in keys:
        cells = iter_cells_any(mat[k])
        if cells and len(cells) >= 4:
            return k, cells

    # CASE 2: nested container inside a top-level struct
    for k in keys:
        obj = unwrap_obj(mat[k])
        if is_mat_struct(obj):
            for f in fieldnames(obj):
                cells = iter_cells_any(getattr(obj, f))
                if cells and len(cells) >= 4:
                    return f"{k}.{f}", cells

    raise RuntimeError(
        "Could not detect cell layout. "
        "The file does not appear to contain an 8-cell container in the expected formats."
    )


# ---------------------------
# Cycle / segment reading
# ---------------------------

def iter_cycles(cell_obj: Any) -> List[Tuple[int, Any]]:
    cell_obj = unwrap_obj(cell_obj)
    if not is_mat_struct(cell_obj):
        return []
    cyc_fields = [f for f in fieldnames(cell_obj) if re.match(r"(?i)^cyc\d+$", f)]
    cyc_fields_sorted = sorted(cyc_fields, key=lambda s: int(re.findall(r"\d+", s)[0]))
    out = []
    for f in cyc_fields_sorted:
        cyc_num = int(re.findall(r"\d+", f)[0])  # cyc0100 -> 100
        out.append((cyc_num, getattr(cell_obj, f)))
    return out


def read_segment(seg_obj: Any) -> Dict[str, np.ndarray]:
    seg_obj = unwrap_obj(seg_obj)
    if not is_mat_struct(seg_obj):
        return {}
    fns = fieldnames(seg_obj)
    d: Dict[str, np.ndarray] = {}
    for s in SIGNALS:
        if s in fns:
            d[s] = coerce_1d_float(getattr(seg_obj, s))
    return d


def extract_traces(mat: Dict[str, Any]) -> List[Trace]:
    container_name, cells = find_cell_container(mat)
    print(f"[INFO] Cell container detected at: {container_name}")
    print(f"[INFO] Number of cells detected: {len(cells)}")

    traces: List[Trace] = []
    for cell_id, cell_obj in cells:
        cycles = iter_cycles(cell_obj)
        if not cycles:
            print(f"[WARN] No cycle fields found in cell index {cell_id}. Skipping.")
            continue

        for cyc_num, cyc_obj in cycles:
            cyc_obj = unwrap_obj(cyc_obj)
            if not is_mat_struct(cyc_obj):
                continue
            cyc_fns = fieldnames(cyc_obj)
            for seg in SEGMENTS:
                if seg not in cyc_fns:
                    continue
                d = read_segment(getattr(cyc_obj, seg))
                if not all(k in d for k in ("t", "v", "q", "T")):
                    continue
                traces.append(
                    Trace(
                        cell=cell_id,
                        cyc=cyc_num,
                        segment=seg,
                        t=d["t"],
                        v=d["v"],
                        q=d["q"],
                        T=d["T"],
                        i=d.get("i", None),
                    )
                )

    print(f"[INFO] Extracted {len(traces)} traces (cell×cycle×segment).")
    return traces


# ---------------------------
# Optional: long-form dataframe saving (big)
# ---------------------------

def save_long_df(traces: List[Trace], out_dir: str) -> None:
    # Build in chunks to reduce peak memory a bit
    chunks = []
    for tr in traces:
        n = tr.t.size
        chunks.append(
            pd.DataFrame(
                {
                    "cell": tr.cell,
                    "cyc": tr.cyc,
                    "segment": tr.segment,
                    "t_s": tr.t,
                    "v_V": tr.v,
                    "q_mAh": tr.q,
                    "T_C": tr.T,
                    "i_mA": tr.i if tr.i is not None else np.full(n, np.nan),
                }
            )
        )
        # flush occasionally
        if len(chunks) >= 50:
            break

    # We won’t concatenate/safe-write huge-by-default here; instead write all in one go
    # for simplicity. If you need streaming parquet writing, we can add it.
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "cell": tr.cell,
                    "cyc": tr.cyc,
                    "segment": tr.segment,
                    "t_s": tr.t,
                    "v_V": tr.v,
                    "q_mAh": tr.q,
                    "T_C": tr.T,
                    "i_mA": tr.i if tr.i is not None else np.full(tr.t.size, np.nan),
                }
            )
            for tr in traces
        ],
        ignore_index=True,
    )

    # Prefer parquet; fallback to csv.gz if pyarrow/fastparquet missing
    parquet_path = os.path.join(out_dir, "traces_long.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[OK] Saved long traces: {parquet_path}")
        return
    except Exception as e:
        print(f"[WARN] Parquet save failed ({type(e).__name__}: {e}). Falling back to CSV.GZ...")

    csv_gz_path = os.path.join(out_dir, "traces_long.csv.gz")
    df.to_csv(csv_gz_path, index=False, compression="gzip")
    print(f"[OK] Saved long traces: {csv_gz_path}")


# ---------------------------
# Metrics directly from traces (memory-friendly)
# ---------------------------

def compute_cycle_metrics_from_traces(traces: List[Trace]) -> pd.DataFrame:
    """
    Uses C1dc traces:
      capacity_mAh = max(q)-min(q)
      Tmax_C       = max(T)
      R0_ohm       = initial ΔV/ΔI if i is available (often not in this dataset)
    """
    rows = []
    for tr in traces:
        if tr.segment != "C1dc":
            continue
        q = tr.q
        T = tr.T
        v = tr.v
        i = tr.i

        cap = float(np.nanmax(q) - np.nanmin(q)) if np.isfinite(q).any() else np.nan
        Tmax = float(np.nanmax(T)) if np.isfinite(T).any() else np.nan

        R0 = np.nan
        if i is not None and np.isfinite(i).any():
            N = min(20, v.size)
            if N >= 3:
                v0, i0 = v[0], i[0] / 1000.0  # A
                v1, i1 = v[N - 1], i[N - 1] / 1000.0
                dI = i1 - i0
                dV = v0 - v1
                if abs(dI) > 1e-12:
                    R0 = float(dV / dI)

        rows.append({"cell": tr.cell, "cyc": tr.cyc, "capacity_mAh": cap, "Tmax_C": Tmax, "R0_ohm": R0})

    df = pd.DataFrame(rows)
    return df.sort_values(["cell", "cyc"]).drop_duplicates(["cell", "cyc"], keep="first").reset_index(drop=True)


def compute_ica_features_from_traces(traces: List[Trace], dv_bin: float = 0.002) -> pd.DataFrame:
    """
    ICA from OCVdc: dQ/dV vs V (binned smoothing), then simple features.
    """
    rows = []
    for tr in traces:
        if tr.segment != "OCVdc":
            continue
        v = tr.v
        q = tr.q

        ok = np.isfinite(v) & np.isfinite(q)
        v, q = v[ok], q[ok]
        if v.size < 50:
            continue

        idx = np.argsort(v)
        v, q = v[idx], q[idx]

        vmin, vmax = float(v.min()), float(v.max())
        bins = np.arange(vmin, vmax + dv_bin, dv_bin)
        if bins.size < 10:
            continue

        digit = np.digitize(v, bins) - 1
        vb, qb = [], []
        for bi in range(bins.size - 1):
            sel = digit == bi
            if np.any(sel):
                vb.append(float(np.mean(v[sel])))
                qb.append(float(np.mean(q[sel])))

        vb = np.array(vb, dtype=float)
        qb = np.array(qb, dtype=float)
        if vb.size < 20:
            continue

        dq_dv = np.gradient(qb, vb)
        mag = np.abs(dq_dv)
        order = np.argsort(mag)[::-1]
        p1 = int(order[0])
        p2 = int(order[1]) if order.size > 1 else int(order[0])

        area_abs = float(np.trapezoid(np.abs(dq_dv), vb))

        rows.append(
            {
                "cell": tr.cell,
                "cyc": tr.cyc,
                "ica_peak1_V": float(vb[p1]),
                "ica_peak1_val": float(dq_dv[p1]),
                "ica_peak2_V": float(vb[p2]),
                "ica_peak2_val": float(dq_dv[p2]),
                "ica_area_abs": area_abs,
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values(["cell", "cyc"]).drop_duplicates(["cell", "cyc"], keep="first").reset_index(drop=True)


# ---------------------------
# Probe / structure print
# ---------------------------

def probe_structure(mat: Dict[str, Any]) -> None:
    keys = nonprivate_keys(mat)
    print("[PROBE] Non-private keys:", keys)

    # Try to find cell container and show cycles/segments quickly
    try:
        name, cells = find_cell_container(mat)
        print(f"[PROBE] Detected cell container at: {name}")
        print(f"[PROBE] Detected {len(cells)} cells")
        c1 = cells[0][1]
        cyc = iter_cycles(c1)
        print(f"[PROBE] Cell1 cycles (first 10): {[c for c,_ in cyc[:10]]}")
        if cyc:
            cyc0 = unwrap_obj(cyc[0][1])
            if is_mat_struct(cyc0):
                print(f"[PROBE] First cycle fields: {fieldnames(cyc0)}")
                for seg in SEGMENTS:
                    if seg in fieldnames(cyc0):
                        seg0 = unwrap_obj(getattr(cyc0, seg))
                        if is_mat_struct(seg0):
                            print(f"[PROBE] Segment {seg} signals: {fieldnames(seg0)}")
    except Exception as e:
        print(f"[PROBE] Could not detect container: {type(e).__name__}: {e}")


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="Path to Oxford_Battery_Degradation_Dataset_1.mat")
    ap.add_argument("--out_dir", default="out", help="Output directory")
    ap.add_argument("--no_long_df", action="store_true", help="Skip saving huge long trace table.")
    ap.add_argument("--probe", action="store_true", help="Print structure info and exit.")
    args = ap.parse_args()

    if not os.path.exists(args.mat):
        raise FileNotFoundError(args.mat)

    print("[INFO] Loading MATLAB v5 file with SciPy...")
    mat = load_mat_v5_robust(args.mat)

    if args.probe:
        probe_structure(mat)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    print("[INFO] Extracting traces...")
    traces = extract_traces(mat)
    if not traces:
        raise RuntimeError("No traces extracted. Use --probe to inspect structure.")

    if not args.no_long_df:
        print("[INFO] Saving long-form traces (can be very large)...")
        save_long_df(traces, args.out_dir)
    else:
        print("[INFO] Skipping long-form traces (--no_long_df).")

    print("[INFO] Computing cycle metrics (capacity/Tmax/R0) from C1dc...")
    metrics = compute_cycle_metrics_from_traces(traces)
    metrics_path = os.path.join(args.out_dir, "cycle_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    print(f"[OK] Saved: {metrics_path}")

    print("[INFO] Computing ICA features from OCVdc...")
    ica = compute_ica_features_from_traces(traces)
    ica_path = os.path.join(args.out_dir, "ica_features.csv")
    ica.to_csv(ica_path, index=False)
    print(f"[OK] Saved: {ica_path}")

    master = metrics.merge(ica, on=["cell", "cyc"], how="left")
    master_path = os.path.join(args.out_dir, "master_table.csv")
    master.to_csv(master_path, index=False)
    print(f"[OK] Saved: {master_path}")

    print("\n[SNAPSHOT] master_table head:")
    print(master.head(10).to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
