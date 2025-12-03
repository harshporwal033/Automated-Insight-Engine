#!/usr/bin/env python3
"""
ingest_transform.py

Scan a user_data/ folder, load supported files (CSV/TSV/Excel/Parquet/JSON/SQLite),
normalize each dataset, save normalized Parquet copies, and write a multi-block JSONL
index file with metadata and simple relations between blocks (via overlapping columns).

Run:
    python ingest_transform.py
"""
import os
import json
import hashlib
import sqlite3
import re
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

# Try to import pyarrow for nicer Parquet writing; fall back to pandas engine
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except Exception:
    HAS_PYARROW = False

DATA_DIR = Path("user_data")
NORMALIZED_DIR = DATA_DIR / "normalized_parquet"
OUTPUT_JSONL = Path("combined_blocks.jsonl")

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xls", ".xlsx", ".parquet", ".json", ".sqlite", ".db"}


def compute_schema_hash(columns: List[str]) -> str:
    """Return a stable hash signature for a schema (sorted columns)."""
    col_str = "|".join(sorted(map(str, columns)))
    return hashlib.md5(col_str.encode("utf-8")).hexdigest()


def normalize_colname(c: str) -> str:
    """Quick column name normalization for comparisons."""
    if c is None:
        return ""
    c = str(c).lower().strip()
    c = re.sub(r"[^\w]+", "_", c)  # replace non-word with underscore
    c = re.sub(r"_+", "_", c)
    return c.strip("_")


def load_sqlite_tables(path: Path) -> List[Tuple[str, pd.DataFrame]]:
    """Load all tables from an SQLite/DB file and return list of (qualified_name, df)."""
    ret = []
    conn = sqlite3.connect(str(path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            df = pd.read_sql_query(f"SELECT * FROM \"{t}\"", conn)
            ret.append((f"{path.name}::{t}", df))
    finally:
        conn.close()
    return ret


def load_file(path: Path) -> List[Tuple[str, pd.DataFrame]]:
    """Load a file at path and return list of (ds_name, DataFrame) since some files have multiple tables/sheets."""
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return []

    try:
        if ext in [".csv", ".tsv"]:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            return [(path.name, df)]

        if ext in [".xls", ".xlsx"]:
            sheets = pd.read_excel(path, sheet_name=None)
            return [(f"{path.name}::{sheet}", df) for sheet, df in sheets.items()]

        if ext == ".parquet":
            df = pd.read_parquet(path)
            return [(path.name, df)]

        if ext == ".json":
            # Attempt to read JSON lines; fallback to normal read_json
            try:
                df = pd.read_json(path, lines=True)
            except ValueError:
                df = pd.read_json(path)
            return [(path.name, df)]

        if ext in [".sqlite", ".db"]:
            return load_sqlite_tables(path)

    except Exception as e:
        print(f"[warn] failed to load {path}: {e}")
        return []


def save_parquet(df: pd.DataFrame, dest: Path):
    """Save DataFrame as Parquet; try pyarrow first, else pandas default."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if HAS_PYARROW:
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(dest))
    else:
        # pandas will choose an available engine (fastparquet or pyarrow if found)
        df.to_parquet(dest)


def summarize_numeric(df: pd.DataFrame) -> Dict[str, Dict]:
    """Return simple numeric summaries for numeric columns."""
    num = df.select_dtypes(include=[np.number])
    out = {}
    for col in num.columns:
        s = num[col]
        out[col] = {
            "count": int(s.count()),
            "mean": float(s.mean()) if s.count() else None,
            "std": float(s.std(ddof=0)) if s.count() else None,
            "min": float(s.min()) if s.count() else None,
            "max": float(s.max()) if s.count() else None,
        }
    return out


def transform_user_data_to_blocks(data_dir: Path = DATA_DIR,
                                  output_file: Path = OUTPUT_JSONL,
                                  normalized_dir: Path = NORMALIZED_DIR) -> List[Dict]:
    """
    Primary function: scan directory, load datasets, normalize, store parquet copies,
    write JSONL index and compute simple relations.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    normalized_dir.mkdir(parents=True, exist_ok=True)
    blocks: List[Dict] = []

    # 1) Load and normalize datasets
    for root, _, files in os.walk(data_dir):
        for fname in files:
            full_path = Path(root) / fname

            # Skip normalized parquet output directory if it's inside user_data
            if normalized_dir in full_path.parents:
                continue

            datasets = load_file(full_path)
            for ds_name, df in datasets:
                if df is None or df.empty:
                    # create an empty block info but skip heavy operations
                    block_id = hashlib.md5(ds_name.encode("utf-8")).hexdigest()
                    blocks.append({
                        "block_id": block_id,
                        "source": ds_name,
                        "rows": 0,
                        "columns": [],
                        "schema": [],
                        "data_preview": [],
                        "full_data_path": None,
                        "hash_signature": compute_schema_hash([]),
                        "relations": []
                    })
                    continue

                # Normalize: convert dtypes, trim column names, drop fully-empty cols
                df = df.copy()
                df = df.convert_dtypes()
                # Clean column names in-place but keep original mapping in metadata source
                orig_cols = list(df.columns)
                # Ensure column names are strings
                df.columns = [str(c) for c in df.columns]
                # Drop columns that are fully null
                df.dropna(axis=1, how="all", inplace=True)

                # Create block id and save normalized parquet
                block_id = hashlib.md5(f"{ds_name}:{len(df)}:{compute_schema_hash(df.columns)}".encode("utf-8")).hexdigest()
                parquet_path = normalized_dir / f"normalized_{block_id}.parquet"

                # Save parquet
                try:
                    save_parquet(df, parquet_path)
                except Exception as e:
                    print(f"[warn] could not write parquet for {ds_name}: {e}")
                    parquet_path = None

                # Build metadata
                col_list = list(df.columns)
                normalized_col_list = [normalize_colname(c) for c in col_list]
                numeric_summary = summarize_numeric(df)

                block = {
                    "block_id": block_id,
                    "source": ds_name,
                    "rows": int(len(df)),
                    "columns": col_list,
                    "normalized_columns": normalized_col_list,
                    "schema": col_list,
                    "data_preview": df.head(10).to_dict(orient="records"),
                    "full_data_path": str(parquet_path) if parquet_path else None,
                    "hash_signature": compute_schema_hash(col_list),
                    "column_types": {c: str(t) for c, t in df.dtypes.items()},
                    "numeric_summary": numeric_summary,
                    "relations": []  # filled in next stage
                }

                blocks.append(block)

    # 2) Simple relation detection between blocks (overlapping normalized column names)
    for i, a in enumerate(blocks):
        cols_a = set(a.get("normalized_columns", []))
        if not cols_a:
            continue
        for j, b in enumerate(blocks):
            if i == j:
                continue
            cols_b = set(b.get("normalized_columns", []))
            if not cols_b:
                continue
            intersection = cols_a & cols_b
            if not intersection:
                continue
            # Overlap ratio vs smaller dataset schema size
            overlap_ratio = len(intersection) / min(len(cols_a), len(cols_b))
            jaccard = len(intersection) / len(cols_a | cols_b)
            # heuristics threshold: if they share a decent portion of columns, record relation
            if overlap_ratio >= 0.25 or jaccard >= 0.15:
                rel = {
                    "other_block_id": b["block_id"],
                    "shared_normalized_columns": sorted(list(intersection)),
                    "overlap_ratio": overlap_ratio,
                    "jaccard": jaccard
                }
                a["relations"].append(rel)

    # 3) Write combined JSONL
    with open(output_file, "w", encoding="utf-8") as fh:
        for blk in blocks:
            fh.write(json.dumps(blk, default=str) + "\n")

    print(f"[info] processed {len(blocks)} blocks -> {output_file}")
    return blocks


def main():
    print("[info] Starting ingestion and transformation...")
    blocks = transform_user_data_to_blocks()
    # Print short summary
    for b in blocks:
        print(f" - {b['block_id'][:8]} | {b['source']} | rows={b['rows']} | cols={len(b.get('columns',[]))} | relations={len(b.get('relations',[]))}")
    print("[info] Done.")


if __name__ == "__main__":
    main()
