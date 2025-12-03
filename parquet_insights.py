#!/usr/bin/env python3
"""
parquet_insights_pretty.py

Improved production-ready analyzer for .parquet files in user_data/.
Creates presentation-quality visualizations (seaborn + matplotlib),
summary.txt with "Data Summary" and image-insight blocks separated by ---.

Run:
    python parquet_insights_pretty.py
"""
from pathlib import Path
import hashlib
import warnings
import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
import statsmodels.api as sm   # optional, for nicer trend fits if available
from pandas.api.types import CategoricalDtype

# Setup
warnings.simplefilter(action="ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

ROOT = Path("user_data")
OUT_DIR = ROOT / "data_meaning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Utilities
def friendly_folder_name(parquet_path: Path) -> str:
    base = parquet_path.stem
    short_hash = hashlib.md5(str(parquet_path).encode()).hexdigest()[:8]
    return f"{base}_{short_hash}"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_save_fig(fig, path: Path, dpi=160):
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

def detect_datetime_column(df: pd.DataFrame):
    # first check real datetime dtype
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # heuristics: names containing 'date'/'time'/'ts'
    for c in df.columns:
        name = str(c).lower()
        if any(k in name for k in ("date", "time", "ts")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > 0:
                df[c] = parsed
                return c
    return None

def clean_column_dtypes_for_analysis(df: pd.DataFrame):
    """
    Convert pandas extension types safely so we can compute correlations and plot.
    - string[...] -> object
    - numeric extension types -> coerce to float via pd.to_numeric
    - leave datetime as datetime
    Returns cleaned df (copy) and lists of numeric/categorical/datetime column names.
    """
    df = df.copy()
    for c in df.columns:
        dtype_str = str(df[c].dtype)
        # convert pandas string extension to object (python str) for seaborn/matplotlib compatibility
        if "string" in dtype_str.lower() or "stringdtype" in dtype_str.lower():
            df[c] = df[c].astype(object)
        # convert pandas boolean extension to python bool/object
        if "boolean" in dtype_str.lower():
            df[c] = df[c].astype(object)
    # identify numeric columns robustly, then coerce them for numeric math
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # also include nullable integer dtypes (Int64) which may be recognized; pd.to_numeric will coerce safely
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # re-evaluate numeric columns after coercion
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [
        c for c in df.columns 
        if pd.api.types.is_object_dtype(df[c]) 
        or isinstance(df[c].dtype, CategoricalDtype)
    ]
    dt_col = detect_datetime_column(df)
    if dt_col:
        # ensure datetime column is datetime dtype
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    return df, numeric_cols, cat_cols, dt_col

def top_numeric_columns_by_variance(df: pd.DataFrame, k=3):
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        return []
    variances = df[numeric_cols].var(numeric_only=True).fillna(0)
    return variances.sort_values(ascending=False).head(k).index.tolist()

def best_categorical_column(df: pd.DataFrame, max_uniques=50):
    cats = [c for c in df.columns if (pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))]
    candidates = []
    for c in cats:
        nunique = int(df[c].nunique(dropna=True))
        if 1 < nunique <= max_uniques:
            candidates.append((nunique, c))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], -x[0]))
    return candidates[0][1]

def compute_top_correlations(df: pd.DataFrame, topk=5):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return []
    corr = df[num_cols].corr(method="pearson")
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                pairs.append((abs(val), cols[i], cols[j], val))
    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:topk]

def pearson_with_p(a, b):
    try:
        r, p = pearsonr(a, b)
        return float(r), float(p)
    except Exception:
        return None, None

# Visualization helpers
def plot_kpi_card(out_folder: Path, metrics: dict):
    """
    Create a simple KPI card style image: show key metrics as big numbers
    metrics: dict of {name: value}
    """
    fig, ax = plt.subplots(figsize=(6,2.4))
    ax.axis("off")
    # layout metrics in columns
    n = len(metrics)
    xs = np.linspace(0.05, 0.95, n)
    for (i, (k,v)) in enumerate(metrics.items()):
        ax.text(xs[i], 0.6, f"{k}", fontsize=10, ha="center", va="bottom", transform=ax.transAxes, color="#444444")
        ax.text(xs[i], 0.15, f"{v}", fontsize=20, ha="center", va="bottom", transform=ax.transAxes, weight="bold", color="#111111")
    path = out_folder / "kpi_summary.png"
    safe_save_fig(fig, path)
    desc = f"Image: {path.name}\nDescription: KPI card showing {', '.join(list(metrics.keys())[:3])}."
    return (path.name, desc)

def plot_distribution_and_box(df, col, out_folder: Path):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(11,4), gridspec_kw={'width_ratios':[3,1]})
    sns.histplot(df[col].dropna(), kde=True, ax=ax1)
    ax1.set_title(f"Distribution of {col}", fontsize=12)
    sns.boxplot(x=df[col].dropna(), ax=ax2)
    ax2.set_title("Boxplot", fontsize=12)
    img = f"{col}_dist_box.png".replace(" ", "_")
    safe_save_fig(fig, out_folder / img)
    # compute stats
    s = df[col].dropna()
    skew = float(s.skew()) if len(s)>2 else 0.0
    desc = f"Image: {img}\nDescription: Distribution and boxplot for '{col}'. n={len(s)}; mean={s.mean():.3f}, std={s.std():.3f}, skew={skew:.2f}."
    return (img, desc)

def plot_pie_or_bar_for_category(df, col, out_folder: Path):
    counts = df[col].value_counts(dropna=True)
    num_unique = len(counts)
    if num_unique == 0:
        return None
    if num_unique <= 6:
        # pie/donut
        fig, ax = plt.subplots(figsize=(6,4))
        wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=140, wedgeprops=dict(width=0.5))
        ax.set_title(f"Category split: {col}", fontsize=12)
        img = f"{col}_pie.png".replace(" ", "_")
        safe_save_fig(fig, out_folder / img)
        desc = f"Image: {img}\nDescription: Pie chart of '{col}' (top categories)."
        return (img, desc)
    else:
        # horizontal bar of top 15
        top = counts.head(15)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=top.values, y=top.index.astype(str), ax=ax)
        ax.set_title(f"Top categories in {col}", fontsize=12)
        for i, v in enumerate(top.values):
            ax.text(v + max(1, top.values.max()*0.01), i, str(int(v)), va='center', fontsize=9)
        img = f"{col}_topbar.png".replace(" ", "_")
        safe_save_fig(fig, out_folder / img)
        desc = f"Image: {img}\nDescription: Bar chart of top categories in '{col}'. Top-5: " + ", ".join([f"{x}:{v}" for x,v in zip(top.index[:5].astype(str), top.values[:5])])
        return (img, desc)

def plot_correlation_heatmap(df, numeric_cols, out_folder: Path):
    corr = df[numeric_cols].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(6, 0.6*len(numeric_cols)), max(4, 0.6*len(numeric_cols))))
    annot = True if len(numeric_cols) <= 12 else False
    sns.heatmap(corr, annot=annot, fmt=".2f", cmap="vlag", center=0, ax=ax, cbar_kws={'shrink': .8})
    ax.set_title("Correlation matrix (Pearson)", fontsize=12)
    img = "correlation_matrix.png"
    safe_save_fig(fig, out_folder / img)
    # top pairs text
    pairs = compute_top_correlations(df, topk=6)
    pair_lines = []
    for absr, a, b, rawr in pairs:
        # pvalue
        common = df[[a,b]].dropna()
        if len(common) >= 3:
            r,p = pearson_with_p(common[a], common[b])
            pair_lines.append(f"{a} vs {b}: r={rawr:.3f}, p={p:.3f}" if p is not None else f"{a} vs {b}: r={rawr:.3f}")
        else:
            pair_lines.append(f"{a} vs {b}: r={rawr:.3f}")
    desc = f"Image: {img}\nDescription: Correlation heatmap. Top pairs: " + "; ".join(pair_lines[:5])
    return (img, desc)

def plot_time_series_trend(df, dt_col, agg_col, out_folder: Path):
    ts = df[[dt_col, agg_col]].dropna().copy()
    ts[dt_col] = pd.to_datetime(ts[dt_col], errors="coerce")
    ts = ts.sort_values(dt_col)
    if ts.empty:
        return None
    ts = ts.set_index(dt_col)
    daily = ts[agg_col].resample("D").mean().ffill().fillna(0)
    if len(daily) < 3:
        return None
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(x=daily.index, y=daily.values, ax=ax)
    ax.set_title(f"Daily mean trend of {agg_col}", fontsize=12)
    ax.set_xlabel("date")
    ax.set_ylabel(f"daily_mean_{agg_col}")
    # linear fit
    x = np.arange(len(daily))
    y = daily.values
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    # annotate trend
    ax.text(0.02, 0.95, f"slope={slope:.4f}, r={r_value:.3f}, p={p_value:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="#888888"))
    img = f"timeseries_{agg_col}.png"
    safe_save_fig(fig, out_folder / img)
    desc = f"Image: {img}\nDescription: Daily mean of '{agg_col}' (based on '{dt_col}'). slope={slope:.4f}, r={r_value:.3f}, p={p_value:.3f}."
    if p_value < 0.05:
        desc += " Trend statistically significant (p < 0.05)."
    return (img, desc)

def plot_scatter_top_pair(df, a, b, out_folder: Path):
    common = df[[a,b]].dropna()
    if len(common) < 10:
        return None
    fig, ax = plt.subplots(figsize=(6,5))
    sns.scatterplot(x=common[a], y=common[b], s=40, alpha=0.6, ax=ax)
    ax.set_title(f"{a} vs {b}", fontsize=12)
    ax.set_xlabel(a); ax.set_ylabel(b)
    rval, pval = pearson_with_p(common[a], common[b])
    ax.text(0.02, 0.95, f"r={rval:.3f}, p={pval:.3f}" if pval is not None else f"r={rval:.3f}", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", fc="white", ec="#888888"))
    img = f"scatter_{a}_vs_{b}.png".replace(" ", "_")
    safe_save_fig(fig, out_folder / img)
    desc = f"Image: {img}\nDescription: Scatter plot between '{a}' and '{b}'. Pearson r={rval:.3f}" + (f", p={pval:.3f}" if pval is not None else "")
    return (img, desc)

# Core analyzer
def analyze_parquet(parquet_path: Path):
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[warn] failed to read {parquet_path}: {e}")
        return

    if df is None or df.empty:
        print(f"[info] skipping empty dataframe: {parquet_path}")
        return

    # Clean dtypes for safety (fix string[python], nullable ints, etc.)
    df_clean, numeric_cols, cat_cols, dt_col = clean_column_dtypes_for_analysis(df)

    # Normalize numeric types: ensure they are floats for math
    for c in numeric_cols:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    folder_name = friendly_folder_name(parquet_path)
    out_folder = OUT_DIR / folder_name
    ensure_dir(out_folder)

    # Metadata
    n_rows, n_cols = df_clean.shape
    dtypes = {c: str(df_clean[c].dtype) for c in df_clean.columns}
    missing = {c: int(df_clean[c].isna().sum()) for c in df_clean.columns}
    missing_pct = {c: round(100 * missing[c] / max(1, n_rows), 2) for c in df_clean.columns}

    # Build Data Summary block
    data_summary = []
    data_summary.append(f"Data Summary: {parquet_path.name}")
    data_summary.append(f"Rows: {n_rows}")
    data_summary.append(f"Columns: {n_cols}")
    data_summary.append("")
    data_summary.append("Columns (name : dtype ; missing%):")
    for c in df_clean.columns:
        data_summary.append(f" - {c} : {dtypes.get(c)} ; missing={missing[c]} ({missing_pct[c]}%)")
    data_summary.append("")

    # Prepare images and descriptions
    images_info = []

    # KPI card: show row count, column count, num numeric cols, num categorical cols
    metrics = {
        "Rows": n_rows,
        "Cols": n_cols,
        "Numeric": len(numeric_cols),
        "Categorical": len(cat_cols)
    }
    try:
        images_info.append(plot_kpi_card(out_folder, metrics))
    except Exception as e:
        print(f"[warn] KPI card failed: {e}")

    # Distribution & box for top numeric columns
    top_nums = top_numeric_columns_by_variance(df_clean, k=2)
    for col in top_nums:
        try:
            res = plot_distribution_and_box(df_clean, col, out_folder)
            if res:
                images_info.append(res)
        except Exception as e:
            print(f"[warn] distribution failed for {col}: {e}")

    # Correlation heatmap (if multiple numeric)
    if len(numeric_cols) >= 2:
        try:
            images_info.append(plot_correlation_heatmap(df_clean, numeric_cols, out_folder))
        except Exception as e:
            print(f"[warn] correlation heatmap failed: {e}")

    # Time-series trend (if datetime detected and numeric present)
    if dt_col and numeric_cols:
        agg_col = top_numeric_columns_by_variance(df_clean, k=1)
        if agg_col:
            try:
                ts_plot = plot_time_series_trend(df_clean, dt_col, agg_col[0], out_folder)
                if ts_plot:
                    images_info.append(ts_plot)
            except Exception as e:
                print(f"[warn] timeseries failed: {e}")

    # Category visualization
    cat_col = best_categorical_column(df_clean)
    if cat_col:
        try:
            cat_plot = plot_pie_or_bar_for_category(df_clean, cat_col, out_folder)
            if cat_plot:
                images_info.append(cat_plot)
        except Exception as e:
            print(f"[warn] category plot failed: {e}")

    # Scatter for top correlated pair that wasn't plotted already
    top_pairs = compute_top_correlations(df_clean, topk=6)
    plotted_pair = False
    for absr, a, b, rawr in top_pairs:
        # try to plot if both numeric and not too small sample
        if a in numeric_cols and b in numeric_cols:
            try:
                scatter = plot_scatter_top_pair(df_clean, a, b, out_folder)
                if scatter:
                    images_info.append(scatter)
                    plotted_pair = True
                    break
            except Exception as e:
                print(f"[warn] scatter failed for {a},{b}: {e}")

    # Limit to 6 images (KPI + up to 5 others)
    images_info = images_info[:6]

    # Compose summary.txt
    summary_path = out_folder / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(data_summary).strip() + "\n")
        fh.write("\n---\n")
        for img_name, desc in images_info:
            fh.write(desc.strip() + "\n")
            fh.write("\n---\n")
        # Overall insights
        overall = []
        overall.append("Overall insights:")
        overall.append(f" - Found {len(images_info)} visual artifact(s).")
        if top_pairs:
            best = top_pairs[0]
            overall.append(f" - Strongest numeric correlation: {best[1]} vs {best[2]} (|r| ≈ {best[0]:.3f}).")
        if dt_col:
            overall.append(f" - Detected datetime column: '{dt_col}' — consider aggregations for trend analysis.")
        high_missing = [c for c,p in missing_pct.items() if p > 20.0]
        if high_missing:
            overall.append(f" - Columns with >20% missing: {', '.join(high_missing)}.")
        else:
            overall.append(" - No columns with >20% missingness.")
        fh.write("\n".join(overall) + "\n")

    print(f"[ok] analyzed '{parquet_path.name}' -> folder: {out_folder} (images: {len(images_info)}, summary: summary.txt)")

def main():
    parquet_files = list(ROOT.rglob("*.parquet"))
    if not parquet_files:
        print("[info] no parquet files found under user_data/. Place .parquet files and re-run.")
        return
    for p in parquet_files:
        try:
            analyze_parquet(p)
        except Exception as e:
            print(f"[error] failed analyzing {p}: {e}")

if __name__ == "__main__":
    main()
