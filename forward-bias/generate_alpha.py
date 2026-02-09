#!/usr/bin/env python3
# alpha_pipeline.py

from __future__ import annotations
import os
import math
import glob
import argparse
import logging
from logging import handlers
from typing import Dict, Tuple, List
import pandas as pd
import time

# ---------- Configuration Defaults (overridable via CLI) ----------
DEFAULT_PREDS_ROOT = os.path.join(os.getcwd(), "preds")
DEFAULT_MODEL_DIRS = {
    "linear": os.path.join(DEFAULT_PREDS_ROOT, "linear_preds"),
    "mlp":    os.path.join(DEFAULT_PREDS_ROOT, "mlp_preds"),
    "lgbm":   os.path.join(DEFAULT_PREDS_ROOT, "lgbm_preds"),
}
DEFAULT_ROLL_WINDOW = 390
DEFAULT_MIN_PERIODS = max(5, math.ceil(DEFAULT_ROLL_WINDOW * 0.2))
DEFAULT_HIT_RATIO_PRIOR = 0.334
DEFAULT_BACKUP_ORIGINAL = False
DEFAULT_OVERWRITE_IN_PLACE = True

REQUIRED_COLS = ["date", "p_up", "p_down", "p_hold", "y_true"]
NEW_COLS = ["pred_class", "hit", "hit_ratio", "raw_score", "weighted_score", "alpha_dn"]

# ---------- Logging ----------
LOGGER_NAME = "alpha_pipeline"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

def setup_logging(log_file: str | None, verbose: bool) -> None:
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

# ---------- IO Utilities ----------
def list_csvs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.csv")))

def infer_asset_name_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def ensure_required_columns(df: pd.DataFrame, path: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: Missing required columns: {missing}")

def to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df.set_index("date")

def map_pred_label_row(row: pd.Series) -> int:
    """
    Deterministic, neutral tie-break:
      - compute argmax over [p_up, p_down, p_hold]
      - if ties, prefer HOLD (0), then UP (+1), then DOWN (-1)
    """
    import numpy as np

    probs  = np.array([row["p_up"], row["p_down"], row["p_hold"]], dtype=float)
    labels = np.array([ 1,-1,0])  # indices: 0=up,1=down,2=hold

    # Indices sorted by preferred priority for ties: HOLD(2) -> UP(0) -> DOWN(1)
    priority_order = np.array([2, 0, 1])
    # Choose the index among priority_order that has the global max probability
    # (ties resolved by the order above)
    chosen_idx = priority_order[np.argmax(probs[priority_order])]
    return int(labels[chosen_idx])

def map_true_label_scalar(y_true) -> int:
    if isinstance(y_true, (int, float)):
        v = int(y_true)
        if v in (-1, 0, 1):
            return v
    if isinstance(y_true, str):
        y = y_true.strip().lower()
        if y in {"-1", "down"}:
            return -1
        if y in {"0", "hold", "flat"}:
            return 0
        if y in {"1", "up"}:
            return 1
    raise ValueError(f"Unrecognized y_true value: {y_true!r}")

def compute_leak_free_hit_ratio(pred_df: pd.DataFrame, roll_window: int, min_periods: int, prior: float) -> pd.Series:
    """
    Compute a *leak-free* reliability (hit) ratio.
    We shift by 1 so that the hit ratio at time t uses information only up to t−1.
    """
    hit = (pred_df["pred_class"] == pred_df["y_true"]).astype(float)

    # Rolling mean over the past window, EXCLUDING the current bar
    h = hit.rolling(window=roll_window, min_periods=min_periods).mean().shift(1)

    # Replace early NaNs (before enough history) with a neutral prior
    h = h.fillna(prior)
    return h

def safe_write_csv(df: pd.DataFrame, path: str, backup_once: bool = True, overwrite: bool = True):
    if overwrite:
        if backup_once:
            bak_path = path + ".bak"
            if not os.path.exists(bak_path):
                try:
                    df_orig = pd.read_csv(path)
                    df_orig.to_csv(bak_path, index=False)
                except Exception as e:
                    logger.warning(f"Could not create backup for {path}: {e}")
        df.to_csv(path, index=False)

# ---------- Core Pipeline ----------
def load_model_predictions(model_dir: str) -> Dict[str, pd.DataFrame]:
    paths = list_csvs(model_dir)
    if not paths:
        logger.warning(f"No CSVs found in {model_dir}")
        return {}

    data: Dict[str, pd.DataFrame] = {}
    total = len(paths)
    for i, p in enumerate(paths, 1):
        try:
            df = pd.read_csv(p)
            ensure_required_columns(df, p)
            df = to_datetime_index(df)
            df["y_true"] = df["y_true"].apply(map_true_label_scalar)
            df["pred_class"] = df.apply(map_pred_label_row, axis=1)
            df["hit"] = (df["pred_class"] == df["y_true"]).astype(int)
            df["raw_score"] = df["p_up"] - df["p_down"]
            asset = infer_asset_name_from_path(p)
            data[asset] = df

            if i == total or i % 10 == 0:
                logger.info(f"Loaded {i}/{total}: {asset}")
        except Exception as e:
            logger.error(f"Skipping {p} due to error: {e}")

    return data

def compute_per_asset_weights(
    data: Dict[str, pd.DataFrame],
    roll_window: int,
    min_periods: int,
    prior: float
) -> Dict[str, pd.Series]:
    h: Dict[str, pd.Series] = {}
    for idx, (asset, df) in enumerate(data.items(), 1):
        h[asset] = compute_leak_free_hit_ratio(df, roll_window, min_periods, prior)
        if idx == len(data) or idx % 20 == 0:
            logger.info(f"Computed rolling weights for {idx}/{len(data)} assets")
    return h

def cross_sectional_alpha_dn(
    data: Dict[str, pd.DataFrame],
    weights: Dict[str, pd.Series],
) -> Dict[str, pd.Series]:
    weighted_scores: Dict[str, pd.Series] = {}
    for asset, df in data.items():
        w = weights[asset].reindex(df.index)
        ws = w * df["raw_score"]
        weighted_scores[asset] = ws

    R_df = pd.DataFrame(weighted_scores)
    cs_mean = R_df.mean(axis=1, skipna=True)

    alpha_dn: Dict[str, pd.Series] = {}
    for asset, ws in weighted_scores.items():
        a = ws - cs_mean.reindex(ws.index)
        alpha_dn[asset] = a
    return alpha_dn

def write_back_model(
    model_dir: str,
    data: Dict[str, pd.DataFrame],
    weights: Dict[str, pd.Series],
    alpha_dn: Dict[str, pd.Series],
    backup_original: bool = DEFAULT_BACKUP_ORIGINAL,
    overwrite_in_place: bool = DEFAULT_OVERWRITE_IN_PLACE,
):
    path_map = {infer_asset_name_from_path(p): p for p in list_csvs(model_dir)}
    total = len(data)
    for i, (asset, df) in enumerate(data.items(), 1):
        try:
            h = weights[asset].reindex(df.index)
            a = alpha_dn[asset].reindex(df.index)
            ws = (h * df["raw_score"]).reindex(df.index)

            out = df.copy()
            out["hit_ratio"] = h
            out["weighted_score"] = ws
            out["alpha_dn"] = a
            out = out.reset_index()

            path = path_map[asset]
            safe_write_csv(out, path, backup_once=backup_original, overwrite=overwrite_in_place)

            if i == total or i % 10 == 0:
                logger.info(f"Wrote {i}/{total}: {asset}")
        except Exception as e:
            logger.error(f"Could not write back for {asset}: {e}")

def run_pipeline_for_model(
    model_key: str,
    model_dirs: Dict[str, str],
    roll_window: int,
    min_periods: int,
    prior: float,
    backup_original: bool,
    overwrite_in_place: bool,
):
    if model_key not in model_dirs:
        raise ValueError(f"Unknown model_key: {model_key}")

    model_dir = model_dirs[model_key]
    logger.info(f"=== Processing model: {model_key} @ {model_dir} ===")
    t0 = time.time()

    data = load_model_predictions(model_dir)
    if not data:
        logger.warning(f"Skipping {model_key} (no data).")
        return

    weights = compute_per_asset_weights(data, roll_window, min_periods, prior)
    alpha_dn = cross_sectional_alpha_dn(data, weights)
    write_back_model(model_dir, data, weights, alpha_dn, backup_original, overwrite_in_place)

    elapsed = time.time() - t0
    logger.info(f"[DONE] Model {model_key}: {len(data)} files updated in {elapsed:.2f}s")

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reliability-weighted alpha pipeline (per model class).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["linear", "mlp", "lgbm"],
        help="Model keys to process (subset of: linear mlp lgbm).",
    )
    parser.add_argument("--preds-root", default=DEFAULT_PREDS_ROOT, help="Root folder containing model prediction folders.")
    parser.add_argument("--linear-dir", default=None, help="Override path for linear model CSVs.")
    parser.add_argument("--mlp-dir", default=None, help="Override path for mlp model CSVs.")
    parser.add_argument("--lgbm-dir", default=None, help="Override path for lgbm model CSVs.")

    parser.add_argument("--roll-window", type=int, default=DEFAULT_ROLL_WINDOW, help="Rolling window (minutes) for hit ratio.")
    parser.add_argument("--min-periods", type=int, default=DEFAULT_MIN_PERIODS, help="Min periods for rolling mean.")
    parser.add_argument("--prior", type=float, default=DEFAULT_HIT_RATIO_PRIOR, help="Cold-start prior for hit ratio.")

    parser.add_argument("--no-backup", action="store_true", help="Disable .bak creation before first overwrite.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite CSVs (dry run).")

    parser.add_argument("--log-file", default="alpha_pipeline.log", help="Path to log file (will append).")
    parser.add_argument("--verbose", action="store_true", help="Verbose console output.")

    return parser.parse_args()

def resolve_model_dirs(args: argparse.Namespace) -> Dict[str, str]:
    model_dirs = {
        "linear": args.linear_dir or os.path.join(args.preds_root, "linear_preds"),
        "mlp":    args.mlp_dir    or os.path.join(args.preds_root, "mlp_preds"),
        "lgbm":   args.lgbm_dir   or os.path.join(args.preds_root, "lgbm_preds"),
    }
    return model_dirs

def main():
    args = parse_args()
    setup_logging(args.log_file, args.verbose)
    logger.info("Starting reliability-weighted alpha pipeline")

    model_dirs = resolve_model_dirs(args)
    logger.info(f"Model directories: {model_dirs}")

    backup_original = not args.no_backup
    overwrite_in_place = not args.no_overwrite

    # Sanity: show effective params
    logger.info(
        f"Params | roll_window={args.roll_window} "
        f"min_periods={args.min_periods} prior={args.prior} "
        f"backup_original={backup_original} overwrite_in_place={overwrite_in_place}"
    )

    for mk in args.models:
        try:
            run_pipeline_for_model(
                mk,
                model_dirs=model_dirs,
                roll_window=args.roll_window,
                min_periods=args.min_periods,
                prior=args.prior,
                backup_original=backup_original,
                overwrite_in_place=overwrite_in_place,
            )
        except Exception as e:
            logger.exception(f"Fatal error processing model {mk}: {e}")

    logger.info("All requested models processed. Exiting.")

if __name__ == "__main__":
    main()
