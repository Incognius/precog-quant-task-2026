#!/usr/bin/env python3
# backtest_alpha_wq.py — WorldQuant-style backtest (unit-gross, additive PnL, no compounding)

from __future__ import annotations
import os, glob, argparse, logging, math, time
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# ---------------------- Logging ----------------------
logger = logging.getLogger("backtest_alpha")
logger.setLevel(logging.INFO)

def setup_logging(log_file: str | None, verbose: bool):
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

# ---------------------- IO helpers ----------------------
def list_csvs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.csv")))

def asset_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _norm_label(col: pd.Series) -> pd.Series:
    out = []
    for v in col.values:
        if pd.isna(v): out.append(np.nan); continue
        if isinstance(v, (int,float)):
            iv = int(v); out.append(iv if iv in (-1,0,1) else np.nan)
        else:
            s = str(v).strip().lower()
            out.append(-1 if s in {"-1","down"} else 0 if s in {"0","hold","flat"} else 1 if s in {"1","up"} else np.nan)
    return pd.Series(out, index=col.index, dtype="float64")

def load_panel(model_dir: str,
               alpha_col: str = "alpha_dn",
               ytrue_col: str = "y_true",
               ret_source: str = "close",     # "close" | "column" | "label"
               ret_col: str = "ret") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (alpha_wide, returns_wide) outer-joined on timestamps.
    - alpha_wide: raw alpha values (already dollar-neutral from your pipeline)
    - returns_wide: forward returns to evaluate (log returns if 'close' available)
    """
    paths = list_csvs(model_dir)
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {model_dir}")

    a_frames, r_frames = [], []
    missing_prices = 0
    for p in paths:
        try:
            # Load minimally; we may add more columns depending on ret_source
            base = pd.read_csv(p)
            if "date" not in base.columns or alpha_col not in base.columns:
                logger.warning(f"Skipping (missing date/alpha) {p}")
                continue

            base["date"] = pd.to_datetime(base["date"], utc=False, errors="coerce")
            base = base.dropna(subset=["date"]).sort_values("date")
            if base.empty:
                continue

            # Alpha 
            a = base[["date", alpha_col]].dropna().set_index("date")
            sym = asset_from_path(p)
            a_frames.append(a.rename(columns={alpha_col: sym}))

            # Returns
            if ret_source == "close":
                if "close" in base.columns:
                    # forward log return: log(C_t / C_{t-1})
                    close = base[["date","close"]].dropna().set_index("date").sort_index()
                    # NEW and CORRECT
                    # This calculates the forward log return from t to t+1.
                    rets = np.log(close.shift(-1) / close)
                    rets.columns = [sym]
                    r_frames.append(rets)
                else:
                    missing_prices += 1
            elif ret_source == "column":
                if ret_col in base.columns:
                    rr = base[["date", ret_col]].dropna().set_index("date").sort_index()
                    rr.columns = [sym]
                    r_frames.append(rr)
                else:
                    logger.warning(f"{sym}: ret column '{ret_col}' not found; skipping returns for this asset.")
            elif ret_source == "label":
                if ytrue_col in base.columns:
                    y = base[["date", ytrue_col]].dropna().set_index("date").sort_index()
                    y[ytrue_col] = _norm_label(y[ytrue_col])
                    y.columns = [sym]
                    r_frames.append(y)
                else:
                    logger.warning(f"{sym}: y_true not found; skipping returns for this asset.")
            else:
                raise ValueError(f"Unknown ret_source: {ret_source}")

        except Exception as e:
            logger.warning(f"Skipping {p}: {e}")

    if not a_frames:
        raise ValueError(f"No usable alpha data in {model_dir}")

    alpha_wide = pd.concat(a_frames, axis=1, join="outer").sort_index()

    if not r_frames:
        raise ValueError(f"No usable returns built for {model_dir} (ret_source={ret_source}).")
    returns_wide = pd.concat(r_frames, axis=1, join="outer").sort_index()

    # Align both on common index (outer OK; we’ll handle missing by dropping per timestamp)
    # We deliberately do not fill returns with 0 to avoid bias.
    # Alpha missing -> treated as 0 in weights later.
    if ret_source == "close":
        if missing_prices:
            logger.info(f"Built returns from 'close' for {len(r_frames)} assets; {missing_prices} had no close.")
    return alpha_wide, returns_wide

# ---------------------- WQ-style math ----------------------
def _row_l1_normalize(row: pd.Series) -> pd.Series:
    gross = row.abs().sum()
    if not np.isfinite(gross) or gross == 0.0:
        return row * 0.0
    return row / gross

def compute_wq_returns(alpha_wide: pd.DataFrame,
                       returns_wide: pd.DataFrame,
                       execution_lag: int = 2,
                       min_assets: int = 10) -> pd.DataFrame:
    """
    WorldQuant-style engine:
      - Lag alpha by 1 to create deployed weights
      - Drop timestamps with < min_assets overlapping (non-NaN returns & nonzero weights)
      - L1-normalize weights (unit gross)
      - Portfolio return r_t = sum_i Wnorm_{t,i} * R_{t,i}  (additive, no compounding)
      - Turnover_t = 0.5 * sum_i |Wnorm_t - Wnorm_{t-1}|
    """
    # Align on the union first
    A = alpha_wide.copy()
    R = returns_wide.reindex_like(A, copy=True)

    # Lag weights
    W = A.shift(execution_lag)

    W = W.sub(W.mean(axis=1), axis=0)

    # We only trade assets that have BOTH a weight and a return at t
    # Replace NaN alphas with 0 (no position). Keep returns NaN so they can be masked out later.
    W = W.fillna(0.0)

    # Build a mask where returns exist
    valid_ret = ~R.isna()

    # Zero out weights where you don't have a return that bar
    W_eff = W.where(valid_ret, 0.0)

    # Require enough live names per timestamp (nonzero W_eff AND valid return)
    live = (W_eff != 0.0).sum(axis=1)
    keep_idx = live[live >= int(min_assets)].index
    if len(keep_idx) == 0:
        return pd.DataFrame(columns=["ret", "turnover", "gross"], index=pd.Index([], name=A.index.name))

    W_eff = W_eff.loc[keep_idx]
    R_eff = R.loc[keep_idx].fillna(0.0)  # safe: weights are zero where returns were NaN

    # Normalize (unit gross)
    Wnorm = W_eff.apply(_row_l1_normalize, axis=1)

    # Per-bar additive portfolio return
    ret = (Wnorm * R_eff).sum(axis=1)

    # Turnover on normalized weights
    Wnorm_prev = Wnorm.shift(1).fillna(0.0)
    turnover = 0.5 * (Wnorm - Wnorm_prev).abs().sum(axis=1)

    # Reference gross of pre-normalized weights (can be small if alpha magnitudes are tiny)
    gross = W_eff.abs().sum(axis=1)

    return pd.DataFrame({"ret": ret, "turnover": turnover, "gross": gross}, index=Wnorm.index)

# ---------------------- Metrics (WQ style: from per-bar series) ----------------------
def equity_additive(ser_ret: pd.Series) -> pd.Series:
    return ser_ret.cumsum()

def wq_metrics(ser_ret: pd.Series, periods_per_year: int) -> Dict[str, float]:
    mu = ser_ret.mean()
    sd = ser_ret.std(ddof=1)
    ann_mu = mu * periods_per_year
    ann_sd = sd * math.sqrt(periods_per_year) if sd > 0 else np.nan
    sharpe = ann_mu / ann_sd if (ann_sd and ann_sd > 0) else np.nan
    return {"mean": mu, "std": sd, "ann_mean": ann_mu, "ann_std": ann_sd, "sharpe": sharpe}

def drawdown_additive(ser_ret: pd.Series) -> Dict[str, float]:
    eq = equity_additive(ser_ret)
    peak = eq.cummax()
    dd = eq - peak                   # additive drawdown
    max_dd = float(dd.min()) if len(dd) else np.nan
    return {"max_drawdown": max_dd, "equity_end": float(eq.iloc[-1]) if len(eq) else np.nan}

def yoy_additive(ser_ret: pd.Series) -> pd.Series:
    df = ser_ret.to_frame("ret")
    df["year"] = df.index.year
    return df.groupby("year")["ret"].sum()   # additive per-year PnL (not product)

def calmar_like(ann_return: float, max_drawdown_abs: float) -> float:
    # If you want a Calmar-like ratio on additive equity, divide ann_return by |max_dd| of additive curve.
    if max_drawdown_abs is None or max_drawdown_abs >= 0:
        return np.nan
    return ann_return / abs(max_drawdown_abs)

# ---------------------- Model backtest wrapper ----------------------
def backtest_model(model_dir: str,
                   alpha_col: str,
                   periods_per_year: int,
                   execution_lag: int,
                   min_assets: int,
                   ret_source: str,
                   ret_col: str,
                   log_prefix: str = "") -> Dict[str, object]:
    t0 = time.time()
    logger.info(f"{log_prefix}Loading panel from {model_dir}")
    alpha_wide, returns_wide = load_panel(model_dir, alpha_col=alpha_col, ret_source=ret_source, ret_col=ret_col)

    # Sanity: live names
    # live_now = (alpha_wide.shift(execution_lag).fillna(0.0) != 0.0).sum(axis=1)
    # desc = live_now.describe()
    # logger.info(f"{log_prefix}Assets per timestamp (after lag): median={desc['50%']:.1f}, min={desc['min']:.0f}, max={desc['max']:.0f}, mean={desc['mean']:.1f}")

    logger.info(f"{log_prefix}Computing WQ returns (lag={execution_lag}, min_assets={min_assets}, source={ret_source})")
    bt = compute_wq_returns(alpha_wide, returns_wide, execution_lag=execution_lag, min_assets=min_assets)

    # Metrics from per-bar series (no compounding)
    eq = wq_metrics(bt["ret"], periods_per_year)
    dd = drawdown_additive(bt["ret"])
    ann_ret = eq["ann_mean"]
    calmar = calmar_like(ann_ret, dd["max_drawdown"])

    yoy = yoy_additive(bt["ret"])
    total_additive = float(bt["ret"].sum())

    elapsed = time.time() - t0
    logger.info(f"{log_prefix}Done in {elapsed:.2f}s | Sharpe={eq['sharpe']:.3f}, AnnMu={ann_ret:.6f}, MaxDD={dd['max_drawdown']:.6f}")

    summary = {
        "periods": int(len(bt)),
        "ann_mean": float(ann_ret),
        "ann_vol": float(eq["ann_std"]) if eq["ann_std"] == eq["ann_std"] else np.nan,
        "sharpe": float(eq["sharpe"]) if eq["sharpe"] == eq["sharpe"] else np.nan,
        "max_drawdown_add": float(dd["max_drawdown"]),
        "calmar_like": float(calmar) if calmar == calmar else np.nan,
        "avg_turnover": float(bt["turnover"].mean()),
        "ann_turnover": float(bt["turnover"].mean()) * periods_per_year,
        "total_additive_return": float(total_additive),
    }

    return {"bt": bt, "summary": summary, "yoy": yoy}

# ---------------------- CLI ----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WorldQuant-style backtest of alpha_dn (unit-gross, additive PnL).")
    p.add_argument("--preds-root", default=os.path.join(os.getcwd(), "preds"),
                   help="Root with model folders (linear_preds, mlp_preds, lgbm_preds).")
    p.add_argument("--models", nargs="+", default=["linear", "mlp", "lgbm"],
                   help="Subset to backtest: linear mlp lgbm")
    p.add_argument("--alpha-col", default="alpha_dn", help="Column name for alpha in CSVs.")
    p.add_argument("--ret-source", choices=["close","column","label"], default="close",
                   help="How to build forward returns: 'close' (log returns), 'column' (use --ret-col), or 'label' (fallback).")
    p.add_argument("--ret-col", default="ret", help="Return column name when --ret-source=column.")
    p.add_argument("--log-file", default="backtest_alpha.log", help="Log file path.")
    p.add_argument("--verbose", action="store_true", help="Verbose console logging.")
    p.add_argument("--periods-per-year", type=int, default=93750, help="Bars per year for annualization.")
    p.add_argument("--execution-lag", type=int, default=1, help="Bars between decision and realization (default 1).")
    p.add_argument("--min-assets", type=int, default=10, help="Minimum live assets required at a timestamp.")
    p.add_argument("--out-dir", default="backtest_out", help="Directory to write results.")
    return p.parse_args()

def model_dir_map(preds_root: str) -> Dict[str, str]:
    return {
        "linear": os.path.join(preds_root, "linear_preds"),
        "mlp": os.path.join(preds_root, "mlp_preds"),
        "lgbm": os.path.join(preds_root, "lgbm_preds"),
    }

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(args.log_file, args.verbose)

    dirs = model_dir_map(args.preds_root)
    to_run = [m for m in args.models if m in dirs]

    all_summaries = {}
    for m in to_run:
        mdir = dirs[m]
        if not os.path.isdir(mdir):
            logger.warning(f"Skipping {m}: directory not found {mdir}")
            continue

        logger.info(f"=== Backtesting model: {m} ===")
        try:
            res = backtest_model(
                mdir, args.alpha_col, args.periods_per_year,
                args.execution_lag, args.min_assets,
                args.ret_source, args.ret_col,
                log_prefix=f"[{m}] "
            )
        except Exception as e:
            logger.exception(f"Skipping {m} due to error: {e}")
            continue

        # Write artifacts
        res["bt"].to_csv(os.path.join(args.out_dir, f"{m}_bt_series.csv"))
        res["yoy"].to_csv(os.path.join(args.out_dir, f"{m}_yoy_returns.csv"))
        pd.Series(res["summary"]).to_csv(os.path.join(args.out_dir, f"{m}_summary.csv"))
        all_summaries[m] = res["summary"]

    if all_summaries:
        pd.DataFrame(all_summaries).T.to_csv(os.path.join(args.out_dir, "all_models_summary.csv"))
        logger.info(f"Wrote results to {args.out_dir}")

if __name__ == "__main__":
    main()
