# train_linear.py (SIMPLIFIED AND CORRECTED)
import os
import re
import glob
import argparse
import csv
from collections import deque
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# --- CONSTANTS ---
FEATS = ["open", "high", "low", "close", "volume"]
REQUIRED = FEATS + ["date", "r_h"] # Now requires r_h in the input
ALL_CLASSES = [-1, 0, 1]

# --- HELPER FUNCTIONS ---

def label_from_eps(r: np.ndarray, eps: float) -> np.ndarray:
    y = np.zeros_like(r, dtype=int)
    y[r > eps] = 1
    y[r < -eps] = -1
    return y

def select_epsilon_rolling(r_window: np.ndarray, target_nonzero: float = 0.70) -> float:
    """Simplified, rolling-friendly epsilon selection using the quantile method."""
    if r_window.size < 50:
        return 0.0
    r_abs = np.abs(r_window[np.isfinite(r_window)])
    if r_abs.size == 0:
        return 0.0
    eps = np.quantile(r_abs, 1.0 - target_nonzero)
    return float(eps)

# --- UTILITIES ---

def extract_symbol(path: str) -> str:
    base = os.path.basename(path)
    sym = re.sub(r"\.csv(\.gz)?$", "", base)
    return re.sub(r"[^A-Za-z0-9_\-]+", "", sym)

def _to_float(x):
    try: return float(x)
    except (ValueError, TypeError): return None

def _safe_softmax_like(probs_map: dict) -> tuple[float, float, float]:
    p_down = float(probs_map.get(-1, 0.0))
    p_hold = float(probs_map.get(0, 0.0))
    p_up = float(probs_map.get(1, 0.0))
    s = p_down + p_hold + p_up
    if not np.isfinite(s) or s <= 0:
        return (1/3.0, 1/3.0, 1/3.0)
    return (p_down/s, p_hold/s, p_up/s)

# --- SIMPLIFIED: MAIN WALK-FORWARD FUNCTION ---

def walkforward_linear_streaming_label(
    in_path: str,
    out_path: str,
    train_size: int,
    retrain_every: int,
    feature_lag: int,
):
    """
    Walk-forward training that reads r_h from the file and performs dynamic labeling.
    """
    X_buf = deque(maxlen=train_size)
    rh_buf = deque(maxlen=train_size)

    x_queue = deque(maxlen=feature_lag + 1)
    rh_queue = deque(maxlen=feature_lag + 1)
    date_queue = deque(maxlen=feature_lag + 1)

    rows_seen = 0
    rows_pred = 0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as out_f, \
         open(in_path, "r", newline="") as in_f:
        
        out_w = csv.writer(out_f)
        out_w.writerow(["date", "p_up", "p_down", "p_hold", "y_true"])
        
        reader = csv.reader(in_f)
        header = {c.strip().lower(): i for i, c in enumerate(next(reader))}
        
        missing = [c for c in REQUIRED if c not in header]
        if missing:
            raise ValueError(f"Input file {in_path} is missing required columns: {missing}")

        col_indices = [header[c] for c in FEATS]
        i_date = header["date"]
        i_rh = header["r_h"]

        scaler = StandardScaler()
        model = LogisticRegression(solver="lbfgs", max_iter=500, multi_class="multinomial")
        retrain_counter = 0
        is_model_trained = False
        current_eps = 0.0

        for row in reader:
            rows_seen += 1
            
            try:
                date_s = row[i_date]
                x = np.array([_to_float(row[i]) for i in col_indices], dtype="float64")
                rh = _to_float(row[i_rh])
                
                if rh is None or np.isnan(x).any():
                    continue
            except (ValueError, IndexError):
                continue
            
            x_queue.append(x)
            rh_queue.append(rh)
            date_queue.append(date_s)

            if len(x_queue) <= feature_lag:
                continue

            # x_queue[0] represents the oldest feature in the queue, at timestep t-feature_lag -----> USED FOR PREDICTION
            # rh_queue[-1] is the most recent r_h at timestep t -----> USED FOR TRUE LABEL
            # rh_queue[0] is the r_h at timestep t-feature_lag -----> USED FOR TRAINING LABEL
            # date_queue[-1] is the current date at timestep t -----> USED FOR OUTPUT ROW
            x_lag, rh_now, rh_lag, date_now = x_queue[0], rh_queue[-1], rh_queue[0], date_queue[-1]
            
            # --- Prediction step ---
            if len(rh_buf) >= train_size:
                # retrain if needed
                if retrain_counter % retrain_every == 0:
                    rh_window = np.array(list(rh_buf))
                    current_eps = select_epsilon_rolling(rh_window)

                    y_tr = label_from_eps(rh_window, current_eps)
                    X_tr = np.array(list(X_buf))

                    if len(np.unique(y_tr)) >= 2:
                        X_tr_s = scaler.fit_transform(X_tr)
                        model.fit(X_tr_s, y_tr)
                        is_model_trained = True

                if is_model_trained:
                    x_lag_scaled = scaler.transform(x_lag.reshape(1, -1))
                    probs_arr = model.predict_proba(x_lag_scaled)[0]
                    probs_map = {int(c): float(p) for c, p in zip(model.classes_, probs_arr)}
                    p_down, p_hold, p_up = _safe_softmax_like(probs_map)
                else:
                    p_down, p_hold, p_up = (1/3.0, 1/3.0, 1/3.0)

                y_now = int(label_from_eps(np.array([rh_now]), current_eps)[0])
                out_w.writerow([date_now, f"{p_up:.8f}", f"{p_down:.8f}", f"{p_hold:.8f}", y_now])
                rows_pred += 1
                retrain_counter += 1

            # --- Update buffers AFTER prediction (prevents forward bias) ---
            X_buf.append(x_lag)
            rh_buf.append(rh_lag)


    print(f"  Done: {rows_seen:,} rows read, {rows_pred:,} predictions written to {out_path}", flush=True)


def process_folder(input_dir, output_dir, train_size, retrain_every, feature_lag, max_files=None):
    os.makedirs(output_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {input_dir}")
    if max_files:
        paths = paths[:max_files]

    print(f"Found {len(paths)} CSVs. Starting process...")
    for idx, path in enumerate(paths, 1):
        sym = extract_symbol(path)
        out_path = os.path.join(output_dir, f"{sym}_linear_preds.csv")
        print(f"[{idx}/{len(paths)}] Processing {sym}...")
        try:
            walkforward_linear_streaming_label(path, out_path, train_size, retrain_every, feature_lag)
        except Exception as e:
            print(f"  ERROR processing {sym}: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Linear model with dynamic rolling-window labeling (streaming version).")
    ap.add_argument("--input_dir", required=True, help="Dir with data containing r_h column.")
    ap.add_argument("--output_dir", required=True, help="Dir to write new predictions.")
    ap.add_argument("--train_size", type=int, default=45360)
    ap.add_argument("--retrain_every", type=int, default=7560)
    ap.add_argument("--feature_lag", type=int, default=2)
    ap.add_argument("--max_files", type=int, default=None)
    args = ap.parse_args()
    
    process_folder(
        args.input_dir,
        args.output_dir,
        args.train_size,
        args.retrain_every,
        args.feature_lag,
        args.max_files
    )

if __name__ == "__main__":
    main()