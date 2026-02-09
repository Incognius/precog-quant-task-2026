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

# Suppresses unnecessary warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# --- CONSTANTS ---

# Defines which columns will be used as features for training
FEATS = ["open", "high", "low", "close", "volume"]
# Defines all required columns expected in the input CSV
REQUIRED = FEATS + ["date", "r_h"]
# Defines the possible output classes for prediction (-1, 0, +1)
ALL_CLASSES = [-1, 0, 1]

# --- HELPER FUNCTIONS ---

# Converts continuous returns into discrete class labels using epsilon thresholds
def label_from_eps(r: np.ndarray, eps: float) -> np.ndarray:
    # Initializes all labels as 0 (hold)
    y = np.zeros_like(r, dtype=int)
    # Marks as +1 (up) where return exceeds epsilon
    y[r > eps] = 1
    # Marks as -1 (down) where return is below -epsilon
    y[r < -eps] = -1
    # Returns the integer label array
    return y

# Dynamically selects an epsilon threshold based on recent volatility
def select_epsilon_rolling(r_window: np.ndarray, target_nonzero: float = 0.70) -> float:
    # Returns 0 if too little data to estimate a stable epsilon
    if r_window.size < 50:
        return 0.0
    # Takes absolute values of finite returns
    r_abs = np.abs(r_window[np.isfinite(r_window)])
    # Returns 0 if no valid entries found
    if r_abs.size == 0:
        return 0.0
    # Uses quantile method to get epsilon such that ~70% of returns exceed it
    eps = np.quantile(r_abs, 1.0 - target_nonzero)
    # Returns epsilon as a float
    return float(eps)

# --- UTILITIES ---

# Extracts a clean symbol name from a given file path
def extract_symbol(path: str) -> str:
    # Gets only the file name part of the path
    base = os.path.basename(path)
    # Removes CSV extensions
    sym = re.sub(r"\.csv(\.gz)?$", "", base)
    # Removes illegal characters for a valid symbol identifier
    return re.sub(r"[^A-Za-z0-9_\-]+", "", sym)

# Safely converts any input value to float, returning None on failure
def _to_float(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

# Normalizes probabilities so that they sum to 1, returning equal weights if invalid
def _safe_softmax_like(probs_map: dict) -> tuple[float, float, float]:
    # Extracts probabilities for each class from dictionary
    p_down = float(probs_map.get(-1, 0.0))
    p_hold = float(probs_map.get(0, 0.0))
    p_up = float(probs_map.get(1, 0.0))
    # Calculates total probability mass
    s = p_down + p_hold + p_up
    # If invalid or zero, returns uniform probabilities
    if not np.isfinite(s) or s <= 0:
        return (1/3.0, 1/3.0, 1/3.0)
    # Returns normalized probabilities (down, hold, up)
    return (p_down/s, p_hold/s, p_up/s)

# --- SIMPLIFIED: MAIN WALK-FORWARD FUNCTION ---

# Performs streaming, walk-forward model training and prediction for one asset
def walkforward_linear_streaming_label(
    in_path: str,     # Input CSV path
    out_path: str,    # Output predictions CSV path
    train_size: int,  # Number of past samples kept for training
    retrain_every: int,  # Frequency (in bars) of retraining
    feature_lag: int, # Bars of lag to apply between features and label
):
    """
    Walk-forward training that reads r_h from the file and performs dynamic labeling.
    """
    # Initializes rolling training buffers for features and returns
    X_buf = deque(maxlen=train_size)
    rh_buf = deque(maxlen=train_size)

    # Initializes shorter queues for feature-lag window management
    x_queue = deque(maxlen=feature_lag + 1)
    rh_queue = deque(maxlen=feature_lag + 1)
    date_queue = deque(maxlen=feature_lag + 1)

    # Tracks total rows seen and total predictions made
    rows_seen = 0
    rows_pred = 0

    # Ensures output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Opens output file for writing and input file for reading simultaneously
    with open(out_path, "w", newline="") as out_f, \
         open(in_path, "r", newline="") as in_f:
        
        # Creates a CSV writer for the output file
        out_w = csv.writer(out_f)
        # Writes output header
        out_w.writerow(["date", "p_up", "p_down", "p_hold", "y_true"])
        
        # Creates a CSV reader for the input file
        reader = csv.reader(in_f)
        # Reads first row to construct a column name -> index mapping
        header = {c.strip().lower(): i for i, c in enumerate(next(reader))}
        
        # Verifies that all required columns are present in the header
        missing = [c for c in REQUIRED if c not in header]
        # Raises error if any required columns are missing
        if missing:
            raise ValueError(f"Input file {in_path} is missing required columns: {missing}")

        # Stores the column indices for the selected feature columns
        col_indices = [header[c] for c in FEATS]
        # Stores index of date column
        i_date = header["date"]
        # Stores index of forward return column
        i_rh = header["r_h"]

        # Initializes scaler for feature normalization
        scaler = StandardScaler()
        # Initializes logistic regression model (multinomial)
        model = LogisticRegression(solver="lbfgs", max_iter=500, multi_class="multinomial")
        # Counter for tracking retraining intervals
        retrain_counter = 0
        # Flag to track whether model has been trained yet
        is_model_trained = False
        # Stores current rolling epsilon threshold
        current_eps = 0.0

        # Iterates over each line of the CSV after the header
        for row in reader:
            # Increments total row counter
            rows_seen += 1
            
            try:
                # Reads date string for this row
                date_s = row[i_date]
                # Reads feature values and converts them to floats
                x = np.array([_to_float(row[i]) for i in col_indices], dtype="float64")
                # Reads forward return r_h for this row
                rh = _to_float(row[i_rh])
                # Skips row if any feature or return value is invalid
                if rh is None or np.isnan(x).any():
                    continue
            # Catches index or conversion errors and skips the row
            except (ValueError, IndexError):
                continue
            
            # Appends current feature, return, and date into their queues
            x_queue.append(x)
            rh_queue.append(rh)
            date_queue.append(date_s)

            # Waits until the queue is long enough to apply the requested feature lag
            if len(x_queue) <= feature_lag:
                continue

            # Extracts lagged feature and corresponding historical values for training/prediction
            # x_queue[0] = oldest feature → used as predictor (t - feature_lag)
            # rh_queue[-1] = most recent forward return → true label for evaluation
            # rh_queue[0] = oldest forward return → used in training (t - feature_lag)
            # date_queue[-1] = current timestamp → assigned to output row
            x_lag, rh_now, rh_lag, date_now = x_queue[0], rh_queue[-1], rh_queue[0], date_queue[-1]
            
            # Ensures enough data accumulated before starting training
            if len(rh_buf) >= train_size:
                # Checks if it’s time to retrain based on retrain frequency
                if retrain_counter % retrain_every == 0:
                    # Converts training buffer of returns to numpy array
                    rh_window = np.array(list(rh_buf))
                    # Computes rolling epsilon from recent return window
                    current_eps = select_epsilon_rolling(rh_window)
                    
                    # Converts continuous returns into discrete labels using epsilon
                    y_tr = label_from_eps(rh_window, current_eps)
                    # Converts feature buffer into numpy array
                    X_tr = np.array(list(X_buf))
                    
                    # Retrains only if multiple classes exist in window
                    if len(np.unique(y_tr)) >= 2:
                        # Normalizes features
                        X_tr_s = scaler.fit_transform(X_tr)
                        # Fits logistic regression model to current window
                        model.fit(X_tr_s, y_tr)
                        # Flags that model is ready for use
                        is_model_trained = True

                # If model has been trained, perform prediction on current lagged feature
                if is_model_trained:
                    # Normalizes single lagged feature vector
                    x_lag_scaled = scaler.transform(x_lag.reshape(1, -1))
                    # Generates class probabilities for the feature
                    probs_arr = model.predict_proba(x_lag_scaled)[0]
                    # Maps probabilities to their corresponding class labels
                    probs_map = {int(c): float(p) for c, p in zip(model.classes_, probs_arr)}
                    # Normalizes to safe probability tuple
                    p_down, p_hold, p_up = _safe_softmax_like(probs_map)
                # If model not trained yet, assigns uniform random probabilities
                else:
                    p_down, p_hold, p_up = (1/3.0, 1/3.0, 1/3.0)

                # Converts current forward return into discrete class label for evaluation
                y_now = int(label_from_eps(np.array([rh_now]), current_eps)[0])
                
                # Writes date, prediction probabilities, and label to output CSV
                out_w.writerow([date_now, f"{p_up:.8f}", f"{p_down:.8f}", f"{p_hold:.8f}", y_now])
                # Increments prediction counter
                rows_pred += 1
                # Increments retrain step counter
                retrain_counter += 1

            # Adds the lagged feature and corresponding lagged return to the training buffers
            X_buf.append(x_lag)
            rh_buf.append(rh_lag)

    # Prints a summary after completing the file
    print(f"  Done: {rows_seen:,} rows read, {rows_pred:,} predictions written to {out_path}", flush=True)

# Processes all CSVs in a folder by applying the walk-forward pipeline to each
def process_folder(input_dir, output_dir, train_size, retrain_every, feature_lag, max_files=None):
    # Ensures output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Lists all CSV files in the input directory
    paths = sorted(glob.glob(os.path.join(input_dir, "*.csv")))
    # Raises error if no files found
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {input_dir}")
    # Optionally limits number of files processed
    if max_files:
        paths = paths[:max_files]

    # Prints total files to be processed
    print(f"Found {len(paths)} CSVs. Starting process...")
    # Iterates over each input file
    for idx, path in enumerate(paths, 1):
        # Extracts clean symbol name from filename
        sym = extract_symbol(path)
        # Builds corresponding output file path
        out_path = os.path.join(output_dir, f"{sym}_linear_preds.csv")
        # Logs current file progress
        print(f"[{idx}/{len(paths)}] Processing {sym}...")
        try:
            # Runs walk-forward pipeline on the current file
            walkforward_linear_streaming_label(path, out_path, train_size, retrain_every, feature_lag)
        # Catches and logs any processing error without halting loop
        except Exception as e:
            print(f"  ERROR processing {sym}: {e}", flush=True)

# Main function to parse CLI arguments and start processing
def main():
    # Creates argument parser for CLI options
    ap = argparse.ArgumentParser(description="Linear model with dynamic rolling-window labeling (streaming version).")
    # Adds required argument for input directory path
    ap.add_argument("--input_dir", required=True, help="Dir with data containing r_h column.")
    # Adds required argument for output directory path
    ap.add_argument("--output_dir", required=True, help="Dir to write new predictions.")
    # Adds optional training buffer size parameter
    ap.add_argument("--train_size", type=int, default=45360)
    # Adds optional retraining frequency parameter
    ap.add_argument("--retrain_every", type=int, default=7560)
    # Adds optional feature lag parameter
    ap.add_argument("--feature_lag", type=int, default=2)
    # Adds optional file count limit
    ap.add_argument("--max_files", type=int, default=None)
    # Parses command line arguments
    args = ap.parse_args()
    
    # Calls folder processing function with parsed arguments
    process_folder(
        args.input_dir,
        args.output_dir,
        args.train_size,
        args.retrain_every,
        args.feature_lag,
        args.max_files
    )

# Ensures script runs only when executed directly (not imported)
if __name__ == "__main__":
    main()
