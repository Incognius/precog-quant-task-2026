#!/usr/bin/env bash
set -euo pipefail

# 1) Activate venv
source /home/vishalrao/AlphaGrep/alpha_venv/bin/activate

cd "/home/vishalrao/AlphaGrep/"

# 2) Paths
PY=python3
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$PROJECT_DIR/run_models/model_scripts"

DATA_DIR="$PROJECT_DIR/data/nifty_100_labeled"
OUT_LINEAR="$PROJECT_DIR/preds/linear_preds"
OUT_MLP="$PROJECT_DIR/preds/mlp_preds"
OUT_LGBM="$PROJECT_DIR/preds/lgbm_preds"

LOG_DIR="$PROJECT_DIR/run_models/logs"
LINEAR_LOG="$LOG_DIR/linear_logs.txt"
MLP_LOG="$LOG_DIR/mlp_logs.txt"
LGBM_LOG="$LOG_DIR/lgbm_logs.txt"

# 3) Create and verify all dirs
mkdir -p "$LOG_DIR" "$OUT_LINEAR" "$OUT_MLP" "$OUT_LGBM"
[ -d "$LOG_DIR" ] || { echo "Failed to create $LOG_DIR"; exit 1; }

export PYTHONUNBUFFERED=1

# 3.5) Clean old predictions
rm -f "$OUT_LINEAR"/*.csv
# rm -f "$OUT_MLP"/*.csv
# rm -f "$OUT_LGBM"/*.csv

# 4) Launch jobs (overwrite logs each run)
nohup bash -c "
  cd \"$SCRIPTS_DIR\"
  $PY train_linear.py \
    --input_dir \"$DATA_DIR\" \
    --output_dir \"$OUT_LINEAR\" \
    --train_size 45360 \
    --retrain_every 7560 \
    --feature_lag 2
" >"$LINEAR_LOG" 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=1 bash -c "
#   cd \"$SCRIPTS_DIR\"
#   $PY train_mlp.py \
#     --input_dir \"$DATA_DIR\" \
#     --output_dir \"$OUT_MLP\" \
#     --train_size 45360 \
#     --retrain_every 7560 \
#     --epochs 10 \
#     --lr 0.001 \
#     --feature_lag 2
# " >"$MLP_LOG" 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=0 bash -c "
#   cd \"$SCRIPTS_DIR\"
#   $PY train_lgbm.py \
#     --input_dir \"$DATA_DIR\" \
#     --output_dir \"$OUT_LGBM\" \
#     --train_size 45360 \
#     --retrain_every 7560 \
#     --n_estimators 100 \
#     --learning_rate 0.05 \
#     --feature_lag 2
# " >"$LGBM_LOG" 2>&1 &

disown -a
