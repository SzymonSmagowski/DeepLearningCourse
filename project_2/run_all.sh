#!/usr/bin/env bash
set -uo pipefail

MODELS=(
  "Conformer.py"
  "DPT.py"
  "GRU.py"
  "HTFP.py"
  "LSTM.py"
  "RNN.py"
  "transformer.py"
)

TASKS=(
  "task1"
  "task2"
  "task3"
  "task4"
)

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
FAILURES=()

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "▶️  Running $model on $task"
    
    # Example command – adapt flags as needed
    python "$model" \
        --data_path "/content/data/speech_commands_v0.01" \
        --results_dir "results/$model/$task" \
        --batch_size 256 \
        --search \
        --task "$task" \
        >"$LOG_DIR/${model}_${task}.out" 2>&1
    
    status=$?
    if [[ $status -ne 0 ]]; then
      echo "❌ $model / $task failed with status $status"
      FAILURES+=("$model:$task")
    else
      echo "✅ finished $model / $task"
    fi
  done
done

echo
echo "==================== SUMMARY ===================="
if ((${#FAILURES[@]})); then
  printf "The following runs failed:\n"
  printf '  • %s\n' "${FAILURES[@]}"
  exit 1          # script itself exits non-zero so CI shows red, but *after* loop
else
  echo "All runs completed successfully 🎉"
fi
