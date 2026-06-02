#!/bin/bash
# One-line training-status summary for the FFT-multiobject full fine-tune.
# Run ON the login pod:  monitor_fft.sh <jobid>
# Emits: job state, step/56000, latest loss + grad_norm, s/step, ETA, local
# checkpoints, and which step-tags have landed on HF. Best-effort + quiet on
# transient failures so a 10-min poll loop never dies.
set -uo pipefail
JOB="${1:?usage: monitor_fft.sh <jobid>}"
WORK="$HOME/farm-train"
CONFIG="pi05_farm_multiobject_fft"
EXP="farm_fft_multiobject_robust_${JOB}"
LOG="$WORK/train-fft-multiobj-${JOB}.out"
CK="$WORK/openpi/checkpoints/$CONFIG/$EXP"
REPO="NoahWeiss/farm_uf850_multiobject_fft_robust"
TARGET=56000

state=$(squeue -j "$JOB" -h -o "%T" 2>/dev/null | head -1)
elapsed=$(squeue -j "$JOB" -h -o "%M" 2>/dev/null | head -1)
if [ -z "$state" ]; then
  fin=$(sacct -j "$JOB" -h -o State -X 2>/dev/null | head -1 | tr -d ' ')
  state="ENDED:${fin:-unknown}"
fi

step=""; loss=""; gn=""; rate=""; eta=""
if [ -f "$LOG" ]; then
  stepline=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -aE "Step [0-9]+: .*loss=" | tail -1)
  step=$(printf '%s' "$stepline" | grep -aoE "Step [0-9]+" | grep -oE "[0-9]+" | tail -1)
  loss=$(printf '%s' "$stepline" | grep -aoE "loss=[0-9.]+" | tail -1)
  gn=$(printf '%s' "$stepline" | grep -aoE "grad_norm=[0-9.]+" | tail -1)
  # throughput from the last two timestamped 100-step log lines
  rate=$(tr '\r' '\n' < "$LOG" 2>/dev/null \
    | grep -aE "^[0-9]{2}:[0-9]{2}:[0-9]{2}.* Step [0-9]+: " | tail -2 \
    | awk '{
        split($1,t,":"); sec=t[1]*3600+t[2]*60+t[3];
        for(i=1;i<=NF;i++) if($i=="Step"){s=$(i+1); gsub(":","",s)}
        ts[NR]=sec; st[NR]=s
      } END { if(NR==2 && st[2]>st[1]){dt=ts[2]-ts[1]; if(dt<0)dt+=86400; printf "%.2f", dt/(st[2]-st[1])} }')
  if [ -n "$rate" ] && [ -n "${step:-}" ]; then
    eta=$(awk -v r="$rate" -v n="$((TARGET-step))" 'BEGIN{s=r*n; if(s<0)s=0; printf "%dh%02dm", int(s/3600), int((s%3600)/60)}')
  fi
fi

cklocal=$(ls "$CK" 2>/dev/null | grep -E "^[0-9]+$" | sort -n | tr '\n' ',' | sed 's/,$//')
hftags=$(python3 - "$REPO" <<'EOF' 2>/dev/null
import sys
try:
    from huggingface_hub import HfApi
    t=[x.name.replace("step-","") for x in HfApi().list_repo_refs(sys.argv[1]).tags]
    print(",".join(sorted(t, key=lambda x:int(x) if x.isdigit() else 0)))
except Exception:
    print("")
EOF
)

printf "state=%s elapsed=%s | step=%s/%s %s %s | rate=%ss/step eta=%s | ckpt_local=[%s] hf=[%s]\n" \
  "$state" "${elapsed:-–}" "${step:-0}" "$TARGET" "${loss:-loss=?}" "${gn:-}" \
  "${rate:-?}" "${eta:-?}" "${cklocal:-none}" "${hftags:-none}"
