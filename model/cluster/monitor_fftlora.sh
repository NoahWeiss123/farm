#!/bin/bash
# One-line progress summary for the 6 FFT-LoRA jobs (name "fftlora"). Latest saved
# checkpoint step per task + queue state. Run on the login pod.
CK="$HOME/farm-train/openpi/checkpoints/pi05_fftlora"
n=$(squeue -u "$USER" -h -n fftlora -o "%i" 2>/dev/null | wc -l | tr -d " ")
states=$(squeue -u "$USER" -h -n fftlora -o "%T" 2>/dev/null | sort | uniq -c | tr "\n" " ")
line="fftlora in_queue=$n [$states] |"
for t in bottle30 bottle100 bear30 duck30 hat30 bottle30s1; do
  s=$(ls "$CK/fftlora_$t" 2>/dev/null | grep -E "^[0-9]+$" | sort -n | tail -1)
  line="$line $t:${s:-.}"
done
echo "$line"
