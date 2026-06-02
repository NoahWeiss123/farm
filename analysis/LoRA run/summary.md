# bottle-LoRA — job 188 — FINAL SUMMARY

_finished 2026-05-31T02:28:01Z_  ·  final state: **COMPLETED**

## sacct
```
188           COMPLETED   03:47:33      0:0 
188.batch     COMPLETED   03:47:33      0:0 
188.extern    COMPLETED   03:47:33      0:0 
188.0         COMPLETED   03:47:32      0:0 
```
## checkpoints (cluster persistent home)
`~/farm-train/openpi/checkpoints/pi05_farm_bottle_lora/pi05_farm_bottle_lora_188/`

saved steps: 2000 4000 6000 8000 9999 

## last training-log lines
```
Step 7800: grad_norm=0.0267, loss=0.0019, param_norm=1806.8250
Step 7900: grad_norm=0.0284, loss=0.0019, param_norm=1806.8284
Step 8000: grad_norm=0.0269, loss=0.0019, param_norm=1806.8317
Step 8100: grad_norm=0.0293, loss=0.0019, param_norm=1806.8350
Step 8200: grad_norm=0.0261, loss=0.0017, param_norm=1806.8384
Step 8300: grad_norm=0.0265, loss=0.0018, param_norm=1806.8411
Step 8400: grad_norm=0.0267, loss=0.0018, param_norm=1806.8440
Step 8500: grad_norm=0.0301, loss=0.0020, param_norm=1806.8462
Step 8600: grad_norm=0.0295, loss=0.0019, param_norm=1806.8485
Step 8700: grad_norm=0.0289, loss=0.0019, param_norm=1806.8502
Step 8800: grad_norm=0.0257, loss=0.0017, param_norm=1806.8522
Step 8900: grad_norm=0.0290, loss=0.0019, param_norm=1806.8538
Step 9000: grad_norm=0.0277, loss=0.0019, param_norm=1806.8552
Step 9100: grad_norm=0.0264, loss=0.0016, param_norm=1806.8568
Step 9200: grad_norm=0.0247, loss=0.0015, param_norm=1806.8580
Step 9300: grad_norm=0.0259, loss=0.0017, param_norm=1806.8596
Step 9400: grad_norm=0.0252, loss=0.0015, param_norm=1806.8612
Step 9500: grad_norm=0.0263, loss=0.0016, param_norm=1806.8627
Step 9600: grad_norm=0.0276, loss=0.0017, param_norm=1806.8639
Step 9700: grad_norm=0.0237, loss=0.0014, param_norm=1806.8650
Step 9800: grad_norm=0.0264, loss=0.0016, param_norm=1806.8662
Step 9900: grad_norm=0.0247, loss=0.0015, param_norm=1806.8672
>>> training finished; final checkpoints:
2000
4000
6000
8000
9999
✓ bottle-LoRA training job 188 finished — GPU released
checkpoints (cluster, persistent home): ~/farm-train/openpi/checkpoints/pi05_farm_bottle_lora/pi05_farm_bottle_lora_188/
```

**GPU released** — job left the SLURM queue (sbatch self-terminated).
