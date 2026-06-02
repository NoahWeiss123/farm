# Flagship GSE-robust (multiobject) — job 190 — live status

_updated 2026-05-31T04:15:04Z_

- **phase:** training
- **SLURM:** `not in queue`  (STATE|ELAPSED|LEFT|NODE) · 6×H100 slinky-0
- **progress:** Step 5900/6000 steps
- **checkpoints saved:** 1000 2000 3000 4000 5000 5999 
- **latest metric:** `Step 5900: grad_norm=0.0252, loss=0.0011, param_norm=1772.2402`

## recent log
```
Step 5900: grad_norm=0.0252, loss=0.0011, param_norm=1772.2402
 6 total 2.72 (kernels 1.40, alloc 0.50, bootstrap 0.00, allgathers 0.01, topo 0.11, graphs 0.01, connections 0.66, rest 0.03)
slinky-0:34636:47955 [3] NCCL INFO ncclCommInitRankConfig comm 0x7f78681f5a60 rank 3 nranks 6 cudaDev 3 nvmlDev 4 busId 11b000 commId 0x2ce24690783c1bda - Init COMPLETE
slinky-0:34636:47955 [3] NCCL INFO Init timings - ncclCommInitRankConfig: rank 3 nranks 6 total 2.72 (kernels 1.40, alloc 0.50, bootstrap 0.00, allgathers 0.01, topo 0.11, graphs 0.01, connections 0.66, rest 0.03)
slinky-0:34636:47953 [1] NCCL INFO ncclCommInitRankConfig comm 0x7f78680ecfa0 rank 1 nranks 6 cudaDev 1 nvmlDev 1 busId 29000 commId 0x2ce24690783c1bda - Init COMPLETE
slinky-0:34636:47953 [1] NCCL INFO Init timings - ncclCommInitRankConfig: rank 1 nranks 6 total 2.72 (kernels 1.40, alloc 0.50, bootstrap 0.00, allgathers 0.02, topo 0.10, graphs 0.01, connections 0.66, rest 0.03)
slinky-0:34636:47957 [5] NCCL INFO ncclCommInitRankConfig comm 0x7f78682fdf00 rank 5 nranks 6 cudaDev 5 nvmlDev 6 busId 145000 commId 0x2ce24690783c1bda - Init COMPLETE
slinky-0:34636:47957 [5] NCCL INFO Init timings - ncclCommInitRankConfig: rank 5 nranks 6 total 2.72 (kernels 1.40, alloc 0.50, bootstrap 0.00, allgathers 0.01, topo 0.10, graphs 0.01, connections 0.66, rest 0.03)
slinky-0:34636:47956 [4] NCCL INFO ncclCommInitRankConfig comm 0x7f7868279cb0 rank 4 nranks 6 cudaDev 4 nvmlDev 5 busId 124000 commId 0x2ce24690783c1bda - Init COMPLETE
slinky-0:34636:47956 [4] NCCL INFO Init timings - ncclCommInitRankConfig: rank 4 nranks 6 total 2.72 (kernels 1.39, alloc 0.51, bootstrap 0.00, allgathers 0.01, topo 0.11, graphs 0.01, connections 0.65, rest 0.04)
slinky-0:34636:47952 [0] NCCL INFO ncclCommInitRankConfig comm 0x7f786806c8e0 rank 0 nranks 6 cudaDev 0 nvmlDev 0 busId 1b000 commId 0x2ce24690783c1bda - Init COMPLETE
slinky-0:34636:47952 [0] NCCL INFO Init timings - ncclCommInitRankConfig: rank 0 nranks 6 total 2.72 (kernels 1.39, alloc 0.51, bootstrap 0.00, allgathers 0.00, topo 0.11, graphs 0.01, connections 0.67, rest 0.03)
>>> draining checkpoint pusher…
>>> final checkpoints:
1000
2000
3000
4000
5000
5999
✓ GSE-multiobject job 190 finished — GPUs released
model: https://huggingface.co/NoahWeiss/farm_uf850_multiobject_gse_robust
```
