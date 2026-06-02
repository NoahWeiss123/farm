# bottle-LoRA — job 187 — FINAL SUMMARY

_finished 2026-05-30T22:38:25Z_  ·  final state: **FAILED**

## sacct
```
187              FAILED   00:04:00      1:0 
187.batch        FAILED   00:04:00      1:0 
187.extern    COMPLETED   00:04:00      0:0 
187.0            FAILED   00:04:00      1:0 
```
## checkpoints (cluster persistent home)
`~/farm-train/openpi/checkpoints/pi05_farm_bottle_lora/pi05_farm_bottle_lora_187/`

saved steps: (none — check log)

## last training-log lines
```
Traceback (most recent call last):
  File "/home/nhweiss/farm-train/openpi/scripts/compute_norm_stats.py", line 117, in <module>
    tyro.cli(main)
  File "/home/nhweiss/farm-train/openpi/.venv/lib/python3.11/site-packages/tyro/_cli.py", line 229, in cli
    return run_with_args_from_cli()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nhweiss/farm-train/openpi/scripts/compute_norm_stats.py", line 98, in main
    data_loader, num_batches = create_torch_dataloader(
                               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nhweiss/farm-train/openpi/scripts/compute_norm_stats.py", line 34, in create_torch_dataloader
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nhweiss/farm-train/openpi/src/openpi/training/data_loader.py", line 140, in create_torch_dataset
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nhweiss/farm-train/openpi/.venv/lib/python3.11/site-packages/lerobot/common/datasets/lerobot_dataset.py", line 98, in __init__
    self.revision = get_safe_version(self.repo_id, self.revision)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/nhweiss/farm-train/openpi/.venv/lib/python3.11/site-packages/lerobot/common/datasets/utils.py", line 330, in get_safe_version
    raise RevisionNotFoundError(
huggingface_hub.errors.RevisionNotFoundError: Your dataset must be tagged with a codebase version.
            Assuming _version_ is the codebase_version value in the info.json, you can run this:
            ```python
            from huggingface_hub import HfApi

            hub_api = HfApi()
            hub_api.create_tag("NoahWeiss/farm_bottle_lora", tag="_version_", repo_type="dataset")
            ```
            
srun: error: slinky-0: task 0: Exited with exit code 1
```

**GPU released** — job left the SLURM queue (sbatch self-terminated).
