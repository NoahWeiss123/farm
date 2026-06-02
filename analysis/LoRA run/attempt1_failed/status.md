# bottle-LoRA — job 187 — live status

_updated 2026-05-30T22:38:25Z_

- **SLURM:** `not in queue`  (STATE|ELAPSED|LEFT|NODE)
- **checkpoints saved:** none yet
- **latest metric:** `<training not started — setup phase>`

## recent training log
```
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
