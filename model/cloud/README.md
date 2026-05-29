# farm-cloud

An optional **Modal-hosted π0.5 policy server** — an alternative to running the
policy on the SLURM cluster (`model/cluster/serve_pi05.sbatch`). Use it when you
want a managed HTTPS endpoint instead of `kubectl port-forward` to the cluster.

- `modal/pi05_serve.py` — a Modal app that loads a π0.5 checkpoint on an A100 and
  exposes a `POST /infer` endpoint returning action chunks.

```bash
pip install modal && modal token new
modal deploy model/cloud/modal/pi05_serve.py
# → prints an https://<workspace>--farm-pi05-infer.modal.run URL
```

## Status / caveat

As written, `pi05_serve.py` serves the **stock `pi05_droid` base** checkpoint
using the DROID observation convention (`exterior_image_1_left`,
`wrist_image_left`, `joint_position`, `gripper_position`; delta actions @ 20 Hz)
— it is a working serving scaffold, **not** the fine-tuned FARM policy. To serve
the trained FARM model you must repoint it to:

- the checkpoint `NoahWeiss/farm_uf850_pi05[_lora|_gse]` (HF) instead of `pi05_droid`,
- the FARM config (`pi05_farm_uf850*`, which requires the openpi config patches
  from `model/cluster/`), and
- the LeRobot/Libero obs keys the fine-tune expects (`observation/image`,
  `observation/wrist_image`, `observation/state`, `prompt`) — matching
  `model/eval_pi05.py`'s `_make_policy_obs`.

The primary, validated serving path is the cluster job; this is a convenience
alternative kept for when a hosted endpoint is preferable.
