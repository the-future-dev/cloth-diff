## diffusion_policy (skeleton overview)

Minimal, modular policy training package (currently supervised sequence prediction; diffusion components may be added later). This README intentionally kept short – see source for details.

### Directory Layout

```
diffusion_policy/
	config/        # YAML configs (inherit via 'extends')
	data/          # Datasets (Demonstrations)
	models/        # Transformer backbone, positional encodings, normalizers
	policies/      # Low-dim, image, privileged policy wrappers
	runner/        # CLI entrypoint
	train/         # Data loader helpers & training loop
	utils/         # Config + misc utilities
	scripts/       # Example train/eval shell scripts
```

### Supported Models
- transformer-lowdim
- transformer-image
- transformer-privileged

### Basic Usage
```bash
python -m diffusion_policy.runner.cli \
	--config diffusion_policy/config/lowdim-transformer.yaml \
	--dataset_path ./data/ClothFold_vars-25_eps-200_img-128.pkl \
	--model transformer-lowdim
```

CLI flags always override file config. Use `--help` for the full list.

### Dataset Keys
- State: `ob_trajs` or `obs_trajs`
- Actions: `action_trajs`
- Images (optional): `ob_img_trajs`
- Privileged: requires both state + image keys

### Logging
Weights & Biases can be toggled via `--wandb` or `--no-wandb` (after config overhaul). Project default: `clothdiff`.

### Online Environment Evaluation (New)
During training you can periodically evaluate the policy inside the SoftGym environment (mirrors legacy pipeline):

Key flags / config entries:
- `--eval_enabled` / `--no-eval`: master switch (file config: `evaluation.enabled`)
- `--eval_interval <N>`: run evaluation every N epochs (file: `evaluation.interval`)
- `--eval_num_episodes <K>`: number of rollout episodes per evaluation
- `--eval_video`: save a GIF for the first episode of each evaluation (stored under `checkpoints/eval_videos/`)
- `--eval_gif_size`: resolution of saved GIF
- `--eval_env_img_size`: image size used when querying env frames
- `--eval_seed`: fix RNG seed for deterministic eval (omit for stochastic runs)

Config YAML example snippet:
```yaml
evaluation:
	enabled: true
	interval: 2
	num_episodes: 10
	video: true
	gif_size: 128
	env_img_size: 128
	seed: 42
```

The evaluator returns and (if WandB enabled) logs:
- `eval/normalized_performance_mean`
- `eval/normalized_performance_std`
- `eval/avg_reward`
- `eval/avg_ep_length`
and optionally `eval/video`.

To provide environment construction parameters set `cfg.env_kwargs` via your config file (e.g. `evaluation.env_kwargs`). If omitted, evaluation is skipped with a notice.

### Status
Active refactor in progress (adding checkpointing, config cleanup, validation). Expect interface changes.

### License
See repository root for licensing information.

- This scaffold compiles and runs a minimal supervised training loop.
- Port specific diffusion models/policies incrementally into models/ and policies/ while keeping the interfaces stable.
- The old code is preserved in `old_diffusion_policy/` for reference.

Key time parameters
- horizon: Training crop length (sequence length). Each batch item is a contiguous window of this length; the model predicts an action for each step in the window.
- n_obs_steps: Context length used to anchor the action slice at inference. We return actions starting at index n_obs_steps−1.
- n_action_steps: How many action steps we return per policy call. The model predicts a full horizon internally; we typically execute a short chunk and then replan.
- batch_size: Number of sequence windows per batch during training.

Typical usage pattern
1) Train with full-horizon supervision (loss over all horizon steps).
2) At inference, call policy.predict_action and execute either the first step (pure MPC) or the returned n_action_steps; then call the policy again with new observations.
