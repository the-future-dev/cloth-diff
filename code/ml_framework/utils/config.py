from __future__ import annotations
import argparse
import json
import os
import yaml  # type: ignore
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class TrainConfig:
    # high-level
    mode: str = "train"  # or "eval"
    seed: int = 1234
    device: str = "auto"

    # data
    dataset_path: Optional[str] = None
    batch_size: int = 256
    num_workers: int = 8
    val_ratio: float = 0.05  # portion of dataset used for validation

    # policy
    model: str = "transformer-lowdim"  # transformer-image, transformer-privileged also supported
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8
    num_inference_steps: int = 100  # (currently unused; reserved for future sampling logic)
    lowdim_weight: float = 1.0      # used by privileged policy (loss weighting)

    # environment / data expectations
    env_img_size: Optional[int] = None  # optional image size guard (square) or tuple later

    # optimizer
    lr: float = 1e-4
    beta1: float = 0.95
    beta2: float = 0.999
    weight_decay: float = 0.0
    max_epochs: int = 50

    # checkpointing
    checkpoint_dir: Optional[str] = None
    checkpoint_every: int = 1
    keep_last: int = 5
    resume: bool = False
    resume_path: Optional[str] = None

    # logging
    wandb: bool = True
    wandb_project: str = "clothdiff"
    exp_name: Optional[str] = None
    no_wandb: bool = False  # CLI convenience flag to force disable

    # evaluation (online during training)
    eval_enabled: bool = True          # master switch
    eval_interval: int = 1             # epochs between eval runs
    eval_num_episodes: int = 10        # episodes per evaluation call
    eval_video: bool = False           # save a gif for first episode of each eval
    eval_gif_size: int = 128           # resolution for saved gif
    eval_env_img_size: int = 128       # size used when querying env.get_image
    eval_max_episode_steps: Optional[int] = None  # if provided, overrides max steps for eval rollout
    eval_seed: Optional[int] = None    # fixed seed for deterministic eval; None -> leave RNG

    # clothfold convenience (auto-resolves dataset_path and env_kwargs if both provided)
    eps: Optional[int] = None          # number of episodes 
    vars: Optional[int] = None         # number of variations

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    # config file (YAML or JSON)
    p.add_argument("--config", type=str, default=None, help="Path to YAML/JSON config file.")
    # mode
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", type=str, default="auto")
    # data
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--val_ratio", type=float, default=0.05)
    # policy
    p.add_argument("--model", type=str, default="transformer-lowdim",
                   choices=["transformer-lowdim","transformer-image","transformer-privileged"]) 
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--n_obs_steps", type=int, default=2)
    p.add_argument("--n_action_steps", type=int, default=8)
    p.add_argument("--num_inference_steps", type=int, default=100)
    p.add_argument("--lowdim_weight", type=float, default=1.0)
    p.add_argument("--env_img_size", type=int, default=None)
    # optim
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.95)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_epochs", type=int, default=50)
    # logging
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging (overrides file config)")
    p.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging (overrides anything else)")
    p.add_argument("--wandb_project", type=str, default="clothdiff")
    p.add_argument("--exp_name", type=str, default=None)

    # evaluation
    p.add_argument("--eval_enabled", action="store_true", help="Enable online evaluation during training")
    p.add_argument("--no-eval", action="store_true", help="Disable online evaluation (overrides file config)")
    p.add_argument("--eval_interval", type=int, default=1)
    p.add_argument("--eval_num_episodes", type=int, default=10)
    p.add_argument("--eval_video", action="store_true", help="Record a gif for first episode each eval")
    p.add_argument("--eval_gif_size", type=int, default=128)
    p.add_argument("--eval_env_img_size", type=int, default=128)
    p.add_argument("--eval_max_episode_steps", type=int, default=None)
    p.add_argument("--eval_seed", type=int, default=None)

    # clothfold dataset/environment convenience args
    p.add_argument("--eps", type=int, default=None, help="Number of episodes (auto-resolves dataset path and env num_variations)")
    p.add_argument("--vars", type=int, default=None, help="Number of variations (auto-resolves dataset path and env num_variations)")

    # checkpointing
    p.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to store checkpoints (default: ./checkpoints under CWD)")
    p.add_argument("--checkpoint_every", type=int, default=1, help="Epoch frequency for checkpointing")
    p.add_argument("--keep_last", type=int, default=5, help="Number of recent checkpoints to retain (rolling)")
    p.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in --resume_path or derived folder")
    p.add_argument("--resume_path", type=str, default=None, help="Explicit checkpoint path (.pth) to load")
    return p


def parse_config(argv=None) -> TrainConfig:
    p = build_argparser()
    args = p.parse_args(argv)

    # Start with dataclass defaults
    cfg = TrainConfig()

    # 1) Load from config file if provided
    if getattr(args, "config", None):
        file_cfg = _load_config_tree(args.config)
        # Map nested config sections to TrainConfig fields
        _apply_file_config(cfg, file_cfg)

    # 2) Overlay CLI args (explicitly provided values win)
    # Determine parser defaults to check if a value was explicitly set
    parser_defaults = {a.dest: a.default for a in p._actions if hasattr(a, "dest")}

    # Build a dict of desired overrides
    overrides: Dict[str, Any] = {}
    # direct name alignment between argparse and dataclass fields
    direct_keys = {
        "mode", "seed", "device",
        "dataset_path", "batch_size", "num_workers", "val_ratio",
        "model", "horizon", "n_obs_steps", "n_action_steps", "num_inference_steps",
        "lowdim_weight", "env_img_size",
        "lr", "beta1", "beta2", "weight_decay", "max_epochs",
        "wandb", "wandb_project", "exp_name",
        "checkpoint_dir", "checkpoint_every", "keep_last", "resume", "resume_path",
        "eps", "vars"
    }
    # evaluation related keys (store_true handled separately below)
    eval_keys_numeric = [
        "eval_interval", "eval_num_episodes", "eval_gif_size", "eval_env_img_size", "eval_max_episode_steps"
    ]
    for k in eval_keys_numeric:
        direct_keys.add(k)
    # add eval_seed which may be None or int
    direct_keys.add("eval_seed")

    for k in direct_keys:
        if not hasattr(args, k):
            continue
        v = getattr(args, k)
        # Special handling for store_true flags: only override if True
        if k == "wandb":
            if bool(v):  # only set True if user passed --wandb
                overrides[k] = True
            continue
        if k == "resume":
            if bool(v):
                overrides[k] = True
            continue
        # For others, override if different from parser default (assumes user passed it)
        if k in parser_defaults and v != parser_defaults[k]:
            overrides[k] = v

    # Handle explicit --no-wandb (takes precedence)
    if getattr(args, "no_wandb", False):
        overrides["wandb"] = False
    # evaluation flags precedence
    if getattr(args, "no_eval", False):
        overrides["eval_enabled"] = False
    elif getattr(args, "eval_enabled", False):
        overrides["eval_enabled"] = True
    if getattr(args, "eval_video", False):
        overrides["eval_video"] = True

    # Apply overrides
    for k, v in overrides.items():
        setattr(cfg, k, v)

    # Auto-resolve ClothFold dataset path and env_kwargs if --eps and --vars provided
    if cfg.eps is not None and cfg.vars is not None:
        _auto_resolve_clothfold_config(cfg)

    return cfg


def pretty_config(cfg: TrainConfig) -> str:
    return json.dumps(cfg.to_dict(), indent=2, sort_keys=True)


# ---------------
# Helpers for file configs
# ---------------

def _read_file_as_dict(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        text = f.read()
    if ext in (".yml", ".yaml"):
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML config must be a mapping at top-level: {path}")
        return data
    elif ext == ".json":
        data = json.loads(text or "{}")
        if not isinstance(data, dict):
            raise ValueError(f"JSON config must be a mapping at top-level: {path}")
        return data
    else:
        raise ValueError(f"Unsupported config extension '{ext}'. Use .yaml/.yml or .json")


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_config_tree(path: str) -> Dict[str, Any]:
    path = os.path.abspath(path)
    data = _read_file_as_dict(path)
    base_key = data.get("extends")
    if isinstance(base_key, str):
        # Resolve extends relative to the current file directory if relative
        base_path = base_key
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(path), base_path)
        base_cfg = _load_config_tree(base_path)
        data = _deep_merge(base_cfg, {k: v for k, v in data.items() if k != "extends"})
    return data


def _apply_file_config(cfg: TrainConfig, d: Dict[str, Any]) -> None:
    # Map file sections into TrainConfig fields
    # wandb
    w = d.get("wandb", {}) or {}
    if "enabled" in w:
        cfg.wandb = bool(w.get("enabled"))
    if "project" in w:
        cfg.wandb_project = str(w.get("project"))
    if "name" in w:
        cfg.exp_name = w.get("name") if w.get("name") is None else str(w.get("name"))

    # optionizer
    o = d.get("optionizer", {}) or {}
    if "mode" in o:
        cfg.mode = str(o.get("mode"))
    if "seed" in o:
        cfg.seed = int(o.get("seed"))
    if "device" in o:
        cfg.device = str(o.get("device"))

    # data
    data = d.get("data", {}) or {}
    if "dataset_path" in data:
        cfg.dataset_path = data.get("dataset_path")
    if "batch_size" in data:
        cfg.batch_size = int(data.get("batch_size"))
    if "num_workers" in data:
        cfg.num_workers = int(data.get("num_workers"))
    if "val_ratio" in data:
        cfg.val_ratio = float(data.get("val_ratio"))
    if "env_img_size" in data:
        cfg.env_img_size = int(data.get("env_img_size")) if data.get("env_img_size") is not None else None

    # model
    m = d.get("model", {}) or {}
    if "name" in m:
        cfg.model = str(m.get("name"))
    if "horizon" in m:
        cfg.horizon = int(m.get("horizon"))
    if "n_obs_steps" in m:
        cfg.n_obs_steps = int(m.get("n_obs_steps"))
    if "n_action_steps" in m:
        cfg.n_action_steps = int(m.get("n_action_steps"))
    if "num_inference_steps" in m:
        cfg.num_inference_steps = int(m.get("num_inference_steps"))
    if "lowdim_weight" in m:
        cfg.lowdim_weight = float(m.get("lowdim_weight"))

    # optimizer
    opt = d.get("optim", {}) or {}
    if "lr" in opt:
        cfg.lr = float(opt.get("lr"))
    if "beta1" in opt:
        cfg.beta1 = float(opt.get("beta1"))
    if "beta2" in opt:
        cfg.beta2 = float(opt.get("beta2"))
    if "max_epochs" in opt:
        cfg.max_epochs = int(opt.get("max_epochs"))
    if "weight_decay" in opt:
        cfg.weight_decay = float(opt.get("weight_decay"))

    # checkpointing section (optional)
    ck = d.get("checkpoint", {}) or {}
    if "dir" in ck:
        cfg.checkpoint_dir = str(ck.get("dir")) if ck.get("dir") else None
    if "every" in ck:
        cfg.checkpoint_every = int(ck.get("every"))
    if "keep_last" in ck:
        cfg.keep_last = int(ck.get("keep_last"))
    if "resume" in ck:
        cfg.resume = bool(ck.get("resume"))
    if "resume_path" in ck:
        cfg.resume_path = ck.get("resume_path") if ck.get("resume_path") else None

    # evaluation section
    ev = d.get("evaluation", {}) or {}
    if "enabled" in ev:
        cfg.eval_enabled = bool(ev.get("enabled"))
    if "interval" in ev:
        cfg.eval_interval = int(ev.get("interval"))
    if "num_episodes" in ev:
        cfg.eval_num_episodes = int(ev.get("num_episodes"))
    if "video" in ev:
        cfg.eval_video = bool(ev.get("video"))
    if "gif_size" in ev:
        cfg.eval_gif_size = int(ev.get("gif_size"))
    if "env_img_size" in ev:
        cfg.eval_env_img_size = int(ev.get("env_img_size"))
    if "max_episode_steps" in ev:
        cfg.eval_max_episode_steps = int(ev.get("max_episode_steps")) if ev.get("max_episode_steps") is not None else None
    if "seed" in ev:
        tmp = ev.get("seed")
        cfg.eval_seed = int(tmp) if tmp is not None else None
    # pass through raw env kwargs if provided for evaluator
    if "env_kwargs" in ev and isinstance(ev.get("env_kwargs"), dict):
        # attach dynamically (not a dataclass field)
        setattr(cfg, 'env_kwargs', ev.get('env_kwargs'))


def _auto_resolve_clothfold_config(cfg: TrainConfig) -> None:
    """Auto-resolve dataset path and env_kwargs based on eps/vars if both provided."""
    import os
    
    # Auto-resolve dataset_path if not explicitly set
    if cfg.dataset_path is None:
        data_dir = "./data"  # relative to cwd, matching pattern in data/ folder
        dataset_filename = f"ClothFold_vars-{cfg.vars}_eps-{cfg.eps}_img-128.pkl"
        auto_path = os.path.join(data_dir, dataset_filename)
        if os.path.exists(auto_path):
            cfg.dataset_path = auto_path
            print(f"[auto-resolve] dataset_path: {auto_path}")
        else:
            print(f"[auto-resolve] Warning: expected dataset not found: {auto_path}")
    
    # Auto-configure env_kwargs for evaluation if not already set
    if not hasattr(cfg, 'env_kwargs') or cfg.env_kwargs is None:
        # Create basic ClothFold env_kwargs with correct num_variations
        auto_env_kwargs = {
            'env': 'ClothFold',
            'env_kwargs': {
                'num_variations': cfg.vars,
                'observation_mode': 'key_point',  # match lowdim training
                'action_mode': 'picker',         # REQUIRED for ClothEnv
                'num_picker': 2,
                'render': True,
                'headless': True,
                'horizon': 100,                  # match registered env default
                'action_repeat': 8,
                'render_mode': 'cloth',
                'use_cached_states': True,       # use pre-generated variations
                'deterministic': False,
            },
            'symbolic': True,  # key_point mode is symbolic
            'seed': cfg.seed,
            'max_episode_length': 50,
            'action_repeat': 1,
            'bit_depth': 8,
            'image_dim': 128,
            'normalize_observation': False,
            'scale_reward': 1.0,
            'clip_obs': None,
            'obs_process': None
        }
        setattr(cfg, 'env_kwargs', auto_env_kwargs)
        print(f"[auto-resolve] env_kwargs: ClothFold with num_variations={cfg.vars}")
    else:
        # If env_kwargs exists, just ensure num_variations is set correctly
        if isinstance(cfg.env_kwargs, dict) and 'env_kwargs' in cfg.env_kwargs:
            cfg.env_kwargs['env_kwargs']['num_variations'] = cfg.vars
            print(f"[auto-resolve] Updated existing env_kwargs num_variations to {cfg.vars}")
