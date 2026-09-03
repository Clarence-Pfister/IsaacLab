![Isaac Lab](docs/source/_static/isaaclab.jpg)

# IsaacLab G1 Jump

[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-3.0.0%20Beta%202-76b900.svg)](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/)
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-6.0.1-76b900.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://docs.python.org/3.12/)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-yellow.svg)](LICENSE)

> [!IMPORTANT]
> This is a research fork of [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab), not an official NVIDIA
> distribution. `main` tracks upstream `release/3.0.0-beta2`; the project-specific Unitree G1 tasks, assets,
> rewards, and container fixes live on topic branches that are combined on `integration/all`.

> [!WARNING]
> **This repository contains code that commands a physical robot.**
> [`scripts/g1_jump_deploy/hardware/`](scripts/g1_jump_deploy/hardware) publishes joint targets to a
> real Unitree G1 over `unitree_sdk2`, and its `--ground_jump` mode commands a jump on a real floor.
> A humanoid executing a jump can injure people and destroy itself. This is unsupported research
> code, not a validated product.
>
> Before running any hardware path, read
> [`scripts/g1_jump_deploy/hardware/README.md`](scripts/g1_jump_deploy/hardware/README.md) in full.
> It defines the required preflight order and the physical preconditions for each stage. Note that
> the validated deployment bundle, shadow-admission files, and audit logs those commands require
> live under `logs/`, which is **not** distributed with this repository — the hashes recorded there
> identify artifacts you must export and validate yourself.
>
> Your robot, your test area, and your safety case are your own responsibility.

This repository trains a 23-DoF Unitree G1 to reproduce a reference jump with RSL-RL.

## What this fork adds

- A six-phase G1 jump task: idle, crouch, takeoff, flight, landing, and stand.
- A CSV reference-motion loader with joint, root, quaternion SLERP, and foot-position interpolation.
- Reference-state initialization, future reference previews, phase observations, contact checks, and
  phase-weighted tracking rewards.
- PPO configuration for the jump task.
- A 23-DoF G1 MJCF asset, meshes, and processed jump reference data.
- Host bind mounts for `logs/` and `data_storage/`, so results generated in Docker are immediately available on
  the host.

## Compatibility

| Component | Version or branch |
| --- | --- |
| Working branch | `integration/all` |
| Isaac Lab base | `release/3.0.0-beta2` (tracked by `main`) |
| Isaac Lab package version | `3.0.0` |
| Isaac Sim container | `6.0.1` |
| Python | `3.12` |
| RL library | RSL-RL |

Isaac Lab and Isaac Sim versions are coupled. Do not change one without checking the upstream
[compatibility documentation](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/index.html).

## Project tasks

| Task ID | Purpose | Default experiment |
| --- | --- | --- |
| `Isaac-Velocity-Jump-G1-v0` | Stage 1: imitate the reference jump, goal fixed at the origin | `g1_jump` |
| `Isaac-Velocity-Jump-G1-Play-v0` | Evaluate a stage 1 checkpoint | `g1_jump` |
| `Isaac-Velocity-Jump-G1-Stage2-v0` | Stage 2: jump to a goal resampled each episode | `g1_jump` |
| `Isaac-Velocity-Jump-G1-Stage2-Play-v0` | Evaluate a stage 2 checkpoint, with the goal drawn | `g1_jump` |
| `Isaac-Velocity-Jump-G1-Stage2-Wide-v0` | Stage 2 at wider goal ranges and a tighter arrival bound | `g1_jump` |
| `Isaac-Velocity-Jump-G1-Stage2-Wide-Play-v0` | Evaluate a wide stage 2 checkpoint | `g1_jump` |

The task registrations are in
[`config/g1/__init__.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py).

### Training stages

The task follows the multi-stage scheme of [Li et al., *Robust and Versatile Bipedal Jumping Control
through Reinforcement Learning* (2023)](https://arxiv.org/abs/2302.09450): learn one jump first, then
generalize it to arbitrary goals. Each stage is a config subclass registered as its own task, so the
stage travels with `--task` and needs no extra flag.

| | Stage 1 | Stage 2 | Stage 2 wide |
| --- | --- | --- | --- |
| Goal | fixed at the origin, no turn | `pos_x` ±0.4 m, `pos_y` ±0.3 m, `yaw` ±30° | `pos_x` −0.3…1.0 m, `pos_y` ±0.6 m, `yaw` ±60° |
| Reference tracking | full weight | heading, angular rate and foot ground track dropped; joint position halved before landing | as stage 2 |
| Task reward | position and velocity only | plus orientation and angular rate | as stage 2, kernels rescaled for the range |
| Arrival bound | 1.0 m / 45° | 1.0 m / 45° | 0.35 m / 35° |
| Elevation | flat | flat (the paper trains elevation as a separate policy) | flat |

The wide stage is a separate task rather than an edit to stage 2, so a working narrow policy stays
reproducible. Two things change with the range beyond the ranges themselves. The task reward kernels
are `exp(-k · squared error)`, so `k` only means something relative to the errors actually seen: at
stage 2's `k = 21.07`, a robot standing still 0.62 m from its goal scores 3e-4 and gets essentially no
gradient, so position and orientation are rescaled to 3.72 and 6.0. And the arrival bound tightens to
the paper's stage 2 values — at ±0.4 m goals a 1.0 m bound could not be violated even by a robot that
never moved, so it was not testing anything.

Both stages share the 165-wide observation — stage 1 simply sees a goal of all zeros — so a stage 2
run can warm-start from a stage 1 checkpoint. They also share the `g1_jump` experiment directory,
which is what makes that warm start straightforward.

Stage 3 (dynamics randomization) is not implemented.

## Branches

The fork keeps upstream, each concern, and the combined working state on separate branches so that a future
project can reuse a subset without inheriting the rest.

| Branch | Contents |
| --- | --- |
| `main` | Upstream `release/3.0.0-beta2`, unmodified. The fork baseline. |
| `feature/g1-jump` | The G1 jump task, reference motion, assets, and PPO config. |
| `fix/docker` | Container tooling fixes only, branched from `main`. Reusable on its own. |
| `integration/all` | Merge of the topic branches. This is the branch to check out for development. |

To start another Isaac Lab project on this fork, branch `feature/<name>` from `main`, merge `fix/docker` into it
if the container fixes are wanted, and combine on a new integration branch.

## Repository map

The jump task is a package under
[`config/g1/jump/`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/jump),
laid out like the `mdp` packages upstream uses:

| Path | Contents |
| --- | --- |
| [`jump/constants.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/jump/constants.py) | Asset paths, joint layout, action scales, actuators, motion phases |
| [`jump/jump_env_cfg.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/jump/jump_env_cfg.py) | Scene, observation, reward and termination configs, and the stage classes |
| [`jump/mdp/motion.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/jump/mdp/motion.py) | Reference-motion loader, interpolation, phase helpers |
| [`jump/mdp/rewards.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/jump/mdp/rewards.py) | Tracking, task-completion and smoothing reward terms |
| [`jump/mdp/terminations.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/jump/mdp/terminations.py) | Reference exhaustion, ground contact, tracking and task-completion bounds |
| [`jump/mdp/commands.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/jump/mdp/commands.py) | Goal command term and its visualization marker |
| [`jump/mdp/`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/jump/mdp) | Also observations, events, and the filtered action term |
| [`rsl_rl_ppo_cfg.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py) | PPO runner configurations |

> **When editing this package, keep USD out of the import path.** `import isaaclab_tasks` walks every
> subpackage, and task configs are resolved by hydra, both before `SimulationApp` starts. Importing
> USD that early aborts the process with `free(): invalid pointer` — for every task, not just this
> one. So `jump/__init__.py` re-exports nothing, and `jump/mdp/__init__.py` exposes the action
> *config* but not the action class, whose base lazy-loads `Articulation`. Runtime classes are
> reached from their `class_type` strings once the app is running. To check a change is safe:
>
> ```bash
> ./isaaclab.sh -p -c "import sys, isaaclab_tasks; \
>   from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry; \
>   load_cfg_from_registry('Isaac-Velocity-Jump-G1-Stage2-v0', 'env_cfg_entry_point'); \
>   print('pxr modules:', len([m for m in sys.modules if m.startswith('pxr')]))"
> ```
>
> It must print `0`.
| [`data_storage/`](data_storage) | G1 MJCF, meshes, processed reference CSV, and generated USD location |
| [`docker/.env.base`](docker/.env.base) | Isaac Sim image, container naming, and streaming host settings |
| [`docker/docker-compose.yaml`](docker/docker-compose.yaml) | Container services and host bind mounts |

## Getting started with Docker

### Prerequisites

- A Linux machine with a supported NVIDIA GPU and driver.
- Docker Engine and Docker Compose.
- NVIDIA Container Toolkit configured for Docker.
- Enough disk space for the Isaac Sim image, build layers, shader caches, logs, and checkpoints.

See the upstream [Docker guide](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/deployment/docker.html)
for host setup and troubleshooting.

### 1. Clone this fork

```bash
git clone --branch integration/all \
  https://github.com/Clarence-Pfister/IsaacLab.git
cd IsaacLab
```

Clone without `--single-branch` so that `main`, `feature/g1-jump`, and `fix/docker` remain available for
rebasing and for starting new projects.

### 2. Configure the container

Review [`docker/.env.base`](docker/.env.base) before building. The project currently uses:

```dotenv
ISAACSIM_VERSION=6.0.1
COMPOSE_PROJECT_NAME=isaac-lab-custom
ISAACSIM_HOST=<GPU_HOST_IP>
```

The image and container name suffix is not set here. It comes from the `--suffix` argument of
[`docker/container.py`](docker/container.py), which exports `DOCKER_NAME_SUFFIX` to Compose itself; setting it in
`.env.base` as well would be redundant and can disagree with the value the script uses. On `integration/all`
the argument defaults to `custom` (the default comes from `fix/docker`), so the commands below pass
`--suffix custom` explicitly to stay correct on any branch. Whatever value you use, use the same one for
`build`, `start`, `enter`, and `stop`: an inconsistent suffix is the usual cause of "the container is not
running" when it plainly is.

Replace `<GPU_HOST_IP>` with the GPU machine address that the streaming client can reach. Keep the EULA enabled only
after reviewing and accepting the NVIDIA Omniverse license terms. Avoid committing private hostnames, credentials, or
new machine-specific addresses.

### 3. Build and enter the container

The suffix keeps this project isolated from other Isaac Lab containers on the same host.

```bash
./docker/container.py build base --suffix custom
./docker/container.py start base --suffix custom
./docker/container.py enter base --suffix custom
```

Stop the container from the host when finished:

```bash
./docker/container.py stop base --suffix custom
```

The fork bind-mounts `source/`, `scripts/`, `docs/`, `tools/`, `logs/`, and `data_storage/`. Code changes and training
artifacts therefore remain visible in the host checkout; routine `docker cp` commands are not required.

### 4. Regenerate the G1 USD (optional)

The generated USD is committed, so a fresh clone can build the environment without running the converter. Upstream
`.gitignore` excludes every `*.usd`/`*.usda`; this one directory is re-included by an explicit exception because
`G1_USD_PATH` loads it from disk. Regenerate it only after changing the MJCF or the converter:

```bash
./isaaclab.sh -p scripts/tools/convert_mjcf.py \
  data_storage/g1_23dof_holo_compat.xml \
  data_storage/g1_23dof_holo_compat \
  --collision-type "Convex Hull" \
  --viz none
```

> [!NOTE]
> The second argument selects a **directory**, not a file. The filename part is discarded:
> [`convert_mjcf.py`](scripts/tools/convert_mjcf.py) passes only `usd_dir=os.path.dirname(output)` and never sets
> `usd_file_name`, and [`MjcfConverter`](source/isaaclab/isaaclab/sim/converters/mjcf_converter.py) then overwrites
> it with `<mjcf_stem>/<mjcf_stem>.usda`. The output is therefore always
> `<dirname_of_second_argument>/<mjcf_stem>/<mjcf_stem>.usda`, one level deeper than the path written on the
> command line. Passing `.../g1_23dof_holo_compat/g1_23dof_holo_compat.usda` and passing
> `.../g1_23dof_holo_compat` produce exactly the same result.

The command above therefore writes:

```text
data_storage/g1_23dof_holo_compat/          <- usd_dir: .asset_hash, config.yaml
└── g1_23dof_holo_compat/                   <- created by the importer
    ├── g1_23dof_holo_compat.usda           <- the file G1_USD_PATH points at
    ├── payloads/
    └── Textures/
```

The jump environment reads both that USD and `data_storage/perfect_jump_processed.csv` at startup.

## Train

The top-level commands below are the supported Isaac Lab 3.0 interface. The older direct
`scripts/reinforcement_learning/rsl_rl/*.py` entry points are deprecated.

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-v0 \
  --num_envs 1000 \
  --viz none
```

The jump runner defaults to 100,000 iterations and saves every 500 iterations. Override that during smoke tests:

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-v0 \
  --num_envs 32 \
  --max_iterations 2 \
  --viz none
```

### Resume training

`--resume` is a flag and is required — `--load_run` and `--checkpoint` alone load nothing. `--load_run`
takes a run directory *name* under the experiment directory, and `--checkpoint` a file name within it.

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-v0 \
  --resume \
  --load_run RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --viz none
```

### Start stage 2 from a stage 1 checkpoint

Stage 2 is a fresh run that continues from stage 1's weights, which works because both stages share
the observation width and the experiment directory:

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-Stage2-v0 \
  --resume \
  --load_run STAGE1_RUN_DIRECTORY \
  --checkpoint model_ITERATION.pt \
  --max_iterations 6000 \
  --run_name stage2 \
  --viz none
```

Expect the reward to drop sharply at iteration 0. Stage 2 zeroes three reference-tracking terms and
enables two task terms, so the scale changes; what matters is that it climbs from there.

## Evaluate a checkpoint

```bash
./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-Play-v0 \
  --checkpoint logs/rsl_rl/g1_jump/RUN_DIRECTORY/model_ITERATION.pt \
  --num_envs 1 \
  --viz kit
```

If `--checkpoint` is omitted, Isaac Lab selects the latest checkpoint under the task's experiment
directory. Because every jump task shares `g1_jump`, the newest run wins regardless of which stage
produced it — a two-iteration smoke test will be picked over a finished run and the robot will appear
frozen in its start pose. Name the run explicitly with `--load_run`, and send throwaway runs somewhere
else with `--experiment_name g1_jump_scratch`.

Stage 2 play draws the commanded landing pose as a frame triad, so a missed landing can be told apart
from a goal that moved:

```bash
./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-Stage2-Play-v0 \
  --checkpoint logs/rsl_rl/g1_jump/RUN_DIRECTORY/model_ITERATION.pt \
  --viz kit
```

Note that `play` and `train` read `--checkpoint` differently. `train` takes a file *name* resolved
inside `--load_run`; `play` takes a *path* and ignores `--load_run` entirely when `--checkpoint` is
given. Passing the train form to `play` fails with `FileNotFoundError`.

The marker shows the goal in world frame, so its heading is the robot's starting yaw plus the
commanded turn — a triad past ±30° is expected, not a bug. `--viz kit` needs a display and so does not
work over plain SSH; record with `--viz none --video` instead.

### Record video

```bash
./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-Play-v0 \
  --checkpoint logs/rsl_rl/g1_jump/RUN_DIRECTORY/model_ITERATION.pt \
  --num_envs 1 \
  --video \
  --video_length 400 \
  --viz kit
```

For periodic video during training, add `--video --video_length 400 --video_interval 4000 --viz kit`.

### Common options

| Option | Meaning |
| --- | --- |
| `--task` | Registered training or play environment ID |
| `--num_envs` | Number of parallel simulated environments |
| `--max_iterations` | PPO iteration limit |
| `--resume` | Resume an existing run |
| `--checkpoint` | Checkpoint path or filename to load |
| `--video` | Enable video recording |
| `--video_length` | Recorded clip length in environment steps |
| `--video_interval` | Training steps between recordings |
| `--viz none` | Disable visualizers for maximum training throughput |
| `--viz kit` | Use the Isaac Sim Kit visualizer |
| `--livestream 2` | Enable WebRTC streaming for private or local networks |

Run `./isaaclab.sh train --help` or `./isaaclab.sh play --help` for the complete version-specific CLI.

## Monitor training

From another shell in the same environment:

```bash
./isaaclab.sh -p -m tensorboard.main --logdir logs/rsl_rl
```

Jump runs are written to `logs/rsl_rl/g1_jump/`.

## Remote execution and streaming

For long runs, connect to the GPU host and keep the container shell in a named `tmux` session:

```bash
ssh USER@GPU_HOST
tmux new -s g1-jump
./docker/container.py enter base --suffix custom
```

Detach with `Ctrl+B`, then `D`, and reconnect later with:

```bash
tmux attach -t g1-jump
```

To stream a play run over a private network:

```bash
./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-Play-v0 \
  --checkpoint logs/rsl_rl/g1_jump/RUN_DIRECTORY/model_ITERATION.pt \
  --num_envs 1 \
  --livestream 2 \
  --viz kit
```

Use an [Isaac Sim livestream client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/manual_livestream_clients.html)
compatible with the configured Isaac Sim version. Firewall rules depend on the client, Isaac Sim version, and network
topology. Expose only the documented ports, prefer a VPN or private network, and do not expose the stream directly to
the public internet.

## Troubleshooting

- **Generated USD not found:** the USD is committed, so check out the assets rather than regenerating. If you did
  regenerate, note that the converter ignores the filename you pass and writes
  `<dir>/<mjcf_stem>/<mjcf_stem>.usda`, so the file sits one directory deeper than the argument suggests.
- **Reference CSV not found:** confirm `data_storage/perfect_jump_processed.csv` exists inside the container.
- **No checkpoint found:** pass an explicit path under `logs/rsl_rl/g1_jump/`.
- **No viewport:** use `--viz kit`; `--viz none` intentionally disables visualization.
- **Video is empty or unavailable:** use `--video --viz kit`; video automatically enables the offscreen camera pipeline.
- **Container naming conflict:** use a unique `--suffix` consistently for `build`, `start`, `enter`, and `stop`.
- **Need GPU diagnostics:** run `watch -n 1 nvidia-smi` on the GPU host.

For framework-level problems, consult the upstream
[troubleshooting guide](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/refs/troubleshooting.html).

## Keeping the fork synchronized

Use `origin` for this fork and reserve `upstream` for NVIDIA Isaac Lab. Advance `main` first so it keeps meaning
"unmodified upstream", then merge it into the topic branches and re-integrate. Review upstream changes before
merging because Isaac Lab and Isaac Sim frequently introduce coupled API changes.

```bash
git remote add upstream https://github.com/isaac-sim/IsaacLab.git  # first time only
git fetch upstream

git switch main                                # main stays pure upstream
git merge --ff-only upstream/release/3.0.0-beta2

git switch fix/docker      && git merge main
git switch feature/g1-jump && git merge main
git switch integration/all && git merge fix/docker feature/g1-jump
```

After resolving any conflicts, regenerate the G1 USD if converter or schema behavior changed, then run a short
jump smoke test before starting a full training job.

## Upstream documentation

- [Isaac Lab 3.0.0 Beta 2 documentation](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/)
- [Installation](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/index.html)
- [Reinforcement learning workflows](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/overview/environments.html)
- [Contributing to Isaac Lab](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/refs/contributing.html)

## License and attribution

Isaac Lab is released under the [BSD 3-Clause License](LICENSE). The `isaaclab_mimic` extension and its standalone
scripts are released under the [Apache License 2.0](LICENSE-mimic). Dependency and asset licenses are collected under
[`docs/licenses/`](docs/licenses).

The G1 meshes, MJCF, and generated USD under [`data_storage/`](data_storage) are derived from the Unitree G1 robot
description and remain under Unitree Robotics' BSD 3-Clause terms. See
[`data_storage/NOTICE.md`](data_storage/NOTICE.md) for the attribution and provenance of everything in that directory.

This fork retains the upstream Isaac Lab copyright notices, contributor history, and
[`CITATION.cff`](CITATION.cff). If you use Isaac Lab in published research, cite the upstream project as described in
the citation file and clearly identify any project-specific modifications used in your experiments.
