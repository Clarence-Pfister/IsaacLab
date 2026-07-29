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

This repository trains a 23-DoF Unitree G1 to reproduce a reference jump with RSL-RL. It also includes a
stand-still task that is useful for validating the robot, simulation, and PPO setup before starting the more
expensive jump training run.

## What this fork adds

- A six-phase G1 jump task: idle, crouch, takeoff, flight, landing, and stand.
- A CSV reference-motion loader with joint, root, quaternion SLERP, and foot-position interpolation.
- Reference-state initialization, future reference previews, phase observations, contact checks, and
  phase-weighted tracking rewards.
- Separate G1 stand training and play environments.
- PPO configurations for the jump and stand tasks.
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
| `Isaac-Velocity-Jump-G1-v0` | Train the reference-motion jump | `g1_jump` |
| `Isaac-Velocity-Jump-G1-Play-v0` | Evaluate a jump checkpoint | `g1_jump` |
| `Isaac-Velocity-Stand-G1-v0` | Train the G1 to stand still | `g1_stand` |
| `Isaac-Velocity-Stand-G1-Play-v0` | Evaluate a stand checkpoint | `g1_stand` |

The task registrations are in
[`config/g1/__init__.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py).

## Branches

The fork keeps upstream, each concern, and the combined working state on separate branches so that a future
project can reuse a subset without inheriting the rest.

| Branch | Contents |
| --- | --- |
| `main` | Upstream `release/3.0.0-beta2`, unmodified. The fork baseline. |
| `feature/g1-jump` | The G1 jump and stand tasks, reference motion, assets, and PPO configs. |
| `fix/docker` | Container tooling fixes only, branched from `main`. Reusable on its own. |
| `integration/all` | Merge of the topic branches. This is the branch to check out for development. |

To start another Isaac Lab project on this fork, branch `feature/<name>` from `main`, merge `fix/docker` into it
if the container fixes are wanted, and combine on a new integration branch.

## Repository map

| Path | Contents |
| --- | --- |
| [`jump_env_cfg.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/jump_env_cfg.py) | Jump robot, motion loader, observations, rewards, events, and terminations |
| [`stand_env_cfg.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/stand_env_cfg.py) | Stand-still environment |
| [`rsl_rl_ppo_cfg.py`](source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py) | PPO runner configurations |
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
`.env.base` as well would be redundant and can disagree with the value the script uses. The argument has no
default, so pass `--suffix custom` consistently to every `container.py` call, as the commands below do. An
inconsistent suffix is the usual cause of "the container is not running" when it plainly is.

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

### Jump task

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-v0 \
  --num_envs 1000 \
  --viz none
```

### Stand task

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Stand-G1-v0 \
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

```bash
./isaaclab.sh train \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-v0 \
  --resume \
  --checkpoint logs/rsl_rl/g1_jump/RUN_DIRECTORY/model_ITERATION.pt \
  --viz none
```

## Evaluate a checkpoint

```bash
./isaaclab.sh play \
  --rl_library rsl_rl \
  --task Isaac-Velocity-Jump-G1-Play-v0 \
  --checkpoint logs/rsl_rl/g1_jump/RUN_DIRECTORY/model_ITERATION.pt \
  --num_envs 1 \
  --viz kit
```

If `--checkpoint` is omitted, Isaac Lab selects the latest checkpoint under the task's experiment directory.

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

Jump runs are written to `logs/rsl_rl/g1_jump/`; stand runs are written to `logs/rsl_rl/g1_stand/`.

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
- **No checkpoint found:** pass an explicit path under `logs/rsl_rl/g1_jump/` or `logs/rsl_rl/g1_stand/`.
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

After resolving any conflicts, regenerate the G1 USD if converter or schema behavior changed, then run a short stand
and jump smoke test before starting a full training job.

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

This fork retains the upstream Isaac Lab copyright notices, contributor history, and
[`CITATION.cff`](CITATION.cff). If you use Isaac Lab in published research, cite the upstream project as described in
the citation file and clearly identify any project-specific modifications used in your experiments.
