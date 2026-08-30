<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers
(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# G1 jump MuJoCo sim2sim

This harness runs the validated longitudinal G1 jump actor against
`data_storage/g1_23dof_holo_compat.xml`. It never edits that source model. At startup,
`model_overlay.py` composes `model_overlay.xml` into an in-memory MJCF and verifies that
the overlay timestep equals `control.sim_dt` in `deploy_manifest.json`.

Run the three-second passive-hold diagnostic to confirm that the model composes and to
record its fall/contact metrics. It intentionally has no stability pass/fail threshold:
the low-gain open-loop default pose is not a balance controller.

```bash
./isaaclab.sh -p scripts/g1_jump_deploy/mujoco/deploy_mujoco.py \
  --manifest logs/g1_jump_deploy_bundle_validated/deploy_manifest.json \
  --self-check --headless --log /tmp/g1_stand_check.npz
```

Then run the ONNX policy. Omitted goal components use the midpoint of their manifest
ranges; every supplied component is checked against those ranges.

```bash
./isaaclab.sh -p scripts/g1_jump_deploy/mujoco/deploy_mujoco.py \
  --manifest logs/g1_jump_deploy_bundle_validated/deploy_manifest.json \
  --goal_pos_x 0.1 --goal_pos_y 0.0 --goal_yaw 0.0 \
  --headless --log /tmp/g1_jump_mujoco.npz
```

The action delay is sampled uniformly once per run from the manifest range, matching an
Isaac reset. `--delay_steps` selects a reproducible value within that range. Policy
inference and action processing run at the manifest policy rate. The filtered position
target is held while MuJoCo position actuators solve PD at the manifest simulation rate.
ONNX Runtime is used when installed; the standard ONNX reference evaluator is a functional,
slower fallback.

## Validation snapshot

The `g1_jump_deploy_bundle_validated` artifact was accepted on 2026-08-27 as a
sim2sim candidate, not as authorization for an untethered hardware jump.

- Isaac endpoint commands of -0.1, 0.0, and 0.1 m produced longitudinal
  displacements of -0.0878, 0.0122, and 0.0955 m. Their planar errors were
  0.0119, 0.0167, and 0.0219 m.
- A nine-command free-MuJoCo sweep had a 0.951 command/displacement
  correlation. Eight cases finished within 0.1 m planar error; the worst was
  0.1012 m. Every case was airborne, upright at completion, and respected the
  0.6 effort-limit ratio and physical knee bounds.
- Fifteen contact-compliance cases spanning five contact time constants and
  three commands all remained upright and within the same torque and joint
  limits. Thirteen finished within 0.1 m planar error; the worst was 0.1376 m.
- The gantry FSM completed nominal zero and endpoint scenarios without a jump
  violation, and its reject, early-abort, and late-abort scenarios followed
  their expected damping paths. Gantry displacement is not used as the
  commandability gate because its attitude restraints alter the motion.

See `../hardware/README.md` for the read-only live-feedback and policy-shadow
checks required before motor-control integration.

## Contactless hardware-envelope replay

Before a suspended hardware rehearsal, reproduce its exact conservative command
contract in MuJoCo. This mode disables ground collision, supplies full simulated
gantry support, calibrates the balance target from initial simulated feedback,
uses separate delayed start/confirm edges, and enables the FSM's explicit
no-contact safety mode:

```bash
./isaaclab.sh -p scripts/g1_jump_deploy/fsm/run_fsm_mujoco.py \
  --manifest logs/g1_jump_deploy_bundle_validated/deploy_manifest.json \
  --scenario nominal --contactless_gantry_rehearsal \
  --goal_pos_x 0.0 --goal_pos_y 0.0 --goal_roll 0.0 \
  --goal_pitch 0.0 --goal_yaw 0.0 \
  --max_duration 15 --start_time_s 4.5 --confirm_time_s 9.0 \
  --effort_scale 0.1 --target_rate_limit_rad_s 1.2 \
  --gantry_support_fraction 1.0 --stand_entry_duration_s 4.0 \
  --stand_ankle_stiffness 80.0 --stand_ankle_damping 7.0 \
  --balance_disable_integral --balance_initial_roll_integral 0.0 \
  --balance_initial_pitch_integral 0.0 --headless \
  --log /tmp/g1_contactless_envelope_fsm.npz
```

The accepted 2026-08-27 replay completed `STAND` -> `GOTO_START` -> `ARMED`
-> `JUMP` -> `SETTLE` -> `STAND` with all 152 policy steps, zero simulated foot force,
no joint-limit violation, 13.49 deg peak tilt, 1.981 rad/s peak joint speed,
and at most 10% of each manifest effort limit. With measured-state settle
acceptance enabled, `SETTLE` lasted 1.20 seconds before the joints met the stand
pose and velocity tolerances. Saturating that deliberately small scaled effort
envelope is expected. This replay validates the restricted command path, not
the real gantry, unmodeled robot dynamics, or ground contact; it does not
authorize a ground jump.

## Ground physics with contact hidden from the FSM

The missing simulator quadrant can be exercised without changing the hardware
boundary. `--unmeasured_ground_validation` leaves the ground and MuJoCo contact
physics enabled, but selects an FSM contract that never reads foot contact. The
logger retains MuJoCo's contact truth solely as an oracle and requires bilateral
support before the jump, an airborne interval, bilateral touchdown, and measured
settlement before reporting `PASS`:

```bash
./isaaclab.sh -p scripts/g1_jump_deploy/fsm/run_fsm_mujoco.py \
  --manifest logs/g1_jump_deploy_bundle_validated/deploy_manifest.json \
  --scenario nominal --unmeasured_ground_validation \
  --goal_pos_x 0.0 --goal_pos_y 0.0 --goal_roll 0.0 \
  --goal_pitch 0.0 --goal_yaw 0.0 --max_duration 10 \
  --headless --log /tmp/g1_unmeasured_ground_zero.npz
```

This is a simulation diagnostic, not hardware authorization. Without measured or
validated estimated contact, a real controller cannot verify loaded feet before
arming or identify touchdown for abort handling.

For the accepted bundle on 2026-08-27, this diagnostic exposed a separate ground
handoff blocker. At full effort with the bundle's standard stand gains, the model
fell during `GOTO_START` and never armed. A simulation-only stronger hip-pitch and
knee stand profile reached flight and bilateral touchdown, but crossed the left
ankle-roll physical limit by about 0.003 rad and therefore ended in `DAMPING`.
Applying the current hardware guard envelope of 70% effort and a 1.2 rad/s target
slew also fell before arming. Consequently, the current bundle has no passing
ground-FSM configuration, with or without contact feedback.

Three compiled-model corrections are enabled by default to match Isaac: solver-side implicit
PD, zero passive joint stiffness/damping, and zero Coulomb joint `frictionloss`. They can be
disabled independently with `--no_use_implicit_pd`, `--no_zero_passive_forces`, and
`--no_zero_frictionloss`. The checked-in MJCF is never modified.

## Stand-only FSM viewer

Use the stand scenario to exercise the 500 Hz IMU/ankle balance controller without
loading or executing the jump policy. Omit `--headless` to open the passive MuJoCo
viewer; closing it before `--max_duration` reports the scenario as incomplete.

```bash
./isaaclab.sh -p scripts/g1_jump_deploy/fsm/run_fsm_mujoco.py \
  --manifest logs/g1_jump_deploy_bundle_validated/deploy_manifest.json \
  --scenario stand --max_duration 10 \
  --log /tmp/g1_stand_fsm.npz
```

Isaac policy order and MJCF declaration order are expected to differ. At startup the
runner requires identical, unique 23-joint name sets, builds name-based joint and actuator
permutations, and prints all four forward/inverse mappings. Joint observations and logs
remain in policy order; targets, gains, and limits are scattered into actuator order for
PD. The precomputed reference preview is never permuted.

The NPZ has one row for the initial state and each physics/PD step. It contains `time`, integer `phase`, policy-order
`qpos` and `qvel`, raw `action`, `delayed_action`, `q_target`, `applied_tau`,
`pelvis_pose`, `pelvis_velocity`, left/right `foot_contact_forces`, and the held full
`observation`. `applied_tau` comes from MuJoCo's `data.actuator_force`, sampled after
`mj_step` and before the diagnostic `mj_forward`, in policy joint order. `metadata_json`
records the three parity switches, torque source, coordinate conventions, and source paths.
The state is sampled after each physics step; its torque was applied over that step, while
the observation is the one assembled at the most recent policy tick.
Cross-check logs additionally include the true pre-physics frame-0 state at `time=0`.

## Deterministic Isaac/MuJoCo cross-check

The fidelity gate drives both simulators with zero raw actions by default (a default-pose
hold through the real action transform), fixes the action delay and seed to zero, and logs
every 2 ms physics step. An optional NPY file supplies one policy-order raw action per
50 Hz policy step and must have shape `[N, 23]`.

```bash
./isaaclab.sh -p scripts/g1_jump_deploy/mujoco/deploy_mujoco.py \
  --manifest logs/g1_jump_deploy_bundle_validated/deploy_manifest.json \
  --cross-check --action-sequence /tmp/actions.npy --headless \
  --log /tmp/mujoco_openloop.npz

./isaaclab.sh -p scripts/g1_jump_deploy/mujoco/isaac_openloop_replay.py \
  --manifest logs/g1_jump_deploy_bundle_validated/deploy_manifest.json \
  --action-sequence /tmp/actions.npy --headless \
  --log /tmp/isaac_openloop.npz

./isaaclab.sh -p scripts/g1_jump_deploy/mujoco/compare_isaac_mujoco.py \
  /tmp/isaac_openloop.npz /tmp/mujoco_openloop.npz
```

The Isaac replay constructs the task named by the manifest with one environment, resets
through its frame-0 training path, explicitly writes and verifies the CSV frame-0 joint
positions, root height/orientation, and zero velocities, and disables dynamics randomization, pushes,
observation noise, action delay, and stochastic odometry drift. The comparison requires
identical manifest contents, timestamps, goals, raw actions, and initial states; remaps each
log's joints by manifest names; then reports joint, pelvis, contact-timing, and MuJoCo
tendon-limit divergence.

## Overlay physics choices

- The timestep is 0.002 s, and gravity is explicitly `(0, 0, -9.81)` m/s².
- `implicitfast` is used to handle stiff actuation robustly at the 2 ms step. `Newton`
  with up to 100 iterations and a `1e-8` tolerance solves the coupled contact/joint
  constraints. The elliptic cone is the closest physical Coulomb-friction model available
  in MuJoCo.
- The infinite plane is at z=0 and uses six-dimensional contact with sliding, torsional,
  and rolling coefficients `(1.0, 0.005, 0.0001)`. Its higher material priority prevents
  contact mixing from changing those values.
- Runtime-composed collision masks allow robot/ground contacts and reject every
  robot/robot pair, matching Isaac's disabled articulation self-collision.

## Physics differences that remain

No MuJoCo settings make its dynamics identical to PhysX. Remaining differences are:

- MuJoCo's implicit integrator and Newton constraint optimizer are not PhysX's temporal
  Gauss-Seidel integration/solve path. Solver warm-starting and convergence differ.
- MuJoCo uses regularized soft constraints (`solref`/`solimp`) and an elliptic friction
  cone. PhysX uses contact offsets, its own compliance/restitution rules, and friction
  patches. Contact manifolds, impact impulses, sticking, torsion, and rolling therefore
  differ even with identical coefficients.
- Convex/capsule collision detection and manifold reduction differ, especially across the
  seven capsule geoms on each foot and at joint-limit impacts.
- The MJCF ankle tendons and joint limits are solved with MuJoCo constraints. Their converted
  USD/PhysX counterparts need not produce identical forces.
- MuJoCo position-actuator PD and Isaac implicit articulation drives can still differ in
  discretization and saturation even though gains, armatures, and effort limits come from
  the same manifest.
- The fixed manifest schema has no root-reset pose. This repository harness therefore reads
  frame 0 from `data_storage/perfect_jump_processed.csv`, verifies its joint positions
  against the manifest, and writes the same root pose to both simulators. A standalone
  deployment bundle still needs an architect-approved root-state source.
- MuJoCo runs CPU floating-point kernels; Isaac commonly runs batched PhysX on GPU.
  Reduction order and accumulated numerical error differ.
- The deterministic runner adds no synthetic observation noise or dynamics
  randomization. Stage 3 training applies both; real hardware supplies sensor noise and
  unmodeled dynamics instead.

These differences bound what the harness can diagnose: a mismatch in observation layout,
quaternion handling, delay, filtering, or torque limits is a harness defect; small contact
and trajectory differences around takeoff and landing can be simulator physics.
