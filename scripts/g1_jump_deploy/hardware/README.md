<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers
(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# G1 jump hardware preflight

> [!WARNING]
> `run_fsm_g1.py --enable_control` actuates a real robot, and `--ground_jump` commands a jump on a
> real floor. Work through this document in order and do not skip a stage: a passing stage is
> evidence only about the stage that passed. A successful contactless rehearsal, in particular, is
> evidence about the command path only — it does not authorize foot contact or a ground jump.
>
> The validated bundle, shadow-admission files, and audit logs referenced below live under `logs/`,
> which is not distributed with this repository. The SHA-256 recorded here identifies the author's
> validated artifact; export, validate, and record your own before any hardware-bound invocation.

The validated deployment bundle is
`logs/g1_jump_deploy_bundle_validated`. Its ONNX SHA-256 is
`e34d9b763fce6df5afd08f8e53c0d32b692e906aea0308f90e232822f05477a2`.
Every hardware-bound invocation verifies the manifest, ONNX policy, reference
preview, and phase table against `validated_bundle.toml` before opening a
Unitree channel. A changed artifact must be revalidated and recorded; it is not
silently accepted.

Run hardware commands from the Unitree deployment environment. On this PC it
is the `g1_deploy` Conda environment:

```bash
conda activate g1_deploy
```

First verify the manifest-to-SDK joint mapping and feedback stream. This is
strictly read-only:

```bash
python scripts/g1_jump_deploy/hardware/read_g1_state.py NETWORK_INTERFACE
```

Then evaluate policy step 0 from live G1 feedback. This mode loads and warms
the ONNX policy, checks inference latency, and applies the action, knee-brake,
and torque-projection contract in memory. It does not create a command
publisher or locomotion client:

```bash
python scripts/g1_jump_deploy/hardware/run_fsm_g1.py \
  NETWORK_INTERFACE --check_policy --goal_pos_x 0.0
```

After the step-0 check passes, keep G1 motionless under its existing native
controller and shadow the complete 152-step policy timeline. This evaluates
live joint/IMU observations at 50 Hz, enforces feedback and inference
deadlines, computes every guarded target, and writes an NPZ audit log. The
targets are counterfactual and are never published:

```bash
python scripts/g1_jump_deploy/hardware/run_fsm_g1.py \
  NETWORK_INTERFACE --shadow_policy --goal_pos_x 0.0 \
  --shadow_log /tmp/g1_jump_shadow_zero.npz
```

Repeat the shadow at `--goal_pos_x -0.1` and `--goal_pos_x 0.1` with distinct
log paths. Existing logs are deliberately not overwritten. A passing shadow
proves the live SDK mapping, observation path, real-time inference, and command
guards execute end to end; it does not prove that the counterfactual targets
are safe to actuate.

Validate all three logs by replaying their observations, actions, action delay,
target transform, knee brake, and torque projection against the accepted
artifacts. The optional evidence file is also read-only and does not authorize
motor control:

```bash
python scripts/g1_jump_deploy/hardware/validate_shadow_logs.py \
  /tmp/g1_jump_shadow_m100.npz /tmp/g1_jump_shadow_zero.npz \
  /tmp/g1_jump_shadow_p100.npz \
  --admission_output /tmp/g1_jump_shadow_admission.json
```

## Contactless gantry policy rehearsal

G1 `LowState` has no root-position or foot-contact fields. The normal
motor-control path therefore remains stand-only and retains its
measured-contact arming gate. Do not bypass that gate or substitute zero
contact values for measurements. The separately acknowledged ground mode
below uses an explicit no-contact safety contract.

The separate `--gantry_policy_rehearsal` mode can actuate one zero-goal policy
timeline only for a deliberately contactless mechanism check. It skips the
otherwise impossible foot-load and touchdown checks, so its physical contract
is stricter:

- A load-bearing gantry must rigidly constrain pelvis fall, roll, and pitch.
- Both feet must remain unable to touch the floor, gantry, or any other
  structure throughout the full motion envelope.
- The area must be clear, a physical emergency stop must be staffed, and a
  second operator must hold the wireless remote with B ready.
- The accepted live-shadow admission must still match the exact manifest,
  policy, and three immutable shadow logs.
- The goal is fixed to zero, effort is limited to 10% of the manifest limits,
  requested joint targets are slew-limited to 1.2 rad/s before the
  torque-envelope projection, and exit is always native PASSIVE/damping. The
  final target sent to a motor may move faster only when the projection must
  compensate measured motion to keep the estimated PD torque inside that
  envelope.

Higher-speed suspended rehearsals are a separate escalation. Pass
`--rehearsal_effort_scale_override 0.3` (or, only after reviewing that audit,
`0.6`) with `--rehearsal_unlimited_slew` and
`--acknowledge_rehearsal_escalation`. These flags are accepted only by the
contactless gantry mode; they retain torque projection and lower-limit braking
and apply the expanded dynamic feedback envelope during `JUMP`.

First reproduce the exact contactless hardware-envelope replay documented in
`../mujoco/README.md`; it must complete with zero simulated contact and without
violating the hardware feedback limits.

After those mechanical conditions have been independently verified, one
rehearsal invocation is:

```bash
python scripts/g1_jump_deploy/hardware/run_fsm_g1.py \
  NETWORK_INTERFACE --gantry_policy_rehearsal --enable_control \
  --entry_mode gantry_standup --exit_mode passive \
  --duration 15 --effort_scale 0.1 \
  --shadow_admission logs/hardware_shadow/upright_20260827_v2_admission.json \
  --rehearsal_log logs/hardware_rehearsal/zero_FIRST_RUN.npz \
  --acknowledge_contactless_rehearsal
```

Existing audit paths are never overwritten. The program first verifies a B
press/release and a two-second L1+R1 activation hold. Once user control is
active, do not press A or Y until the enforced 4.5-second stabilization ends
and `REHEARSAL READY` is printed. Then tap and release A, wait for `ARMED`, and
tap and release Y when `CONFIRM NOW` is printed. The suspended-only confirmation
window is 15 seconds; expiry stops the rehearsal. Press B at any time to command damping and
return to native PASSIVE. Review the resulting NPZ before considering any
further test. A successful suspended rehearsal is evidence about the command
path only; it still does not authorize foot contact or a ground jump.

After the policy timeline, `SETTLE` continues for at least 0.5 seconds and does
not report success until every non-ankle joint is within 0.05 rad of the stand
pose and every joint is moving at no more than 0.5 rad/s. Failure to converge
within 4 seconds stops the rehearsal and returns to damping.

## Ground jump

`--ground_jump` is an opt-in real-floor mode for the already validated
unmeasured-contact FSM contract. Touchdown cannot be detected. Before takeoff,
an abort enters damping immediately; after takeoff, an abort only latches and
the policy continues through its complete finite episode. A joint may touch a
manifest stop by at most 0.02 rad during `JUMP`; the runner prints the joint
name and penetration depth and records the touch, but does not abort or lock
the session. Travel more than 0.02 rad beyond a stop latches a joint-limit
abort. At episode end only a joint-limit-only abort on an upright robot (at
most 20 degrees body tilt) may proceed through `SETTLE` to `STAND`. Tilt,
stale-feedback, deadline, operator, or combined abort reasons enter damping.
Any latched abort locks the session against further goals until B or `q`
exits. B, `q`, and every fault restore native PASSIVE/damping. A successful
session entered from `native_stand` may instead request `--exit_mode
native_walkrun` to return to the captured native standing pose and restore FSM
801; `--exit_mode passive` remains available as an explicit damping exit.

Every ground session requires all of these physical preconditions:

- Clear the complete motion and landing area.
- Assign a spotter to hold the wireless remote with B ready throughout.
- Adjust the fall-arrest line so it remains slack through the commanded jump
  but catches a damping collapse before either knee reaches the floor.
- Follow this order, reviewing every immutable audit: contactless rehearsal at
  effort 0.3 with unlimited slew, contactless rehearsal at 0.6 with unlimited
  slew, zero-goal ground jump, then the displacement endpoints.
- Do not attempt repeat jumps on hardware until consecutive complete cycles
  pass in MuJoCo.
- Use only an accepted shadow-admission file for the exact validated bundle.

Ground STAND uses 200 N·m/rad and 5 N·m·s/rad on both hip-pitch and knee
joints, plus 80 N·m/rad and 7 N·m·s/rad at the ankles. Its balance target is
the manifest frame-0 attitude, with integral feedback enabled and initial
roll/pitch integral states of 0.0/0.2 rad·s. During the one-second stand entry,
the attitude target moves linearly from the measured handover attitude to that
reference; this prevents a level native-stand handover from being treated as an
instantaneous 7.5-degree pitch error. These values are printed before handover
and recorded in the immutable ground audit. After each policy episode,
`SETTLE` keeps evaluating the policy's final `STAND` reference with the
policy's jump gains for at least 0.5 seconds, rather than switching at landing
to the independent measured-settle controller. It must converge within 4.0
seconds before the FSM returns to `STAND`; otherwise it enters damping.
After the configured sequence reaches `STAND`, a successful native-WALKRUN exit
first checks whether the policy-native pose is within 0.15 rad of the captured
entry pose. If needed, it makes a two-second quintic move to the manifest stand
with the ground stand gains, reports the remaining error, then runs the existing
captured-pose blend, handoff gates, and native-controller monitor. Any failure
in this sequence falls back to PASSIVE/damping.

Torque projection still targets the configured scaled effort envelope. If a
physical target-position bound prevents that projected target from cancelling
the measured damping torque, a bounded command may exceed the scaled envelope
but must remain within the manifest's full physical effort limit. The runner
prints this exception once per affected joint; the audit records its count and
maximum excess. A bounded command beyond the physical effort limit remains a
safety fault.

Every fault path and the native PASSIVE exit publish `kp=0`, `kd=1.5`. This
damping command does **not** hold a standing G1; without the correctly adjusted
fall-arrest line, the robot can collapse before native PASSIVE takes over.

Enter from a motionless native stand (recommended) or PASSIVE while the robot
is already upright on its feet. The goal sequence contains longitudinal
displacements in metres; lateral, yaw, roll, and pitch goals remain fixed at
zero. Existing audit paths are refused. For example, the first zero-goal stage
is:

```bash
python scripts/g1_jump_deploy/hardware/run_fsm_g1.py \
  NETWORK_INTERFACE --ground_jump --enable_control \
  --entry_mode native_stand --exit_mode native_walkrun --duration 30 \
  --effort_scale 0.3 --goal_sequence "0.0" --interactive_goals \
  --blend_in_duration_s 0.0 --blend_out_duration_s 5.0 \
  --stand_hold_duration_s 1.0 \
  --shadow_admission logs/hardware_shadow/upright_20260827_v2_admission.json \
  --ground_log logs/hardware_ground/zero_FIRST_RUN.npz \
  --acknowledge_unmeasured_ground_jump
```

The sequence is consumed first. A positive blend-in uses frozen-phase policy
preparation while retaining independent balance; zero arms directly. The
completed policy blends back to the validated stand before the quiet stand hold.
After that hold, interactive mode prints `READY: next goal dx (or q) ->` and
reads stdin on a background thread, so
the 500 Hz command loop never waits for the terminal. Stdin EOF only disables
interactive prompts: it is never interpreted as `q` or as an abort, and queued
`--goal_sequence` entries remain available. Only an explicit `q` line or B ends
the session. After `READY`, tap and release A to arm; after the FSM prints
`ARMED` and `CONFIRM NOW`, tap and release Y. The NPZ records the command stream
and, for every jump, its goal, policy-step bounds, outcome, latched abort reason,
maximum tilt, and maximum estimated torque fraction.

The same post-takeoff behavior can be replayed in MuJoCo with
`--unmeasured_ground_validation --latched_abort_upright_settle` before any
hardware session.
