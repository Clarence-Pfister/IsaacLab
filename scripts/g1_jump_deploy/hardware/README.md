<!--
Copyright (c) 2022-2026, The Isaac Lab Project Developers
(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
-->

# G1 jump hardware preflight

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

G1 `LowState` has no root-position or foot-contact fields. Consequently, no
mode here authorizes a ground jump: the normal motor-control path remains
stand-only and retains its measured-contact arming gate. Do not bypass that
gate or substitute zero contact values for measurements.

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
