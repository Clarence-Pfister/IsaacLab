Added
^^^^^

* Added ``--delay_steps``, ``--refresh_frame0_state`` and ``--joint_velocity_limit`` to the
  closed-loop Isaac G1 jump rollout logger so a sim-to-sim comparison can match the MuJoCo
  harness's action delay and frame-0 observation, and can measure what the solver-side joint
  velocity limit is worth.
* Added ``--clamp_joint_velocity`` to the MuJoCo G1 jump harness as a diagnostic probe of the
  joint velocity limit that PhysX enforces and MuJoCo does not. The clamp is applied to
  ``qvel`` after each step without the matching constraint reaction, so it is a coarse A/B
  probe rather than a fidelity-grade implementation.

Fixed
^^^^^

* Fixed the frame-0 observation of the Isaac G1 jump rollout loggers, which reported the
  pre-write base orientation because ``sim.forward()`` does not invalidate cached derived
  state such as ``projected_gravity_b``. Pass ``--refresh_frame0_state`` to recompute it
  from the written pose.
* Fixed the MuJoCo G1 jump manifest parser to retain ``actuators.velocity_limit`` instead of
  validating and discarding it, so the harness can act on the limit it reads.
