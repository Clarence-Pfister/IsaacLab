Added
^^^^^

* Added a shared NumPy runtime for G1 jump deployment observations and action
  transforms.

Changed
^^^^^^^

* Changed the G1 jump actor to latch its relative landing goal at trigger time,
  removing the unavailable low-level root-odometry dependency. Custom tasks with
  external localization should explicitly select live goal feedback instead.
* Changed Stage 2 G1 jump yaw rewards to discriminate commanded turns. Retrain
  policies whose yaw behavior was learned with the previous nearly flat kernels.
