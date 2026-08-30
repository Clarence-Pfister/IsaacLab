Added
^^^^^

* Added a MuJoCo backend and reproducible scenario runner for exercising the G1
  jump controller state machine end to end.
* Added a viewer-enabled stand-only scenario that does not load or execute the
  jump policy.

Fixed
^^^^^

* Fixed the simulated load-bearing gantry to leave commanded horizontal motion
  unrestrained while supporting height and roll-pitch attitude.
