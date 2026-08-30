Added
^^^^^

* Added a simulator-agnostic state machine with safety interlocks and smooth
  gain transitions for deploying the G1 jump policy.
* Added read-only ONNX inference checks on live G1 feedback without creating a
  command publisher or locomotion client.
* Added a read-only full-policy shadow that records live G1 observations,
  inference timing, guarded counterfactual targets, and torque projections
  without creating a command publisher or locomotion client.
* Added accepted-artifact digest verification before the G1 hardware boundary
  opens a Unitree channel.
* Added offline replay validation and read-only admission evidence for the
  three-command live G1 policy-shadow matrix.

Fixed
^^^^^

* Fixed the first jump-policy action being delayed by one controller period
  after operator confirmation.
* Fixed loaded start-pose deflection causing a knee-limit abort by carrying a
  bounded static-load target offset through the arming pose.
* Fixed arming while a joint was moving faster than the configured pre-arm
  velocity threshold.
