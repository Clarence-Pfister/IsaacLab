Added
^^^^^

* Added a G1 jump policy exporter that emits validated TorchScript and ONNX policies,
  resolved deployment metadata, and precomputed reference observation tables.
* Added velocity-aware lower-limit braking to the G1 jump deployment contract so
  fast knee extension retains a command margin from the mechanical stop.
* Added physical joint limits separately from action-target limits in deployment
  manifests so hardware feedback checks do not reject valid tracking error.

Fixed
^^^^^

* Fixed reordered G1 articulation joints being reported as an undeployable bundle
  after their name and SDK-slot mappings had been validated.
* Fixed bounded G1 jump policies failing TorchScript export because the deterministic
  action transform referenced a Python class constant.
