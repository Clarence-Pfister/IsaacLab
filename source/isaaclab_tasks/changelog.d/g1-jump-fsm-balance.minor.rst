Added
^^^^^

* Added measured ankle-balance feedback to standing states and a residual-offset
  interlock before jump arming, while preserving exclusive policy ankle control
  during jumps.

Fixed
^^^^^

* Fixed the jump controller's pre-arm tilt interlock to use the balance target
  attitude and respected each deployment manifest's flight-freeze setting.
* Fixed the jump controller's balance feedback to run in the 500 Hz actuator
  loop while the state machine continues to update base targets at 50 Hz.
* Fixed the start-pose transition to interpolate from the held base target
  without adding the measured balance correction a second time.
* Fixed stand entry to engage balance immediately and allowed balance-actuated
  ankle offsets in the pre-arm start-pose check.
