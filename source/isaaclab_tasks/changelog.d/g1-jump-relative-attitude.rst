Added
^^^^^

* Added a translation-first G1 jump curriculum with deployable trigger-relative
  attitude feedback for closed-loop heading control.
* Added a contact-randomized translation stage with a standard-G1 torque margin
  for sim-to-real policy hardening.

Fixed
^^^^^

* Fixed the G1 jump yaw-rate reward so natural jump pitch and roll rates no
  longer erased its heading-control signal.
