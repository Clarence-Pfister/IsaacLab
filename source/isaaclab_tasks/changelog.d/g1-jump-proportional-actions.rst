Added
^^^^^

* Added filtered Stage 1 and Stage 2 G1 jump curricula that progressively relax
  reference-target regularization before dynamics randomization.

Changed
^^^^^^^

* Changed the G1 jump action scales and actuator gains to preserve proportional
  control across the reference motion. Retrain existing G1 jump policies with
  the bounded action space.
* Changed G1 jump training to penalize normalized joint targets that depart from
  the reference motion. Retrain policies that learned to rely on actuator effort
  clipping instead of reference-like position targets.
