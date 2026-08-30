Fixed
^^^^^

* Fixed the MuJoCo G1 jump harness to match Isaac's actuator and passive-force
  configuration by using implicit position-actuator PD and removing passive
  joint damping and Coulomb joint friction absent from PhysX.
