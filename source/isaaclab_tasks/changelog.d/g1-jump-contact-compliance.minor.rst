Added
^^^^^

* Added ``randomize_contact_compliance`` to the G1 jump events, a startup randomization term
  that samples a log-uniform contact spring stiffness per environment, derives the damping
  from a sampled damping ratio, and leaves a configurable fraction of the environments rigid.
  It is wired into ``G1JumpStage2DeployLongitudinalContactEnvCfg`` and
  ``G1JumpStage3DeployTranslationEnvCfg`` so the policy trains against a range of ground
  compliance instead of a single rigid contact.
* Added ``--contact_stiffness`` and ``--contact_damping`` to the closed-loop Isaac G1 jump
  rollout logger, and ``--contact_timeconst`` and ``--contact_dampratio`` to the MuJoCo
  harness, so the same contact-compliance sweep can be run in both engines.
* Added ``--force_manifest_dynamics`` and ``--policy_onnx`` to the Isaac G1 jump rollout
  logger so an exported deployment bundle can be replayed against the action scaling and
  actuator gains recorded in its own manifest, independently of the live task configuration.
