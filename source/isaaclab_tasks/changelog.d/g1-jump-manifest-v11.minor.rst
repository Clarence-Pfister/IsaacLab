Changed
^^^^^^^

* **Breaking:** Changed G1 jump deployment manifests to schema 1.1 with reference
  motion provenance and a frame-0 root pose. Re-export existing deployment bundles
  before using the updated deployment tools.

Deprecated
^^^^^^^^^^

* Deprecated the ``--reference_csv`` deployment option. Re-export the deployment
  bundle and use its declared reference source instead.
