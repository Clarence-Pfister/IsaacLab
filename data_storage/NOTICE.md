# Third-party notices for `data_storage/`

This directory redistributes robot description data that originates outside this project.
The notices below are provided to satisfy the attribution terms of the upstream licenses.
Everything else in this repository is covered by the top-level [`LICENSE`](../LICENSE).

## Unitree G1 robot description

**Files:** `mesh/*.STL` (60 collision and visual meshes) and `g1_23dof_holo_compat.xml`.

These are derived from the Unitree G1 robot description published by Unitree Robotics in
[`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros) under
`robots/g1_description/`.

```
Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
All rights reserved.

SPDX-License-Identifier: BSD-3-Clause
```

The full license text is reproduced at
[`docs/licenses/assets/unitree-license.txt`](../docs/licenses/assets/unitree-license.txt).

**Modifications:** `g1_23dof_holo_compat.xml` is a modified 23-DoF variant prepared for this
project; it is not the upstream file unchanged. The meshes are redistributed as published.

## Generated USD

**Files:** `g1_23dof_holo_compat/**`.

Converted from `g1_23dof_holo_compat.xml` by the Isaac Lab `MjcfConverter`; the conversion
settings are recorded in [`config.yaml`](config.yaml). As a derivative of the Unitree
description, this content carries the same BSD 3-Clause terms as the source MJCF.

## Jump reference motion

**Files:** `perfect_jump_processed.csv`.

The reference jump was generated from a text prompt with
[Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/), NVIDIA Research's kinematic
motion diffusion model, using the `Kimodo-SOMA-RP` checkpoint, and retargeted from the SOMA
skeleton to the 23-DoF G1 with NVIDIA's [SOMA retargeter](https://github.com/NVIDIA/soma-retargeter).
The trajectories derived from it by post-processing, and the tooling that produces them, live on
`feature/g1-jump-repeat-flow` and `integration/all`.

Both upstream components permit commercial use and redistribution of what they produce:

* The Kimodo SOMA checkpoint is released under the
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/),
  and was trained on the proprietary, commercially licensed Bones Rigplay motion-capture dataset —
  not on a research-only motion corpus. Under that license NVIDIA "claims no ownership rights in
  outputs", and no attribution is required when redistributing generated motion.
* The [SOMA body model](https://github.com/NVlabs/SOMA-X) and the SOMA retargeter are released
  under the Apache License 2.0.

SOMA is a unifying framework that maps several parametric body models onto one shared rig; SMPL-X
is one optional identity model within it and was **not** used in this pipeline. No Max Planck
SMPL-X model assets, which carry non-commercial research terms, are part of the chain that
produced these files.

These CSVs are therefore covered by the top-level [`LICENSE`](../LICENSE) like the rest of this
project. The provenance above is recorded for reproducibility and credit, not to satisfy a
license condition.
