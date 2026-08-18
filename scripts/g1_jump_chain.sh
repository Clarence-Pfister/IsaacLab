#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Train the G1 jump policy through every stage in one pass.
#
# Each stage warm-starts from the last checkpoint of the one before it, which works because
# all four share an observation width and an experiment directory. Iteration counts are the
# ones the stages actually needed when they were developed separately.
#
# Run from the repository root inside the container:
#
#   ./scripts/g1_jump_chain.sh
#
# The chain is resumable. Stages whose run directory already exists are skipped, so after a
# crash or a deliberate stop the same command picks up at the stage that did not finish.
# Pass --from STAGE to force a restart at a given stage, and --dry-run to print the commands
# without running them.
#
# The chain writes to its own experiment directory rather than the shared g1_jump one. Runs
# from before the observation terms were moved into the body frame are 165 elements wide and
# would fail to load into the 164 of the current task, and keeping them apart also stops the
# newest-run checkpoint search from reaching across into the wrong lineage. Override with
# --experiment.

set -euo pipefail

EXPERIMENT="g1_jump_deploy"

# stage name | task id | iterations
STAGES=(
    "stage1|Isaac-Velocity-Jump-G1-v0|2000"
    "stage2|Isaac-Velocity-Jump-G1-Stage2-v0|6000"
    "stage2wide|Isaac-Velocity-Jump-G1-Stage2-Wide-v0|5000"
    "wideland|Isaac-Velocity-Jump-G1-Stage2-Wide-Land-v0|4000"
)

FROM=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --experiment) EXPERIMENT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

LOG_ROOT="logs/rsl_rl/${EXPERIMENT}"

# Newest run directory for a stage, by name suffix. Empty if the stage has not run.
find_run() {
    local name="$1"
    # shellcheck disable=SC2012
    ls -1d "${LOG_ROOT}"/*_"${name}" 2>/dev/null | sort | tail -1 || true
}

# Highest-numbered checkpoint in a run directory. Iterations sort numerically, not
# lexically, so model_9000 must not beat model_11000.
# Guarded against pipefail: with no checkpoints the leading ls fails, which would otherwise
# fail the whole pipeline and abort the script through set -e without printing anything.
find_checkpoint() {
    local run="$1" out
    # shellcheck disable=SC2012
    out="$( { ls -1 "${run}"/model_*.pt 2>/dev/null || true; } \
        | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' \
        | sort -n -k1,1 \
        | tail -1 \
        | cut -d' ' -f2 )" || true
    # must return success even when empty, or the caller's assignment trips set -e
    [[ -n "${out}" ]] && basename "${out}"
    return 0
}

run_stage() {
    local name="$1" task="$2" iters="$3" prev_run="$4"
    local -a cmd=(
        ./isaaclab.sh train
        --rl_library rsl_rl
        --task "${task}"
        --max_iterations "${iters}"
        --run_name "${name}"
        --experiment_name "${EXPERIMENT}"
        --viz none
    )

    if [[ -n "${prev_run}" ]]; then
        local ckpt
        ckpt="$(find_checkpoint "${prev_run}")"
        if [[ -z "${ckpt}" ]]; then
            if [[ "${DRY_RUN}" -eq 1 ]]; then
                ckpt="model_LAST.pt"  # the previous stage has not run, so there is none to name
            else
                echo "error: no checkpoint in ${prev_run}; cannot start ${name}" >&2
                exit 1
            fi
        fi
        cmd+=(--resume --load_run "$(basename "${prev_run}")" --checkpoint "${ckpt}")
        echo ">>> ${name}: ${iters} iters, resuming from $(basename "${prev_run}")/${ckpt}"
    else
        echo ">>> ${name}: ${iters} iters, from scratch"
    fi

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        printf '    %q ' "${cmd[@]}"; echo
        return
    fi
    "${cmd[@]}"
}

started=0
[[ -z "${FROM}" ]] && started=1
prev_run=""

for entry in "${STAGES[@]}"; do
    IFS='|' read -r name task iters <<<"${entry}"
    existing="$(find_run "${name}")"

    if [[ "${FROM}" == "${name}" ]]; then
        started=1
    fi

    if [[ "${started}" -eq 0 ]]; then
        # Before the requested start point: adopt the existing run as the parent.
        if [[ -z "${existing}" ]]; then
            echo "error: --from ${FROM} needs ${name} to have run already, and it has not" >&2
            exit 1
        fi
        prev_run="${existing}"
        echo "--- ${name}: reusing $(basename "${existing}")"
        continue
    fi

    if [[ -n "${existing}" && "${FROM}" != "${name}" ]]; then
        echo "--- ${name}: already present as $(basename "${existing}"), skipping"
        prev_run="${existing}"
        continue
    fi

    run_stage "${name}" "${task}" "${iters}" "${prev_run}"

    if [[ "${DRY_RUN}" -eq 1 ]]; then
        prev_run="${LOG_ROOT}/DRYRUN_${name}"
        continue
    fi

    prev_run="$(find_run "${name}")"
    if [[ -z "${prev_run}" ]]; then
        echo "error: ${name} produced no run directory under ${LOG_ROOT}" >&2
        exit 1
    fi
done

echo
echo "chain complete; final run: ${prev_run}"
