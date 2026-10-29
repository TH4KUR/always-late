# Gemma-2-9B: one maintained branch

Updated 2026-10-29. Use **`gemma9b-confirmation`** for all 9B work. The former adapter and v2 branches are consolidated into this branch; their commits remain in its history.

- **Completed:** original-RTN functional validation and Study G, [reported results](verdict.md), from commit `0c3d3c7`. Its [execution freeze](legacy_rtn_run_freeze_2026-10-29.json) is preserved byte for byte. Numbers are reported by the original runner; they have not been independently reproduced from raw artifacts here.
- **Prepared, results missing:** GPTQ and matched-RTN comparisons, with the existing [active freeze](freeze.json), [A1](amendment_A1_three_arm.json), implementation and tests preserved from the former v2 branch. The active freeze and adapter stay unchanged; the completed run used another adapter revision, retained in commit history.
- **Budget:** 3 Jarvis GPU-hours remaining plus Ada. The full driver is not a measured recovery plan for that limit. No GPU job or paid machine was started by consolidation.
- [Execution handoff](JARVIS_HANDOFF.md) contains the canonical branch name and the current status notice. Older plans and handoffs retain historical branch names.

Consolidation parents: `9a0864f` (GPTQ implementation) and `0c3d3c7` (legacy-RTN results). The adapter branch `8b6fb5e` is already part of the former v2 history.
