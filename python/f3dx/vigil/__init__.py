"""f3dx.vigil — bridge between f3dx JSONL traces and VIGIL/Robin B's
event log shape.

VIGIL ([cruz209/V.I.G.I.L](https://github.com/cruz209/V.I.G.I.L)) is
Christopher Cruz's reflective-runtime supervisor. It reads an
`events.jsonl` log (each line: `ts`/`actor`/`kind`/`status`/`payload`/
`note`/`source`) of a sibling agent's runtime behaviour, builds an
"emotional bank" (Roses / Buds / Thorns appraisal), diagnoses
reliability issues, and proposes prompt + code adaptations.

f3dx is the perf-fast-path runtime that the supervised agent runs over.
This module converts the JSONL trace f3dx emits into VIGIL's event log
shape so a `Robin B` reflection cycle can run over real f3dx-driven
agent runs.

    pip install f3dx[vigil]

Usage:

    f3dx.configure_traces("traces.jsonl", capture_messages=True)
    # ... agent runs ...
    from f3dx.vigil import f3dx_jsonl_to_vigil_events
    f3dx_jsonl_to_vigil_events("traces.jsonl", "vigil_events.jsonl",
                                actor="robin_a")
    # then point Robin B at vigil_events.jsonl
"""

from f3dx.vigil._bridge import f3dx_jsonl_to_vigil_events, f3dx_row_to_vigil_events

__all__ = ["f3dx_jsonl_to_vigil_events", "f3dx_row_to_vigil_events"]
