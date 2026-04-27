"""f3dx[atlas-rtc] verify: controlled_completion wraps Cruz's
ATLAS-RTC RuntimeController under f3dx's observability surface.

Three scenarios via MockAdapter (no GPU, no vLLM): a clean first-pass
generation, a generation where the controller intervenes to keep the
output on contract, and a deliberate-failure case so callers know how
the not-valid path looks.
"""

from __future__ import annotations

from atlas_rtc.adapters.mock_adapter import MockAdapter, MockScenario

import f3dx
from f3dx.atlas_rtc import controlled_completion


def _scenario(planned: list[str], candidates: list[dict[str, float]]) -> MockScenario:
    return MockScenario(prompt="", planned_tokens=planned, candidates=candidates)


def case_clean_first_pass() -> None:
    print("-- clean first-pass: model emits valid JSON without intervention --")
    scen = _scenario(
        planned=["{", '"name"', ":", '"alice"', ",", '"age"', ":", "30", "}"],
        candidates=[
            {"{": 2.0},
            {'"name"': 2.0},
            {":": 2.0},
            {'"alice"': 2.0},
            {",": 2.0},
            {'"age"': 2.0},
            {":": 2.0},
            {"30": 2.0},
            {"}": 2.0},
        ],
    )
    out = controlled_completion(
        prompt="Return JSON with name and age.",
        contract=["name", "age"],
        adapter=MockAdapter(scen),
    )
    print(f"  text={out.text!r}")
    print(f"  valid={out.valid} interventions={out.interventions} first_pass={out.succeeded_first_pass}")


def case_drift_with_intervention() -> None:
    print("-- drift: candidate token list has tempting wrong-key alternatives --")
    scen = _scenario(
        planned=["{", '"name"', ":", '"alice"', ",", '"age"', ":", "30", "}"],
        candidates=[
            {"Hello": 2.0, "{": 1.1},
            {'"name"': 2.0, '"junk"': 1.8},
            {":": 2.4, "-": 0.3},
            {'"alice"': 2.0, '"bob"': 1.2},
            {",": 2.0, "}": 0.9},
            {'"age"': 2.1, '"city"': 1.9},
            {":": 2.5},
            {"30": 2.2, '"unknown"': 1.0},
            {"}": 2.4},
        ],
    )
    out = controlled_completion(
        prompt="Return JSON with name and age.",
        contract=["name", "age"],
        adapter=MockAdapter(scen),
    )
    print(f"  text={out.text!r}")
    print(f"  valid={out.valid} interventions={out.interventions} first_pass={out.succeeded_first_pass}")


def case_unrecoverable_failure() -> None:
    print("-- contract violation: scenario won't ever produce required key --")
    scen = _scenario(
        planned=["{", '"only_name"', ":", '"alice"', "}"],
        candidates=[
            {"{": 2.0},
            {'"only_name"': 2.0},
            {":": 2.0},
            {'"alice"': 2.0},
            {"}": 2.0},
        ],
    )
    out = controlled_completion(
        prompt="Return JSON with name AND age.",
        contract=["name", "age"],
        adapter=MockAdapter(scen),
    )
    print(f"  text={out.text!r}")
    print(f"  valid={out.valid} errors={out.errors}")


def main() -> None:
    print(f"f3dx version: {f3dx.__version__}\n")
    case_clean_first_pass()
    print()
    case_drift_with_intervention()
    print()
    case_unrecoverable_failure()
    print()
    print("OK — f3dx[atlas-rtc] composition verified")
    print("composes: f3dx (transport + observability) over Cruz's ATLAS-RTC (decode-time control)")


if __name__ == "__main__":
    main()
