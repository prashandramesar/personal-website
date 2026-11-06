"""Checks the simulator's calculations in a real browser.

Needs Playwright and Chromium, so it's skipped in CI. To run it locally:

    pip install playwright
    playwright install chromium
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

MODEL = Path(__file__).parent.parent / "static" / "js" / "simulator_model.js"
BASE = {
    "calls": 50000,
    "adoptionPct": 50,
    "minutesSaved": 0.75,
    "wrongPct": 12,
    "extraMinutes": 1.5,
    "hourlyCost": 30,
    "runningCost": 6000,
}


@pytest.fixture(scope="module")
def page() -> Iterator[Any]:
    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        browser_page = browser.new_page()
        browser_page.add_script_tag(path=str(MODEL))
        yield browser_page
        browser.close()


def calculate(page: Any, **changes: float) -> dict[str, Any]:
    return page.evaluate(
        """inputs => {
            const results = SimulatorModel.calculate(inputs);
            return {
                results,
                stage: SimulatorModel.assess(inputs, results).stage.id,
                breakEven: SimulatorModel.breakEvenAdoptionPct(inputs),
            };
        }""",
        {**BASE, **changes},
    )


def test_calculates_known_values(page: Any) -> None:
    results = calculate(page)["results"]

    assert results["assistedCalls"] == 25000
    assert results["minutesSaved"] == 18750
    assert results["minutesLost"] == 4500
    assert results["netHours"] == pytest.approx(237.5)
    assert results["staffSaving"] == pytest.approx(7125)
    assert results["netMonthly"] == pytest.approx(1125)


def test_break_even_is_where_the_net_result_reaches_zero(page: Any) -> None:
    adoption = calculate(page)["breakEven"]

    assert calculate(page, adoptionPct=adoption)["results"][
        "netMonthly"
    ] == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(
    "changes",
    [{"minutesSaved": 0}, {"wrongPct": 50, "extraMinutes": 5}, {"calls": 0}],
)
def test_no_break_even_when_calls_add_no_value(
    page: Any, changes: dict[str, float]
) -> None:
    assert calculate(page, **changes)["breakEven"] is None


@pytest.mark.parametrize(
    ("changes", "stage"),
    [
        ({}, "pilot"),
        ({"wrongPct": 30}, "drawing-board"),
        ({"wrongPct": 30, "minutesSaved": 3}, "drawing-board"),
        ({"adoptionPct": 15, "wrongPct": 8}, "fix-experience"),
        ({"runningCost": 30000}, "hold"),
        ({"adoptionPct": 75, "wrongPct": 6, "minutesSaved": 1}, "scale"),
    ],
)
def test_rollout_stage(page: Any, changes: dict[str, float], stage: str) -> None:
    assert calculate(page, **changes)["stage"] == stage
