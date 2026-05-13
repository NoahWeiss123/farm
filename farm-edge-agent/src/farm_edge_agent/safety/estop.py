"""Pre-run check that the hardware e-stop circuit is armed."""

from __future__ import annotations

from typing import Protocol

from . import CheckResult, SafetyEvent


class SupportsEstop(Protocol):
    def is_estop_armed(self) -> bool: ...


class EstopCheck:
    """Refuse to start a run if the driver reports the e-stop circuit unarmed."""

    def __init__(self, driver: SupportsEstop) -> None:
        self._driver = driver

    def check(self) -> CheckResult:
        if not self._driver.is_estop_armed():
            return CheckResult.fail(
                SafetyEvent(
                    kind="estop_not_armed",
                    severity="violation",
                    code="FARM-E3004",
                    message="e-stop circuit not detected; refusing to start run",
                )
            )
        return CheckResult.pass_()
