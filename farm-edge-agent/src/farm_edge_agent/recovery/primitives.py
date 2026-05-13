"""The five recovery primitives plus a name-to-callable Registry.

Every primitive is shaped ``(driver, safety, **kwargs) -> RecoveryResult`` and
emits a single ``RecoveryEvent`` via the optional ``sink``. Motion always uses
``safety.velocity_cap`` so the velocity cap cannot be bypassed by reaching for
the recovery layer instead of the regular control path.
"""

from __future__ import annotations

from collections.abc import Callable

from . import (
    AbortedRunError,
    Driver,
    EventSink,
    Perception,
    Pose,
    RecoveryEvent,
    RecoveryResult,
    Safety,
)


def _emit(sink: EventSink | None, primitive: str, detail: dict | None = None) -> None:
    if sink is None:
        return
    sink(RecoveryEvent(primitive=primitive, detail=detail or {}))


def home(
    driver: Driver,
    safety: Safety,
    *,
    sink: EventSink | None = None,
) -> RecoveryResult:
    """Move to ``safety.home_pose`` and open the gripper."""

    driver.move_to(safety.home_pose, safety.velocity_cap)
    driver.set_gripper("open")
    _emit(sink, "home")
    return RecoveryResult(primitive="home", ok=True)


def open_gripper(
    driver: Driver,
    safety: Safety,  # noqa: ARG001 — kept for a uniform (driver, safety) call shape
    *,
    sink: EventSink | None = None,
) -> RecoveryResult:
    """Release whatever the gripper is holding."""

    driver.set_gripper("open")
    _emit(sink, "open_gripper")
    return RecoveryResult(primitive="open_gripper", ok=True)


def relocalize(
    driver: Driver,
    perception: Perception,
    *,
    sink: EventSink | None = None,
) -> RecoveryResult:
    """Capture fresh frames and re-read pose so the next backend starts fresh."""

    frames = perception.capture()
    tcp_pose = driver.read_tcp_pose()
    joint_state = driver.read_joint_state()
    detail = {
        "frames": frames,
        "tcp_pose": list(tcp_pose),
        "joint_state": list(joint_state),
    }
    _emit(sink, "relocalize", detail)
    return RecoveryResult(primitive="relocalize", ok=True, detail=detail)


def retry_grasp(
    driver: Driver,
    safety: Safety,
    last_tcp: Pose,
    *,
    sink: EventSink | None = None,
) -> RecoveryResult:
    """Re-attempt the last grasp from the current TCP pose."""

    driver.move_to(last_tcp, safety.velocity_cap)
    driver.set_gripper("closed")
    _emit(sink, "retry_grasp", {"last_tcp": list(last_tcp)})
    return RecoveryResult(primitive="retry_grasp", ok=True)


def abort_safely(
    driver: Driver,
    safety: Safety,
    *,
    sink: EventSink | None = None,
) -> RecoveryResult:
    """Descend to nearest in-envelope waypoint, open gripper, disarm watchdog.

    Terminal: a second call raises ``AbortedRunError`` because the watchdog
    has already been disarmed for this run.
    """

    if not safety.watchdog_armed:
        raise AbortedRunError(
            "abort_safely is terminal; watchdog already disarmed for this run"
        )
    current = driver.read_tcp_pose()
    safe_pose = safety.clamp_to_envelope(current)
    driver.move_to(safe_pose, safety.velocity_cap)
    driver.set_gripper("open")
    safety.disarm_watchdog()
    _emit(sink, "abort_safely", {"safe_pose": list(safe_pose)})
    return RecoveryResult(primitive="abort_safely", ok=True)


PrimitiveCallable = Callable[..., RecoveryResult]


class Registry:
    """Maps the primitive names a capability card lists to their callables.

    The Dispatcher consumes ``recovery_chain: ["home", "relocalize"]`` and calls
    ``registry.get(name)(...)`` for each entry. Lookup is exact-match; an
    unknown name raises ``KeyError`` so a typo in a card fails loudly rather
    than silently skipping the recovery step.
    """

    def __init__(
        self, primitives: dict[str, PrimitiveCallable] | None = None
    ) -> None:
        self._primitives: dict[str, PrimitiveCallable] = (
            dict(primitives) if primitives is not None else dict(_DEFAULTS)
        )

    def register(self, name: str, fn: PrimitiveCallable) -> None:
        self._primitives[name] = fn

    def get(self, name: str) -> PrimitiveCallable:
        try:
            return self._primitives[name]
        except KeyError as e:
            raise KeyError(f"unknown recovery primitive: {name!r}") from e

    def names(self) -> list[str]:
        return list(self._primitives.keys())


_DEFAULTS: dict[str, PrimitiveCallable] = {
    "home": home,
    "open_gripper": open_gripper,
    "relocalize": relocalize,
    "retry_grasp": retry_grasp,
    "abort_safely": abort_safely,
}


__all__ = [
    "PrimitiveCallable",
    "Registry",
    "abort_safely",
    "home",
    "open_gripper",
    "relocalize",
    "retry_grasp",
]
