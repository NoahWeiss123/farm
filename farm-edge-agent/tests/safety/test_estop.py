from dataclasses import dataclass

from farm_edge_agent.safety.estop import EstopCheck


@dataclass
class FakeDriver:
    armed: bool

    def is_estop_armed(self) -> bool:
        return self.armed


def test_armed_estop_passes() -> None:
    assert EstopCheck(FakeDriver(armed=True)).check().ok


def test_unarmed_estop_blocks_run() -> None:
    result = EstopCheck(FakeDriver(armed=False)).check()
    assert not result.ok
    assert result.event is not None
    assert result.event.code == "FARM-E3004"
    assert result.event.severity == "violation"
