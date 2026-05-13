from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolVersion:
    """Wire-protocol semver exchanged on the Edge Agent ↔ Dispatcher handshake.

    Decoupled from package versions. Major must match for two peers to talk.
    """

    major: int
    minor: int
    patch: int

    def is_compatible_with(self, other: "ProtocolVersion") -> bool:
        return self.major == other.major

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


CURRENT_PROTOCOL: ProtocolVersion = ProtocolVersion(1, 2, 0)
