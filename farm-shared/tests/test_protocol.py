from farm_shared.protocol import CURRENT_PROTOCOL, ProtocolVersion


def test_current_protocol_is_1_2_0():
    assert CURRENT_PROTOCOL == ProtocolVersion(1, 2, 0)


def test_same_major_is_compatible():
    a = ProtocolVersion(1, 2, 0)
    b = ProtocolVersion(1, 2, 5)
    assert a.is_compatible_with(b)
    assert b.is_compatible_with(a)


def test_minor_drift_within_major_is_compatible():
    assert ProtocolVersion(1, 2, 0).is_compatible_with(ProtocolVersion(1, 9, 7))


def test_major_bump_breaks_compatibility():
    a = ProtocolVersion(1, 2, 0)
    b = ProtocolVersion(2, 0, 0)
    assert not a.is_compatible_with(b)
    assert not b.is_compatible_with(a)


def test_str_renders_dotted_semver():
    assert str(ProtocolVersion(1, 2, 0)) == "1.2.0"
