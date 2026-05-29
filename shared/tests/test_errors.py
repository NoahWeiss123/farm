import string

from farm_shared.errors import ErrorCode, format_error


def _slots_in(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def test_format_e1001_camera_missing_matches_design_md():
    msg = format_error(ErrorCode.E1001, device="/dev/video0")
    assert msg == (
        "[FARM-E1001] No camera found at /dev/video0 — fix: "
        "'farm doctor cameras', then 'farm config set camera.wrist.device /dev/videoN'"
    )


def test_codes_are_unique():
    codes = [e.code for e in ErrorCode]
    assert len(codes) == len(set(codes))


def test_names_are_unique():
    names = [e.name for e in ErrorCode]
    assert len(names) == len(set(names))


def test_every_template_can_render_with_its_slots():
    for err in ErrorCode:
        slots = _slots_in(err.template)
        stubs = {s: f"<{s}>" for s in slots}
        rendered = format_error(err, **stubs)
        assert rendered.startswith(f"[FARM-{err.name}] ")
        for s in slots:
            assert f"<{s}>" in rendered


def test_format_e1006_version_mismatch():
    msg = format_error(ErrorCode.E1006, agent_version="1.0", required_version="1.2")
    assert msg == (
        "[FARM-E1006] Edge Agent v1.0 detected, Dispatcher requires v1.2+ "
        "— fix: 'pip install -U farm-edge-agent'"
    )


def test_format_e3001_no_slots():
    msg = format_error(ErrorCode.E3001)
    assert msg == (
        "[FARM-E3001] Safety envelope violation: commanded pose outside workspace. "
        "Soft-stopped."
    )


def test_every_code_has_docs_slug():
    for err in ErrorCode:
        assert err.docs_url_slug == err.name
