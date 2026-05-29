"""Lazy import wrapper for ``xarm.wrapper.XArmAPI``.

The xArm Python SDK is an optional install (``pip install
farm-edge-agent[arm]``). Keeping the import behind a function lets the rest of
the package — and its test suite — be importable on machines without the SDK.
Tests patch :func:`XArmAPI` to bypass the real SDK entirely.
"""

from __future__ import annotations

from typing import Any

_INSTALL_FIX = "pip install 'farm-edge-agent[arm]'"
_SDK_MISSING_MESSAGE = (
    "xArm Python SDK is not installed — fix: " + _INSTALL_FIX
)


class XArmSDKMissingError(Exception):
    """Raised when ``xarm.wrapper.XArmAPI`` cannot be imported.

    Shaped like a ``FarmError`` (``.code`` attribute, canonical
    ``[FARM-Exxxx] ...`` string) so callers can treat it uniformly. The error
    code is a placeholder until a dedicated slot lands in the shared catalog;
    see ``tasks/_followups.md``.
    """

    code: str = "FARM-E1010"

    def __init__(self) -> None:
        super().__init__(f"[{self.code}] {_SDK_MISSING_MESSAGE}")


def XArmAPI(ip: str, **kwargs: Any) -> Any:
    """Instantiate the real xArm SDK client at ``ip``.

    Raises :class:`XArmSDKMissingError` if the SDK is not importable, so the
    edge agent can surface a clean install hint instead of a bare ``ImportError``.
    """
    try:
        from xarm.wrapper import XArmAPI as _XArmAPI
    except ImportError as e:
        raise XArmSDKMissingError() from e
    return _XArmAPI(ip, **kwargs)
