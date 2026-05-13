from __future__ import annotations

import json as _json
import sys
from enum import Enum
from typing import Any, NoReturn

from farm_shared.errors import ErrorCode, format_error


class Severity(Enum):
    CONFIGURATION = "configuration"
    RUNTIME = "runtime"


_SEVERITY: dict[ErrorCode, Severity] = {
    ErrorCode.E1001: Severity.CONFIGURATION,
    ErrorCode.E1002: Severity.CONFIGURATION,
    ErrorCode.E1003: Severity.RUNTIME,
    ErrorCode.E1004: Severity.CONFIGURATION,
    ErrorCode.E1005: Severity.RUNTIME,
    ErrorCode.E1006: Severity.CONFIGURATION,
    ErrorCode.E1007: Severity.RUNTIME,
    ErrorCode.E2001: Severity.CONFIGURATION,
    ErrorCode.E3001: Severity.RUNTIME,
    ErrorCode.E3002: Severity.RUNTIME,
}


_EXIT_CODES: dict[Severity, int] = {
    Severity.CONFIGURATION: 2,
    Severity.RUNTIME: 1,
}


_DOCS_URL_BASE = "https://farm.dev/errors/"
_FIX_SEPARATOR = " — fix: "


class FarmError(Exception):
    """Structured error in the FARM catalog.

    Carries an :class:`ErrorCode` plus the named slot values its template needs.
    Stringifies to the canonical ``[FARM-Exxxx] ...`` format via
    :func:`farm_shared.errors.format_error`.
    """

    def __init__(self, code: ErrorCode, **slots: Any) -> None:
        self.code = code
        self.slots: dict[str, Any] = dict(slots)
        super().__init__(format_error(code, **slots))

    def __str__(self) -> str:
        return format_error(self.code, **self.slots)

    @property
    def severity(self) -> Severity:
        return _SEVERITY[self.code]

    @property
    def exit_code(self) -> int:
        return _EXIT_CODES[self.severity]

    @property
    def docs_url(self) -> str:
        return f"{_DOCS_URL_BASE}{self.code.docs_url_slug}"

    def to_dict(self) -> dict[str, Any]:
        text = str(self)
        prefix = f"[FARM-{self.code.name}] "
        body = text[len(prefix):]
        if _FIX_SEPARATOR in body:
            message, fix = body.split(_FIX_SEPARATOR, 1)
            fix_value: str | None = fix
        else:
            message = body
            fix_value = None
        return {
            "code": f"FARM-{self.code.name}",
            "message": message,
            "fix": fix_value,
            "docs_url": self.docs_url,
        }


def emit_to_cli(err: FarmError, json: bool) -> NoReturn:
    """Print ``err`` to stderr and exit with the severity-derived code.

    Text mode prints the canonical ``[FARM-Exxxx] ...`` line. JSON mode prints
    a single object with ``code``, ``message``, ``fix``, ``docs_url`` so wrappers
    can machine-read it.
    """
    if json:
        sys.stderr.write(_json.dumps(err.to_dict()) + "\n")
    else:
        sys.stderr.write(str(err) + "\n")
    sys.stderr.flush()
    sys.exit(err.exit_code)
