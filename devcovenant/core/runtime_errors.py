"""Typed error contracts for explicit DevCovenant runtime failures."""

from __future__ import annotations

from enum import Enum
from typing import Mapping


class ErrorCode(str, Enum):
    """Stable error-code taxonomy for command/runtime boundaries."""

    INVALID_ARGUMENT = "invalid-argument"
    MANAGED_ENVIRONMENT = "managed-environment"
    COMMAND_RUNTIME = "command-runtime"
    INTERNAL_ERROR = "internal-error"


def _normalize_exit_code(raw_value: object) -> int:
    """Return a normalized non-zero process exit code for error paths."""
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 1
    if value <= 0:
        return 1
    return value


class DevCovenantError(RuntimeError):
    """Structured error with stable code, optional hint, and exit code."""

    def __init__(
        self,
        *,
        code: ErrorCode,
        message: str,
        hint: str = "",
        exit_code: int = 1,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize one explicit DevCovenant error payload."""
        normalized_message = str(message).strip() or "Unknown error."
        super().__init__(normalized_message)
        self.code = code
        self.message = normalized_message
        self.hint = str(hint).strip()
        self.exit_code = _normalize_exit_code(exit_code)
        self.details = dict(details or {})

    def to_display_message(self) -> str:
        """Render deterministic user-facing error text."""
        lines = [f"Error [{self.code.value}]: {self.message}"]
        if self.hint:
            lines.append(f"Hint: {self.hint}")
        return "\n".join(lines)


_DEFAULT_INTERNAL_HINT = (
    "Inspect run logs for traceback details and failing command context."
)


def normalize_unhandled_exception(
    error: BaseException,
) -> DevCovenantError:
    """Normalize one unexpected runtime exception into typed error form."""
    if isinstance(error, DevCovenantError):
        return error

    if isinstance(error, ValueError):
        return DevCovenantError(
            code=ErrorCode.INVALID_ARGUMENT,
            message=str(error).strip() or "Invalid argument.",
            hint="Review command arguments and configuration values.",
            exit_code=2,
        )

    if isinstance(error, OSError):
        return DevCovenantError(
            code=ErrorCode.COMMAND_RUNTIME,
            message=str(error).strip() or "Command runtime failure.",
            hint=(
                "Verify file paths, permissions, and managed-environment "
                "state."
            ),
            exit_code=1,
        )

    message = str(error).strip()
    if message:
        message = f"Unexpected runtime failure: {message}"
    else:
        message = "Unexpected runtime failure."
    return DevCovenantError(
        code=ErrorCode.INTERNAL_ERROR,
        message=message,
        hint=_DEFAULT_INTERNAL_HINT,
        exit_code=1,
    )


def render_error(error: DevCovenantError) -> str:
    """Render one typed error for console output."""
    return error.to_display_message()


__all__ = [
    "DevCovenantError",
    "ErrorCode",
    "normalize_unhandled_exception",
    "render_error",
]
