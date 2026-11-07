"""Project exception type that captures where the error occurred."""

import sys


def _detail(error: Exception, exc_info) -> str:
    _, _, tb = exc_info.exc_info()
    if tb is None:
        return str(error)
    fname = tb.tb_frame.f_code.co_filename
    return f"Error in [{fname}] line [{tb.tb_lineno}]: {error}"


class VertebralError(Exception):
    """Wrap any exception with file/line context for clearer logs."""

    def __init__(self, error: Exception, exc_info=sys):
        super().__init__(str(error))
        self.message = _detail(error, exc_info)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message
