from __future__ import annotations


class OptionalModelDependencyError(RuntimeError):
    """Raised when an optional model family is selected without its runtime stack."""

