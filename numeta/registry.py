from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .numeta_function import NumetaCompilationTarget

_registry: list["NumetaCompilationTarget"] = []


def register_compilation_target(target: "NumetaCompilationTarget") -> None:
    _registry.append(target)


def registered_functions() -> list["NumetaCompilationTarget"]:
    """Return a snapshot of all instantiated :class:`NumetaFunction` objects."""
    return list(_registry)


def clear_registered_functions() -> None:
    """Clear the registry of instantiated :class:`NumetaFunction` objects."""
    _registry.clear()
