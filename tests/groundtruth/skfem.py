"""Ground-truth probes for scikit-fem.

scikit-fem ships every public element and mesh class on the top-level
``skfem`` module, so the source-of-truth check is straightforward Python
introspection: the catalog must not promise an ``Element*`` or ``Mesh*``
name that does not exist in the installed package.

The probes return ``None`` when scikit-fem is not importable so the test
suite can skip rather than fail in environments without it.

This module is the canonical example of the *Python-introspection* probe
family.  The same pattern transfers to FEniCSx/dolfinx, NGSolve, Kratos
and DUNE-fem — each exposes its catalog-relevant names as attributes on
a top-level package, and ``scripts/fingerprint_solvers.py`` already
captures most of those for drift-vs-prior-fingerprint comparisons.
"""

from __future__ import annotations

from typing import Set


def _classes_with_prefix(prefix: str) -> Set[str] | None:
    """Return the names of public classes whose name starts with ``prefix``
    on the ``skfem`` module, or ``None`` if scikit-fem is not installed."""
    try:
        import inspect

        import skfem  # type: ignore
    except ImportError:
        return None
    return {
        name
        for name in dir(skfem)
        if name.startswith(prefix)
        and not name.startswith("_")
        and inspect.isclass(getattr(skfem, name))
    }


def element_classes() -> Set[str] | None:
    """Public ``Element*`` classes exposed by ``skfem`` (~80 in 12.0)."""
    return _classes_with_prefix("Element")


def mesh_classes() -> Set[str] | None:
    """Public ``Mesh*`` classes exposed by ``skfem`` (~20 in 12.0)."""
    return _classes_with_prefix("Mesh")
