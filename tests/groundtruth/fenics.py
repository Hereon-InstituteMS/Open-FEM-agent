"""Ground-truth probes for FEniCSx / dolfinx.

FEniCSx exposes its API as a Python package tree rooted at
``dolfinx``: mesh constructors under ``dolfinx.mesh``, function-space
machinery under ``dolfinx.fem``, IO under ``dolfinx.io``, PETSc
integration under ``dolfinx.fem.petsc``, and so on.  The catalog
references these by their dotted path (``dolfinx.mesh.create_rectangle``,
``dolfinx.fem.functionspace``, ``dolfinx.io.XDMFFile``), so the
verification is a per-path attribute check.

Unlike NGSolve / scikit-fem, dolfinx is rarely pip-installable in a
clean way -- the canonical install is conda, with optional system
wheel availability for some platform/Python combinations.  All probes
here return ``None`` (or ``False`` for individual checks) when
dolfinx is not importable so the test skips cleanly in environments
that lack it.

The catalog references for FEniCSx already exist in
``scripts/fingerprint_solvers.py`` (``fingerprint_fenics()``); this
module exposes the underlying primitives so the catalog-consistency
test can reuse the same API map.
"""

from __future__ import annotations

import importlib


def has_attr(dotted_path: str) -> bool | None:
    """Return ``True``/``False`` if the dotted-path attribute can be
    resolved on the installed ``dolfinx`` package; return ``None``
    when ``dolfinx`` is not importable at all so the caller can
    distinguish "missing attribute" from "module unavailable".

    Example::

        has_attr("dolfinx.mesh.create_rectangle")  # True / False / None
        has_attr("dolfinx.fem.petsc.LinearProblem")
    """
    parts = dotted_path.split(".")
    if not parts or parts[0] != "dolfinx":
        # The function is dolfinx-specific; refuse to evaluate
        # anything else so the caller doesn't accidentally pass an
        # arbitrary path and get a misleading True.
        return None
    try:
        importlib.import_module("dolfinx")
    except ImportError:
        return None
    # Walk the path; promote missing submodule imports to AttributeError
    # so the answer is uniformly bool from here down.
    obj: object | None = None
    module_path = parts[0]
    obj = importlib.import_module(module_path)
    for segment in parts[1:]:
        try:
            obj = getattr(obj, segment)
        except AttributeError:
            return False
        # If the segment is itself a submodule it may not have been
        # imported yet; try the import explicitly.
        if obj is None or (not callable(obj) and not hasattr(obj, "__dict__")):
            try:
                obj = importlib.import_module(f"{module_path}.{segment}")
                module_path = f"{module_path}.{segment}"
            except ImportError:
                return False
        else:
            module_path = f"{module_path}.{segment}"
    return True


def is_available() -> bool:
    """``True`` if ``dolfinx`` is importable in the current environment."""
    try:
        importlib.import_module("dolfinx")
    except ImportError:
        return False
    return True


# Canonical API map -- the catalog promises these dotted paths exist.
# Sourced from `scripts/fingerprint_solvers.py::fingerprint_fenics`
# so both modules stay in sync.  Extend when the catalog adds new
# references.
CATALOG_API_PATHS: tuple[str, ...] = (
    "dolfinx.mesh.create_rectangle",
    "dolfinx.mesh.create_box",
    "dolfinx.mesh.create_unit_square",
    "dolfinx.mesh.create_unit_cube",
    "dolfinx.fem.functionspace",
    "dolfinx.fem.Function",
    "dolfinx.fem.dirichletbc",
    "dolfinx.fem.locate_dofs_topological",
    "dolfinx.fem.locate_dofs_geometrical",
    "dolfinx.fem.petsc.LinearProblem",
    "dolfinx.fem.petsc.NonlinearProblem",
    "dolfinx.io.XDMFFile",
    "dolfinx.io.VTKFile",
    "dolfinx.default_scalar_type",
)
