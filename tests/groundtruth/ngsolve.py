"""Ground-truth probes for NGSolve.

NGSolve's public surface lives entirely on the top-level ``ngsolve``
module (FE spaces ``H1``/``HCurl``/``HDiv``/``L2``, ``Mesh``,
``GridFunction``, ``BilinearForm``, ``LinearForm``, ``Integrate``,
``InnerProduct``, the ``VOL`` / ``BND`` markers, etc.).  Catalog
templates use ``from ngsolve import *``, so every identifier that
appears as a constructor or factory call in those templates must be
a real attribute on the package.

This is the third instance of the Python-introspection probe family
(``skfem.py`` was the first; deal.II uses the source-grep family).
Returns ``None`` when NGSolve is not importable so the test skips
rather than fails in environments without it.
"""

from __future__ import annotations


def public_attrs() -> set[str] | None:
    """Public attribute names on the top-level ``ngsolve`` module.

    Returns ``None`` if NGSolve is not installed.  The returned set
    is the full public surface (~200 entries) rather than a curated
    sub-list, so a contributor adding a typo'd identifier to a
    template (e.g. ``BilinearFrom`` instead of ``BilinearForm``)
    fails the catalog-consistency test regardless of which corner of
    the API the typo lives in.
    """
    try:
        import ngsolve  # type: ignore
    except ImportError:
        return None
    return {name for name in dir(ngsolve) if not name.startswith("_")}
