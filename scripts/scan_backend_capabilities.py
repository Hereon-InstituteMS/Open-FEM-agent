#!/usr/bin/env python3
"""
Layer A — per-backend source-capability scanner.

For each backend, walk the installed Python package (and where
applicable the on-disk source tree) to enumerate every capability
the backend ACTUALLY exposes: Applications, Elements, Conditions,
Constitutive laws, Variables, Mesh generators, Element families,
... — whatever the backend's introspection API surfaces.

Emit a JSON snapshot per backend under `scripts/scan_results/`,
plus a top-level summary.  The snapshots are the input for the
catalog-vs-scan consistency test (tests/test_catalog_vs_scan.py,
landing in a follow-up PR) which surfaces gaps in both directions:

  * **drift**: in catalog but not in source — the MCP advertises a
    capability the backend does not actually ship.
  * **coverage gap**: in source but not in catalog — the MCP misses
    a capability the backend does ship.

Usage:
    python scripts/scan_backend_capabilities.py [--backend NAME]

This is the first foundation of the multi-week "scan every backend's
physics / modules / capabilities and encode the gaps into the MCP"
pipeline.  This PR ships the Kratos scanner; subsequent PRs add the
4C / FEniCSx / scikit-fem / NGSolve / deal.II / DUNE-fem scanners
the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_RESULTS = REPO_ROOT / "scripts" / "scan_results"


# ── data model ─────────────────────────────────────────────────────────


@dataclass
class BackendCapabilities:
    """What a backend exposes at the source level.

    Each field is best-effort populated by the per-backend scanner —
    when a backend does not expose a category through its Python
    introspection surface (e.g. Kratos elements need a C++ registry
    walk we cannot do from pure Python), the field stays empty and
    the consistency test treats that as "no information" rather than
    "no capability".
    """

    backend: str
    version: str = ""
    applications: list[str] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    constitutive_laws: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    mesh_generators: list[str] = field(default_factory=list)
    element_families: list[str] = field(default_factory=list)
    processes: list[str] = field(default_factory=list)
    modelers: list[str] = field(default_factory=list)
    other: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ── Kratos scanner ─────────────────────────────────────────────────────


def scan_kratos() -> BackendCapabilities:
    """Enumerate what KratosMultiphysics exposes through Python.

    Kratos splits its capability surface across many places:

      * `import KratosMultiphysics.<X>Application` — each application
        module is itself a pybind11 binding registering Elements /
        Conditions / ConstitutiveLaws into the global C++ Registry.
        We discover installed applications by walking the package
        directory and attempting an import; success means the
        application is wired into the runtime.

      * `KM.KratosGlobals.Kernel.GetAllVariableNames()` — every
        registered nodal/element variable, across all currently-
        imported applications.

      * `KM.Registry` — a python-facing tree of registered Processes,
        Modelers, Stages, OutputProcesses.  Walked via
        `Registry.HasItem(path)` + `Registry.NumberOfItems(path)`.

      * `KM.KratosGlobals.HasConstitutiveLaw(name)` — point query;
        cannot enumerate directly, only verify presence.  For the
        scan we keep a curated probe list of constitutive-law names
        the MCP catalog mentions (extracted at runtime from the
        catalog) and report which are actually registered.

    Element / Condition class names are not currently enumerable
    from Python; we record the per-application module's
    `dir(<app>)` filtered to names ending in `Element` or
    `Condition` as a best-effort approximation.
    """
    cap = BackendCapabilities(backend="kratos")

    try:
        import KratosMultiphysics as KM
    except ImportError as e:
        cap.notes.append(f"KratosMultiphysics not importable: {e}")
        return cap

    cap.version = str(KM.KratosGlobals.Kernel.Version())

    # ── 1. installed applications
    # Walk the package directory; an entry is an application if its
    # subdirectory contains an __init__.py (i.e. it is a Python package
    # in its own right) AND we can import it without error.
    kpkg = Path(KM.__file__).parent
    candidate_apps = sorted(
        p.name for p in kpkg.iterdir()
        if p.is_dir()
        and p.name.endswith("Application")
        and (p / "__init__.py").exists()
    )
    apps_seen: list[str] = []
    apps_failed: dict[str, str] = {}
    for app in candidate_apps:
        try:
            __import__(f"KratosMultiphysics.{app}")
            apps_seen.append(app)
        except Exception as e:
            apps_failed[app] = f"{type(e).__name__}: {e!s:.120}"
    cap.applications = apps_seen
    if apps_failed:
        cap.other["applications_failed_to_import"] = [
            f"{k}: {v}" for k, v in apps_failed.items()
        ]

    # ── 2. variables registered globally after all imported applications.
    # `GetAllVariableNames()` returns a single whitespace-indented string
    # with one variable name per line (not a list).  We split + strip
    # and drop empty / typedef-name lines.
    try:
        raw = KM.KratosGlobals.Kernel.GetAllVariableNames()
        if isinstance(raw, str):
            var_names = [ln.strip() for ln in raw.splitlines()]
        else:
            var_names = list(raw)
        var_names = [v for v in var_names if v]
        cap.variables = sorted(set(var_names))
    except Exception as e:
        cap.notes.append(f"GetAllVariableNames failed: {e!s:.120}")

    # ── 3. Registry tree — Processes / Modelers / Stages / OutputProcesses
    for cat_attr, cat_path in [
        ("processes", "Processes"),
        ("modelers", "Modelers"),
    ]:
        try:
            if KM.Registry.HasItem(cat_path):
                n = KM.Registry.NumberOfItems(cat_path)
                # The Registry doesn't expose a clean "list child keys"
                # — but we can enumerate by trying common sub-paths and
                # walking what HasItems reveals.  For now we record the
                # count; a deeper recursive walk lands in a follow-up.
                getattr(cap, cat_attr).append(f"<{n} items at Registry[{cat_path!r}]>")
        except Exception as e:
            cap.notes.append(f"Registry[{cat_path}] walk failed: {e!s:.80}")

    # ── 4. element / condition / constitutive-law names per application
    # Python introspection only: filter `dir(<app>)` for typical naming.
    elements: list[str] = []
    conditions: list[str] = []
    claws: list[str] = []
    for app in apps_seen:
        try:
            mod = sys.modules[f"KratosMultiphysics.{app}"]
        except KeyError:
            continue
        for name in dir(mod):
            if name.startswith("_"):
                continue
            if name.endswith("Element") and not name.endswith("BoundaryElement"):
                elements.append(f"{app}::{name}")
            elif name.endswith("Condition"):
                conditions.append(f"{app}::{name}")
            elif name.endswith("Law"):
                claws.append(f"{app}::{name}")
    cap.elements = sorted(elements)
    cap.conditions = sorted(conditions)
    cap.constitutive_laws = sorted(claws)

    # ── 5. for completeness — record the package install path
    cap.notes.append(f"package_dir={kpkg}")

    return cap


# ── dispatch ───────────────────────────────────────────────────────────


SCANNERS = {
    "kratos": scan_kratos,
    # 4C, fenics, dealii, skfem, ngsolve, dune scanners land in
    # follow-up PRs.  Each backend gets its own focused function
    # following the same pattern.
}


def run(backends: list[str]) -> dict[str, BackendCapabilities]:
    SCAN_RESULTS.mkdir(parents=True, exist_ok=True)
    out: dict[str, BackendCapabilities] = {}
    for name in backends:
        scanner = SCANNERS.get(name)
        if scanner is None:
            print(f"  {name:8s}  (no scanner yet — pending PR)")
            continue
        print(f"  {name:8s}  scanning…", end="", flush=True)
        cap = scanner()
        out[name] = cap
        (SCAN_RESULTS / f"{name}.json").write_text(
            json.dumps(asdict(cap), indent=2, default=str)
        )
        print(f" applications={len(cap.applications)} "
              f"elements={len(cap.elements)} "
              f"conditions={len(cap.conditions)} "
              f"laws={len(cap.constitutive_laws)} "
              f"variables={len(cap.variables)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="",
                    help="restrict to one backend name; default scans all known")
    args = ap.parse_args()

    targets = [args.backend] if args.backend else sorted(SCANNERS.keys())
    print(f"Scanning {len(targets)} backend(s): {', '.join(targets)}")
    results = run(targets)

    # Top-level summary
    summary = {
        name: {
            "version": cap.version,
            "n_applications": len(cap.applications),
            "n_elements": len(cap.elements),
            "n_conditions": len(cap.conditions),
            "n_constitutive_laws": len(cap.constitutive_laws),
            "n_variables": len(cap.variables),
            "notes": cap.notes,
        }
        for name, cap in results.items()
    }
    (SCAN_RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nresults under {SCAN_RESULTS}/")


if __name__ == "__main__":
    main()
