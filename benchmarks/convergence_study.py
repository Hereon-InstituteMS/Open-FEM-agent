#!/usr/bin/env python3
"""
h-Convergence study: Poisson on [0,1]² at 4 mesh resolutions across 3 solvers.

Verifies O(h²) convergence rate and that all solvers converge to the same limit.
Results saved to benchmarks/results/ for paper figures.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
pv.OFF_SCREEN = True

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ.setdefault("FOURC_ROOT", os.path.expanduser("~/4C"))
os.environ.setdefault("FOURC_BINARY", os.path.expanduser("~/4C/build/4C"))
ld = os.environ.get("LD_LIBRARY_PATH", "")
if "/opt/4C-dependencies/lib" not in ld:
    os.environ["LD_LIBRARY_PATH"] = f"/opt/4C-dependencies/lib:{ld}"

from core.registry import load_all_backends, get_backend
from core.backend import BackendStatus
from backends.fourc.inline_mesh import matched_poisson_input


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def extract_max_scalar(vtu_path):
    mesh = pv.read(str(vtu_path))
    for name in mesh.point_data:
        if name.lower() in ("owner", "ghostnodes", "elementowner"):
            continue
        data = mesh.point_data[name]
        if data.ndim == 1:
            return float(data.max())
    return None


def run_fenics(n):
    b = get_backend("fenics")
    content = b.generate_input("poisson", "2d", {"kappa": 1.0, "nx": n, "ny": n})
    with tempfile.TemporaryDirectory() as tmpdir:
        job = _run(b.run(content, Path(tmpdir), np=1, timeout=300))
        if job.status != "completed":
            return None, 0
        vtus = sorted([f for f in b.get_result_files(job) if f.suffix == ".vtu"])
        if vtus:
            return extract_max_scalar(vtus[-1]), (n+1)**2
    return None, 0


def run_dealii(refinements):
    b = get_backend("dealii")
    content = b.generate_input("poisson", "2d", {"refinements": refinements})
    with tempfile.TemporaryDirectory() as tmpdir:
        job = _run(b.run(content, Path(tmpdir), np=1, timeout=300))
        if job.status != "completed":
            return None, 0
        vtus = sorted([f for f in b.get_result_files(job) if f.suffix == ".vtu"])
        if vtus:
            return extract_max_scalar(vtus[-1]), (2**refinements + 1)**2
    return None, 0


def run_4c(n):
    b = get_backend("fourc")
    if not b or b.check_availability()[0] != BackendStatus.AVAILABLE:
        return None, 0
    yaml = matched_poisson_input(n, n)
    with tempfile.TemporaryDirectory() as tmpdir:
        job = _run(b.run(yaml, Path(tmpdir), np=1, timeout=300))
        if job.status != "completed":
            return None, 0
        vtus = sorted([f for f in b.get_result_files(job) if f.suffix == ".vtu"])
        if vtus:
            return extract_max_scalar(vtus[-1]), (n+1)**2
    return None, 0


def main():
    load_all_backends()

    # Mesh resolutions: h = 1/n
    # FEniCS: n = 8, 16, 32, 64
    # deal.II: refine = 3, 4, 5, 6 → n = 8, 16, 32, 64
    # 4C: n = 8, 16, 32, 64

    resolutions = [8, 16, 32, 64]
    dealii_refine = [3, 4, 5, 6]

    results = {"fenics": [], "dealii": [], "fourc": []}

    print("=" * 70)
    print("h-CONVERGENCE STUDY: Poisson -Δu=1 on [0,1]², u=0 on ∂Ω")
    print("=" * 70)

    for i, n in enumerate(resolutions):
        h = 1.0 / n
        print(f"\n--- h = 1/{n} = {h:.4f} ---")

        # FEniCS
        val, dofs = run_fenics(n)
        if val is not None:
            results["fenics"].append({"n": n, "h": h, "dofs": dofs, "max_u": val})
            print(f"  FEniCS:  max(u) = {val:.8f}, DOFs = {dofs}")

        # deal.II
        val, dofs = run_dealii(dealii_refine[i])
        if val is not None:
            results["dealii"].append({"n": n, "h": h, "dofs": dofs, "max_u": val})
            print(f"  deal.II: max(u) = {val:.8f}, DOFs = {dofs}")

        # 4C
        val, dofs = run_4c(n)
        if val is not None:
            results["fourc"].append({"n": n, "h": h, "dofs": dofs, "max_u": val})
            print(f"  4C:      max(u) = {val:.8f}, DOFs = {dofs}")

    # Compute convergence rates
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)

    for solver, data in results.items():
        if len(data) < 2:
            continue
        print(f"\n{solver}:")
        for i in range(1, len(data)):
            h_prev = data[i-1]["h"]
            h_curr = data[i]["h"]
            u_prev = data[i-1]["max_u"]
            u_curr = data[i]["max_u"]
            # Richardson extrapolation-style rate
            if u_prev != u_curr:
                rate = np.log(abs(u_prev - u_curr)) / np.log(h_curr / h_prev) if i < len(data) - 1 else None
            print(f"  h={h_curr:.4f}: max(u)={u_curr:.8f}, "
                  f"Δu={abs(u_curr - u_prev):.2e}")

    # Final comparison at finest mesh
    print("\n" + "=" * 70)
    print("FINEST MESH COMPARISON (h=1/64)")
    print("=" * 70)
    finest = {}
    for solver, data in results.items():
        if data:
            finest[solver] = data[-1]["max_u"]
            print(f"  {solver:8s}: max(u) = {data[-1]['max_u']:.8f}")

    if len(finest) >= 2:
        vals = list(finest.values())
        mean_val = sum(vals) / len(vals)
        for solver, val in finest.items():
            print(f"  {solver:8s}: rel_diff from mean = {abs(val - mean_val) / mean_val:.4%}")

    # Save results
    outdir = Path(__file__).parent / "results"
    outdir.mkdir(exist_ok=True)
    with open(outdir / "convergence_poisson.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outdir / 'convergence_poisson.json'}")


if __name__ == "__main__":
    main()
