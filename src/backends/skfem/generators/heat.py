"""scikit-fem heat conduction generators and knowledge."""


def _heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Heat conduction with Dirichlet BCs on left and right."""
    nx = params.get("nx", 32)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    return f'''\
"""Heat conduction on [0,1]² — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace
import numpy as np
import json

_tol = 1e-10
m = (MeshQuad.init_tensor(np.linspace(0, 1, {nx+1}), np.linspace(0, 1, {nx+1}))
     .with_boundaries({{
         "left":  lambda x: x[0] < _tol,
         "right": lambda x: x[0] > 1.0 - _tol,
     }}))
e = ElementQuad1()
ib = Basis(m, e)

K = laplace.assemble(ib)
f = ib.zeros()

# Dirichlet BCs.  In scikit-fem >= 8, dofs are looked up by passing the
# boundary tag name to `get_dofs(...)` directly; the older
# `ib.get_dofs()["name"]` subscript form raises TypeError because
# `DofsView` is not subscriptable by string.
left_dofs = ib.get_dofs("left").flatten()
right_dofs = ib.get_dofs("right").flatten()
D = np.concatenate([left_dofs, right_dofs])
# `condense(..., x=...)` expects a full-size vector (length = total DOFs)
# with the prescribed values at the constrained positions, NOT just the
# boundary values concatenated.  Build the full vector explicitly.
x_full = ib.zeros()
x_full[left_dofs]  = {T_left}
x_full[right_dofs] = {T_right}

u = solve(*condense(K, f, x=x_full, D=D))
print(f"Temperature: max={{u.max():.6f}}")

import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{"max_value": float(u.max()), "n_dofs": int(K.shape[0])}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


def _heat_transient_2d_skfem(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Time-dependent heat equation with backward Euler time integration."""
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 0.1)
    f_val = params.get("f", 1.0)
    return f'''\
"""Transient heat equation: du/dt - Δu = f — backward Euler — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass, unit_load
import numpy as np
from scipy.sparse.linalg import spsolve
import json

# Mesh: structured quad mesh
m = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e = ElementQuad1()
ib = Basis(m, e)

# Assembly: stiffness and mass matrices
K = laplace.assemble(ib)
M = mass.assemble(ib)

# Source vector
f = {f_val} * unit_load.assemble(ib)

# Boundary DOFs: u=0 on all boundaries
D = ib.get_dofs().flatten()
I = ib.complement_dofs(D)

# Backward Euler: (M + dt*K) * u_new = M * u_old + dt * f
dt = {dt}
A = M + dt * K

# Factor the system matrix once (reused each step)
from scipy.sparse.linalg import factorized
A_solve = factorized(A[I][:, I].tocsc())

# Initial condition: u=0
u = ib.zeros()

# Time stepping
t = 0.0
n_steps = int({T_end} / dt)
for step in range(n_steps):
    rhs = M @ u + dt * f
    u[D] = 0.0
    u[I] = A_solve(rhs[I])
    t += dt

max_val = u.max()
print(f"t={{t:.4f}}, max(u) = {{max_val:.10f}}, steps={{n_steps}}")

import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"temperature": u}})
mio.write("result.vtu")

summary = {{
    "max_value": float(max_val),
    "n_dofs": len(u),
    "n_elements": m.nelements,
    "time": t,
    "steps": n_steps,
    "dt": dt,
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Transient heat solve complete.")
'''


KNOWLEDGE = {
    "heat": {
        "description": "Heat conduction (steady/transient) — examples 19, 25, 28, 39, 50",
        "solver": "Direct sparse (scipy), or time-stepping for transient",
        "pitfalls": [
            "Non-homogeneous Dirichlet: condense(K, f, x=boundary_values, D=boundary_dofs)",
            "Transient: M*du/dt + K*u = f -> backward Euler: (M + dt*K)*u_new = M*u_old + dt*f",
            "Conjugate heat transfer (example 28): couple fluid and solid subdomains",
        ],
    },
    "heat_transient": {
        "description": "Time-dependent heat equation with backward Euler (scikit-fem)",
        "solver": "Backward Euler: (M + dt*K)*u_new = M*u_old + dt*f, factorized for efficiency",
        "elements": "ElementQuad1, ElementTriP1 (any standard H1 element)",
        "pitfalls": [
            "Backward Euler: unconditionally stable, first-order accurate in time",
            "Factor system matrix once with factorized() and reuse each time step",
            "Crank-Nicolson (theta=0.5): (M + 0.5*dt*K)*u_new = (M - 0.5*dt*K)*u_old + dt*f",
            "Mass matrix: use mass from skfem.models.poisson",
            "For non-homogeneous BCs changing in time: condense at each step",
        ],
    },
}

GENERATORS = {
    "heat_2d": _heat_2d,
    "heat_2d_steady": _heat_2d,
    "heat_transient_2d": _heat_transient_2d_skfem,
}
