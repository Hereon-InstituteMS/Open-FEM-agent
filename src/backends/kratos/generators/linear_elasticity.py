"""Kratos linear elasticity generators and knowledge."""


def _elasticity_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity on rectangular domain — Kratos (manual assembly)."""
    nx = params.get("nx", 40)
    ny = params.get("ny", 4)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Linear elasticity: rectangular domain, fixed left — Kratos (manual assembly)"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

nx, ny, lx, ly = {nx}, {ny}, {lx}, {ly}
nid = 1; node_map = {{}}; coords = {{}}
for j in range(ny+1):
    for i in range(nx+1):
        coords[nid] = (i*lx/nx, j*ly/ny)
        node_map[(i,j)] = nid; nid += 1
n_nodes = nid - 1

elements = []
for j in range(ny):
    for i in range(nx):
        n1,n2,n3,n4 = node_map[(i,j)],node_map[(i+1,j)],node_map[(i+1,j+1)],node_map[(i,j+1)]
        elements.append((n1,n2,n4)); elements.append((n2,n3,n4))

ndof = 2 * n_nodes
K = lil_matrix((ndof, ndof))
F = np.zeros(ndof)
mu, lam = {mu}, {lam}

for tri in elements:
    ids = [t-1 for t in tri]
    x = np.array([coords[t][0] for t in tri])
    y = np.array([coords[t][1] for t in tri])
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2*area)
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2*area)

    B = np.zeros((3, 6))
    for a in range(3):
        B[0, 2*a] = b[a]; B[1, 2*a+1] = c[a]
        B[2, 2*a] = c[a]; B[2, 2*a+1] = b[a]
    D = np.array([[lam+2*mu, lam, 0], [lam, lam+2*mu, 0], [0, 0, mu]])
    Ke = area * B.T @ D @ B

    dofs = []
    for a in range(3):
        dofs.extend([2*ids[a], 2*ids[a]+1])
    for i in range(6):
        F[dofs[i]] += -1.0 * area / 3.0 if i % 2 == 1 else 0  # body force — set for your problem
        for j_idx in range(6):
            K[dofs[i], dofs[j_idx]] += Ke[i, j_idx]
K = K.tocsr()

# Fix left edge
fixed = set()
for j in range(ny+1):
    n = node_map[(0,j)] - 1
    fixed.add(2*n); fixed.add(2*n+1)
interior = sorted(set(range(ndof)) - fixed)

u = np.zeros(ndof)
u[interior] = spsolve(K[np.ix_(interior, interior)], F[interior])

uy = u[1::2]
print(f"Max tip displacement: {{uy.min():.6f}}")
summary = {{"max_displacement_y": float(uy.min()), "n_dofs": ndof}}
with open("results_summary.json", "w") as _f: json.dump(summary, _f, indent=2)
'''


def _elasticity_nonlinear_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Nonlinear elasticity via Kratos StructuralMechanicsApplication."""
    return f'''\
"""Nonlinear structural mechanics — Kratos StructuralMechanicsApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.StructuralMechanicsApplication as SMA
    print("StructuralMechanicsApplication available")
    # Full Kratos structural analysis would use:
    # from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import StructuralMechanicsAnalysis
    # with ProjectParameters.json + mesh.mdpa
    summary = {{"note": "Kratos SMA available — use ProjectParameters.json workflow for full analysis"}}
except ImportError:
    print("StructuralMechanicsApplication not installed")
    print("Install: pip install KratosStructuralMechanicsApplication")
    summary = {{"note": "KratosStructuralMechanicsApplication not installed"}}
with open("results_summary.json", "w") as _f: json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Structural mechanics via StructuralMechanicsApplication (SMA)",
        "application": "StructuralMechanicsApplication (pip install KratosStructuralMechanicsApplication)",
        "elements": {
            "2D": ["SmallDisplacementElement2D3N/4N/6N/8N/9N (linear, small strain)",
                   "TotalLagrangianElement2D3N/4N (nonlinear, large deformation)",
                   "UpdatedLagrangianElement2D3N/4N"],
            "3D": ["SmallDisplacementElement3D4N/8N/10N/20N/27N",
                   "TotalLagrangianElement3D4N/8N"],
            "shells": ["ShellThinElement3D3N (MITC, Kirchhoff-Love)",
                      "ShellThickElement3D4N (Reissner-Mindlin)"],
            "beams": ["CrBeamElement3D2N (co-rotational)", "CrLinearBeamElement3D2N"],
            "trusses": ["TrussElement3D2N", "TrussLinearElement3D2N"],
            "cables": ["CableElement3D2N"],
            "springs": ["SpringDamperElement3D2N", "NodalConcentratedElement2D1N/3D1N"],
        },
        "constitutive_laws": {
            "linear": ["LinearElastic3DLaw", "LinearElasticPlaneStrain2DLaw",
                       "LinearElasticPlaneStress2DLaw", "LinearElasticAxisymmetric2DLaw",
                       "TrussConstitutiveLaw", "BeamConstitutiveLaw"],
            "hyperelastic": ["HyperElastic3DLaw (Saint Venant-Kirchhoff)",
                            "HyperElasticIsotropicNeoHookean2D/3DLaw"],
            "plasticity": "Factory: 7 yield surfaces x 5 plastic potentials x 6 hardening curves",
            "damage": ["IsotropicDamage (factory)", "DplusDminusDamage (tension/compression split)"],
            "viscoelastic": ["GeneralizedMaxwell (relaxation)", "GeneralizedKelvin (creep)"],
        },
        "solver_types": ["static (Newton-Raphson)", "dynamic (Newmark, Bossak, GenAlpha)",
                        "explicit (central differences)", "formfinding"],
        # Pitfalls are tagged with Table-1 category prefix
        # ([Syntax]/[Physics]/[Numerical]/[API]/[Integration]) so the
        # agent and any downstream tooling can filter on category.
        # New entries SHOULD include a `Signal:` clause stating the
        # observable symptom — silent no-ops (the FSI velocity-30x
        # case in the Open-FEM-Agent paper, Section 3.2) are the
        # category that has hurt users most, so making the symptom
        # explicit lets the post-exec critic match against it.
        "pitfalls": [
            "[Syntax] Element names MUST include node count: "
            "SmallDisplacementElement2D3N, not SmallDisplacement2D. "
            "Signal: `KratosMultiphysics.Exception: Element name not found`.",
            "[API] Materials defined in StructuralMaterials.json, referenced "
            "by Properties ID. The .json + mdpa + ProjectParameters.json "
            "trio must agree on the Properties ID or the law silently "
            "defaults to a zero constitutive response. "
            "Signal: displacements are linear in load but unrealistically large.",
            "[Syntax] SubModelParts must match between .mdpa and "
            "ProjectParameters.json exactly. A typo binds the process "
            "to an empty SubModelPart and the BC silently no-ops. "
            "Signal: BC nodes show non-zero residuals at convergence.",
            "[Numerical] For nonlinear: increase max_iterations beyond "
            "the default 10. With contact, plasticity, or large rotation "
            "10 iterations is usually insufficient. "
            "Signal: solver reports 'max iterations reached' but exits 0.",
            "[Physics] DISPLACEMENT is the primary DOF for solid elements; "
            "ROTATION is also required on beams/shells. Adding only "
            "DISPLACEMENT for a beam element silently drops the rotational "
            "DOF and gives a hinge-like response. "
            "Signal: beam tip deflects but does not rotate under moment load.",
            "[Numerical] SHEAR LOCKING — Linear hex8 (3D8N) and quad4 "
            "(2D4N) elements lock in bending-dominated problems, "
            "producing overly stiff results and wrong frequencies. Use "
            "quadratic elements (3D20N, 3D27N, 2D8N, 2D9N) for any "
            "problem with significant bending. "
            "Signal: tip deflection is order-of-magnitude smaller than "
            "Euler-Bernoulli prediction.",
            "[API] For POINT_LOAD application: use assign_vector_variable_process "
            "with constrained: [false, false, false]. Do NOT use "
            "assign_vector_by_direction_process — it expects a kinematic "
            "variable (DISPLACEMENT class) and silently no-ops for load "
            "variables, leaving the model unloaded. "
            "Signal: nodal reactions sum to zero and tip displacement is zero "
            "despite the process being present in ProjectParameters.json.",
            "[Syntax] problem_data section MUST include 'echo_level' "
            "field. Missing it raises a confusing 'Parameters' KeyError "
            "inside the analysis stage rather than at parameter validation.",
            # --- Retroactive entries from PR #24 Kratos LE stub-replacement ---
            "[Syntax] SmallDisplacementElement2D{3,4,6,8,9}N inherits the "
            "SolidElementCheck path from BaseSolidElement, which queries "
            "the Z DOF on every node even in plane analyses (Kratos uses "
            "3-component vectors internally). Add DISPLACEMENT_Z and "
            "REACTION_Z DOFs to every node and Fix(DISPLACEMENT_Z)=0. "
            "Signal: `Check failed for DISPLACEMENT_Z` at strat.Check(), "
            "long before the first solver step.",
            "[API] CONSTITUTIVE_LAW Properties binding takes an *instance*, "
            "not a class: `prop.SetValue(KM.CONSTITUTIVE_LAW, "
            "SMA.LinearElasticPlaneStrain2DLaw())`. Passing the class "
            "itself silently keeps the default null law and the element "
            "responds with zero stiffness. "
            "Signal: K matrix assembles but is singular; spsolve reports "
            "extreme condition number or NaN displacements.",
            "[API] Nodal solution-step variables (DISPLACEMENT, REACTION, "
            "VOLUME_ACCELERATION, etc.) MUST be added via "
            "`mp.AddNodalSolutionStepVariable(v)` BEFORE the first node "
            "is created and BEFORE `mp.SetBufferSize(2)`. Adding after "
            "leaves the variable unallocated on existing nodes. "
            "Signal: `RuntimeError: trying to access non-existing "
            "Kratos component` when reading DISPLACEMENT post-solve.",
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d_kratos,
    "linear_elasticity_2d_nonlinear": _elasticity_nonlinear_kratos,
}
