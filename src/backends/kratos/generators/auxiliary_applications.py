"""Kratos auxiliary applications — utility, infrastructure, and legacy apps.

These do not own their own physics problem types, so they have no
GENERATORS — but the agent needs to know about them because other
generators reference them (e.g. FSI pulls in MappingApplication;
parallel runs need TrilinosApplication + MetisApplication; HDF5
output needs HDF5Application).

Source: upstream Kratos `applications/` directory listing.  Every name
in `KNOWLEDGE` below must correspond to a real sub-application.
"""


KNOWLEDGE = {
    "_auxiliary_overview": {
        "description": (
            "Kratos applications that the agent must know about even though "
            "they do not provide their own physics problem type: parallel "
            "infrastructure, I/O, meshing, mapping, statistics, and legacy "
            "predecessors of currently-active applications.  Pulled in as "
            "dependencies of physics applications rather than driven directly."
        ),
        "infrastructure_apps": {
            "TrilinosApplication": (
                "Trilinos linear-solver wrappers (Epetra, AztecOO, Amesos, ML, "
                "MueLu) for distributed-memory MPI runs.  Required by any "
                "parallel Kratos analysis using iterative or AMG-preconditioned "
                "solvers.  Pip install hint: KratosTrilinosApplication."
            ),
            "MetisApplication": (
                "Metis-based mesh partitioner for MPI runs.  Used by the model-"
                "part-IO splitter to produce per-rank .mdpa files.  Pulled in "
                "by every MPI workflow; no Python-facing physics on its own.  "
                "Pip install hint: KratosMetisApplication."
            ),
            "LinearSolversApplication": (
                "Linear-solver wrappers beyond Kratos core (Eigen-based "
                "sparse_qr/sparse_lu/sparse_cg, PARDISO, complex-valued solvers).  "
                "Used by structural and electromagnetic analyses needing direct "
                "factorisation or complex arithmetic.  Pip install hint: "
                "KratosLinearSolversApplication."
            ),
            "HDF5Application": (
                "Parallel HDF5 I/O.  Provides HDF5OutputProcess and HDF5IO for "
                "checkpointing, restart, and PETSc-friendly result storage.  "
                "Used by long simulations and FSI restart workflows.  Pip "
                "install hint: KratosHDF5Application."
            ),
            "MedApplication": (
                "MED file format I/O (Salome ecosystem).  Provides import/"
                "export of Salome-generated meshes and post-processing into "
                "MED format.  Pip install hint: KratosMedApplication."
            ),
        },
        "meshing_and_mapping": {
            "MeshingApplication": (
                "Adaptive mesh refinement (h-refinement) and remeshing.  "
                "Provides MMG / PMMG bindings for tetrahedral and triangular "
                "remeshing driven by error indicators.  Used by large-"
                "deformation solid mechanics and adaptive CFD.  Pip install "
                "hint: KratosMeshingApplication."
            ),
            "MeshMovingApplication": (
                "ALE (arbitrary Lagrangian-Eulerian) mesh-moving strategies: "
                "Laplacian smoothing, structural-similarity, and rigid-body "
                "displacement of inner boundaries.  Required by FSI (the fluid "
                "mesh follows the structural interface) and by any moving-"
                "boundary CFD.  Pip install hint: KratosMeshMovingApplication."
            ),
            "MappingApplication": (
                "Inter-mesh field mapping for non-conforming or non-matching "
                "discretisations.  Provides nearest-neighbour, barycentric, "
                "and mortar mappers used by FSI partitioned solvers, "
                "thermo-mechanical coupling, and CoSimulation.  Pip install "
                "hint: KratosMappingApplication."
            ),
        },
        "analysis_utilities": {
            "StatisticsApplication": (
                "Statistical post-processing of time-series and ensemble "
                "fields: mean, variance, RMS, time-averaged Reynolds stresses, "
                "spatially-averaged quantities.  Used in turbulence post-"
                "processing and uncertainty quantification.  Pip install "
                "hint: KratosStatisticsApplication."
            ),
            "SystemIdentificationApplication": (
                "System identification and inverse-problem parameter "
                "estimation: fits material parameters or boundary-condition "
                "magnitudes against measured response data.  Pip install "
                "hint: KratosSystemIdentificationApplication."
            ),
        },
        "legacy_apps": {
            "SolidMechanicsApplication": (
                "Legacy solid-mechanics application.  Predecessor of "
                "StructuralMechanicsApplication — prefer the latter for new "
                "work.  Still present upstream for backward compatibility "
                "with older input decks.  Pip install hint: "
                "KratosSolidMechanicsApplication."
            ),
            "ContactMechanicsApplication": (
                "Legacy contact-mechanics application.  Largely superseded "
                "by ContactStructuralMechanicsApplication for the "
                "structural-contact workflow.  Pip install hint: "
                "KratosContactMechanicsApplication."
            ),
        },
        "pitfalls": [
            "TrilinosApplication and MetisApplication ship in the MPI-parallel "
            "Kratos build only; pip-installed serial wheels do not include them.",
            "MappingApplication and MeshMovingApplication are *required* (not "
            "optional) for partitioned FSI even if not named explicitly in "
            "the user-facing JSON — the FSIApplication imports them.",
            "Prefer StructuralMechanicsApplication over SolidMechanicsApplication "
            "and ContactStructuralMechanicsApplication over ContactMechanicsApplication "
            "for new analyses; the legacy apps are kept only for input-deck "
            "backward compatibility and receive minimal maintenance.",
        ],
    },
}


GENERATORS: dict = {}
