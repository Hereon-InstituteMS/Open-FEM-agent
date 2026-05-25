"""Paper-ready figures for HOE-v1.

After redesign per reviewer critique:
  Fig R1: per-task × condition pass-rate heatmap with discrete colormap,
          bold tier dividers, marginal column means, k/N annotations on every cell.
  Fig R2: per-task observed convergence rate (Tier C only) — bars per
          condition with a horizontal line at the formal order. (Replaces
          the degenerate scatter from v1.)
  Fig R3: behavioural ablation — fraction of cells where the agent
          actually spawned a critic sub-agent, per condition. (Replaces
          the cost-Pareto plot, which was misleading because the MCP
          conditions did not have metered cost.)
  Fig R4: per-tier pass rate × condition with 95% Wilson CIs, k/N
          numerical labels, and an inset showing the paired pooled deltas
          with task-bootstrap 95% intervals.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
REPORT_OUT = ROOT / "report" / "out"
CSV = REPORT_OUT / "all_rows.csv"
STATS_JSON = REPORT_OUT / "stats.json"
SESSION_DIR = Path("/home/alexander/.claude/projects/-home-alexander-Schreibtisch-open-fem-agent")

TIER_OF = {
    **{f"A{i}": "A" for i in range(1, 6)},
    **{f"B{i}": "B" for i in range(1, 6)},
    **{f"C{i}": "C" for i in range(1, 6)},
    **{f"D{i}": "D" for i in range(1, 3)},
}
TASK_ORDER = sorted(TIER_OF, key=lambda t: (t[0], int(t[1:])))
COND_ORDER = ["BARE", "MCP_FULL", "MCP_NO_PITFALL_DB", "MCP_NO_CRITIC"]
COND_LABEL_TWO_LINE = {
    "BARE":              "BARE\n(no MCP)",
    "MCP_FULL":          "MCP\n(full)",
    "MCP_NO_PITFALL_DB": "MCP\n(no pitfall DB)",
    "MCP_NO_CRITIC":     "MCP\n(no critic)",
}
COND_LABEL = {
    "BARE":              "BARE (no MCP)",
    "MCP_FULL":          "MCP (full)",
    "MCP_NO_PITFALL_DB": "MCP (no pitfall DB)",
    "MCP_NO_CRITIC":     "MCP (no critic)",
}
COND_COLOR = {
    "BARE":              "#9E9E9E",
    "MCP_FULL":          "#2E7D32",
    "MCP_NO_PITFALL_DB": "#FB8C00",
    "MCP_NO_CRITIC":     "#1565C0",
}
TIER_DESC = {
    "A": "Textbook (5 tasks)",
    "B": "Compositional engineering (5 tasks)",
    "C": "MMS verification (5 tasks)",
    "D": "Adversarial / developer (2 tasks)",
}


def _load() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with CSV.open() as fh:
        for r in csv.DictReader(fh):
            r["passed"] = r["passed"] in ("True", "1", "true")
            try: r["seed"] = int(r["seed"])
            except: r["seed"] = -1
            r["tier"] = TIER_OF.get(r["task_id"], "?")
            rows.append(r)
    return rows


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0: return 0.0, 0.0
    z = 1.959963984540054
    p = k / n
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    half = (z/(1 + z*z/n)) * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return max(0.0, centre-half), min(1.0, centre+half)


# --------------------------------------------------------------------------- #
# Fig R1 — heatmap (redesigned)
# --------------------------------------------------------------------------- #
def fig_r1_heatmap(rows: list[dict[str, Any]], out_path: Path) -> None:
    # k/3 per (task, condition)
    buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        buckets[(r["task_id"], r["condition"])].append(bool(r["passed"]))
    M = np.full((len(TASK_ORDER), len(COND_ORDER)), np.nan)
    for i, t in enumerate(TASK_ORDER):
        for j, c in enumerate(COND_ORDER):
            cs = buckets.get((t, c), [])
            if cs:
                M[i, j] = sum(cs) / 3.0  # always 3 seeds expected

    # Discrete 4-level colormap
    cmap = ListedColormap(["#C62828", "#EF6C00", "#F9A825", "#2E7D32"])  # 0, 1/3, 2/3, 3/3
    norm = BoundaryNorm([0.0, 0.17, 0.5, 0.83, 1.001], ncolors=4)

    fig = plt.figure(figsize=(6.8, 8.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[5.2, 0.6],
                          height_ratios=[14.0, 0.7],
                          hspace=0.05, wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[1, 0], sharex=ax)
    ax_cb = fig.add_subplot(gs[0, 1])

    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([COND_LABEL_TWO_LINE[c] for c in COND_ORDER], fontsize=8)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_yticks(range(len(TASK_ORDER)))
    ax.set_yticklabels(TASK_ORDER, fontsize=8.5)

    # k/3 annotations on every cell
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{int(round(3*v))}/3"
            color = "white" if (np.isnan(v) or v <= 0.5) else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=8.5, fontweight="bold")

    # Bold tier dividers
    for cum, name in zip([5, 10, 15], "BCD"):
        ax.axhline(cum - 0.5, color="black", lw=1.7)
    # Tier brackets on left
    for tier, (top, bot) in zip("ABCD", [(0, 4), (5, 9), (10, 14), (15, 16)]):
        ax.annotate(
            "", xy=(-0.85, top - 0.4), xytext=(-0.85, bot + 0.4),
            xycoords="data",
            arrowprops=dict(arrowstyle="-", lw=1.5, color="black"),
            annotation_clip=False,
        )
        ax.text(-1.05, (top + bot) / 2, f"Tier {tier}",
                rotation=90, va="center", ha="center",
                fontsize=9, fontweight="bold")

    # Highlight failing cells with a black outline
    for i, t in enumerate(TASK_ORDER):
        for j, c in enumerate(COND_ORDER):
            v = M[i, j]
            if not np.isnan(v) and v < 0.5:
                ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96,
                                       fill=False, edgecolor="black",
                                       lw=1.6, zorder=10))

    ax.set_xlim(-0.55, len(COND_ORDER) - 0.45)
    ax.set_title("Seeds passed per task and condition",
                 fontsize=11, pad=6)

    # Bottom marginal: column means with k/51 labels AND condition labels on x
    col_mean = np.array([np.nanmean(M[:, j]) for j in range(M.shape[1])])
    ax_m.imshow(col_mean[None, :], cmap=cmap, norm=norm,
                aspect="auto", interpolation="nearest")
    ax_m.set_yticks([0]); ax_m.set_yticklabels(["pooled"], fontsize=8.5)
    # k/51 pooled annotation per column
    for j, cond in enumerate(COND_ORDER):
        cs = [r for r in rows if r["condition"] == cond]
        k = sum(1 for r in cs if r["passed"])
        n = len(cs)
        ax_m.text(j, 0, f"{k}/{n}",
                  ha="center", va="center",
                  color="white", fontsize=9, fontweight="bold")
    ax_m.set_xticks(range(len(COND_ORDER)))
    ax_m.set_xticklabels([COND_LABEL_TWO_LINE[c] for c in COND_ORDER],
                         fontsize=8.5)
    ax_m.tick_params(axis="x", length=0, pad=4)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", which="both", length=0)

    # Discrete 4-level colour bar with one tick at the centre of each band
    # (red 0/3, orange 1/3, yellow 2/3, green 3/3). BoundaryNorm boundaries
    # are [0.0, 0.17, 0.5, 0.83, 1.001], so band centres are:
    band_centres = [0.085, 0.335, 0.665, 0.917]
    cb = fig.colorbar(im, cax=ax_cb, ticks=band_centres)
    cb.ax.set_yticklabels(["0/3", "1/3", "2/3", "3/3"], fontsize=8.5)
    cb.ax.tick_params(length=0, pad=4)
    cb.outline.set_visible(False)
    cb.ax.set_title("seeds\npassed", fontsize=8, pad=6)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig R2 — per-task Tier C observed convergence orders
# --------------------------------------------------------------------------- #
def fig_r2_tier_c_orders(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Per-task bar chart of observed mean convergence rate (averaged over seeds),
    with a horizontal dashed line at the formal order. Replaces the previous
    scatter, which collapsed to two vertical strips."""
    # We need to recompute by reading grader verdicts (interactive cells store
    # them under <cell>/grader_verdict.json? — actually they don't; we re-grade
    # on the fly to get observed rates).
    sys.path.insert(0, str(ROOT.parent.parent))
    from benchmarks.hoe_v1.grader.grader import grade, load_spec

    obs_by_task_cond: dict[tuple[str, str], list[float]] = defaultdict(list)
    formal_by_task: dict[str, float] = {}
    for r in rows:
        if not r["task_id"].startswith("C"):
            continue
        cell_dir = Path(r["cell_dir"])
        result_path = cell_dir / ("work" / Path("result.txt")) \
            if "eval_interactive" in str(cell_dir) else cell_dir / "work" / "result.txt"
        if not result_path.exists():
            continue
        spec = load_spec(ROOT / "tasks" / f"{r['task_id']}.yaml")
        try:
            v = grade(spec, result_path, seed=r["seed"], run_dir=result_path.parent)
        except Exception:
            continue
        details = v.details or {}
        # single-axis MMS
        if "mean_rate" in details:
            obs = details["mean_rate"]
            formal = details.get("formal_order", 2.0)
        else:
            axes = [d for d in details.values()
                    if isinstance(d, dict) and "mean_rate" in d]
            if not axes:
                continue
            obs = sum(a["mean_rate"] for a in axes) / len(axes)
            formal = sum(a.get("formal_order", 2.0) for a in axes) / len(axes)
        obs_by_task_cond[(r["task_id"], r["condition"])].append(float(obs))
        formal_by_task[r["task_id"]] = float(formal)

    c_tasks = sorted([t for t in TASK_ORDER if t.startswith("C")])
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(c_tasks))
    bar_w = 0.20
    for j, cond in enumerate(COND_ORDER):
        means = []
        errs = []
        for t in c_tasks:
            vals = obs_by_task_cond.get((t, cond), [])
            if vals:
                means.append(float(np.mean(vals)))
                errs.append(float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0)
            else:
                means.append(0.0)
                errs.append(0.0)
        ax.bar(x + (j - 1.5) * bar_w, means, bar_w,
               yerr=errs, capsize=2.5,
               color=COND_COLOR[cond], edgecolor="black",
               linewidth=0.4, label=COND_LABEL[cond],
               error_kw={"elinewidth": 0.6})
    # Reference lines at formal order per task
    for i, t in enumerate(c_tasks):
        p = formal_by_task.get(t, 2.0)
        ax.hlines(p, i - 0.45, i + 0.45, colors="black",
                  linestyles="--", linewidths=1.0,
                  label="formal order $p$" if i == 0 else None)
        ax.text(i, p + 0.06, f"$p={p:g}$", ha="center", va="bottom",
                fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(c_tasks)
    ax.set_xlabel("Tier-C task")
    ax.set_ylabel("observed convergence rate")
    ax.set_ylim(0, 3.6)
    ax.set_title("Tier C convergence rates (dashed = formal order $p$)",
                 fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=5, fontsize=8, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig R3 — behavioural ablation: critic spawn rate per condition
# --------------------------------------------------------------------------- #
def _agent_spawns_per_cell() -> dict[str, dict[str, Any]]:
    """Return {cell_id: {agent_spawns, mcp_calls}} from interactive session
    logs. BARE cells get filled as agent=0, mcp=0 from sandboxed transcripts."""
    out: dict[str, dict[str, Any]] = {}
    if SESSION_DIR.exists():
        for f in SESSION_DIR.glob("*.jsonl"):
            cell = None
            n_agent = n_mcp = 0
            try:
                for line in f.open():
                    try: d = json.loads(line)
                    except: continue
                    if cell is None and d.get("type") == "user":
                        c = d.get("message", {}).get("content", "")
                        if isinstance(c, str):
                            m = re.search(
                                r"eval_interactive/([A-Z]\d+_(?:MCP_[A-Z_]+|BARE)_seed\d+)(?:_rerun)?/", c
                            )
                            if m:
                                cell = m.group(1)
                    msg = d.get("message") or {}
                    for blk in msg.get("content", []) or []:
                        if isinstance(blk, dict) and blk.get("type") == "tool_use":
                            name = blk.get("name", "")
                            if name == "Agent": n_agent += 1
                            if "mcp__" in name: n_mcp += 1
            except Exception:
                continue
            if cell:
                # Multiple sessions can map to the same cell (rerun); keep max
                prev = out.get(cell, {"agent_spawns": 0, "mcp_calls": 0})
                out[cell] = {
                    "agent_spawns": max(prev["agent_spawns"], n_agent),
                    "mcp_calls": max(prev["mcp_calls"], n_mcp),
                }
    return out


def fig_r3_critic_spawn(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Bar chart of (fraction of cells with critic spawned) per condition.
    BARE has no critic instruction and no MCP → 0 by construction.
    """
    spawns = _agent_spawns_per_cell()
    # Aggregate per condition: count cells (interactive only — BARE was sandboxed)
    by_cond_cells: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        cell_id = f"{r['task_id']}_{r['condition']}_seed{r['seed']}"
        if r["condition"] == "BARE":
            by_cond_cells[r["condition"]].append(False)
        else:
            spawn = spawns.get(cell_id)
            by_cond_cells[r["condition"]].append(
                bool(spawn and spawn["agent_spawns"] > 0)
            )

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    x = np.arange(len(COND_ORDER))
    fracs = []
    totals = []
    for cond in COND_ORDER:
        cs = by_cond_cells.get(cond, [])
        n = len(cs)
        k = sum(cs)
        fracs.append(k / max(n, 1))
        totals.append((k, n))
    bars = ax.bar(x, [100*f for f in fracs],
                  color=[COND_COLOR[c] for c in COND_ORDER],
                  edgecolor="black", linewidth=0.4)
    for i, (k, n) in enumerate(totals):
        ax.text(i, 100*fracs[i] + 2, f"{k}/{n}\n({100*fracs[i]:.0f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL_TWO_LINE[c] for c in COND_ORDER], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("reviewer spawn rate (% of cells)")
    ax.set_ylim(0, 105)
    ax.set_title("Reviewer invocation rate by condition", fontsize=10)
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Fig R4 — per-tier bars with paired-delta inset
# --------------------------------------------------------------------------- #
def fig_r4_tier_bars(rows: list[dict[str, Any]], out_path: Path) -> None:
    tiers = ["A", "B", "C", "D"]
    fig, (ax, ax_inset) = plt.subplots(1, 2, figsize=(9.6, 4.4),
                                       gridspec_kw={"width_ratios": [3.2, 1.4]})
    x = np.arange(len(tiers))
    bar_w = 0.20
    for j, cond in enumerate(COND_ORDER):
        means = []; lo_err = []; hi_err = []; ns = []
        for tier in tiers:
            cs = [r for r in rows if r["condition"] == cond and r["tier"] == tier]
            k = sum(1 for r in cs if r["passed"])
            n = len(cs)
            m = k / n if n else 0.0
            lo, hi = _wilson(k, n)
            means.append(m); lo_err.append(m-lo); hi_err.append(hi-m); ns.append((k,n))
        offset = (j - 1.5) * bar_w
        bars = ax.bar(x + offset, means, bar_w,
                      color=COND_COLOR[cond], edgecolor="black",
                      linewidth=0.4, label=COND_LABEL[cond],
                      yerr=[lo_err, hi_err], capsize=2.5,
                      error_kw={"elinewidth": 0.7})
        for i, (kn, m) in enumerate(zip(ns, means)):
            ax.text(x[i] + offset, m + 0.02, f"{kn[0]}/{kn[1]}",
                    ha="center", va="bottom", fontsize=6.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Tier {t}\n{TIER_DESC[t].split('(')[1].rstrip(')')}"
                       for t in tiers], fontsize=8.5)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("pass rate (95% Wilson CI)")
    ax.set_ylim(0, 1.10)
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=4, fontsize=8, frameon=False)
    ax.set_title("Pass rate by difficulty tier", fontsize=10.5)
    ax.grid(True, axis="y", alpha=0.3)
    # Annotate Tier A ceiling
    ax.annotate("ceiling", xy=(0, 1.0), xytext=(0, 0.6),
                ha="center", fontsize=7.5, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.6))

    # Inset: paired-delta bootstrap CIs
    deltas = []
    if STATS_JSON.exists():
        sd = json.loads(STATS_JSON.read_text())
        for key, label in [
            ("bootstrap_BARE_vs_MCP_FULL", "MCP-Full\n− BARE"),
            ("bootstrap_BARE_vs_MCP_NO_CRITIC", "MCP-NoCritic\n− BARE"),
            ("bootstrap_MCP_FULL_vs_MCP_NO_CRITIC", "MCP-Full\n− MCP-NoCritic"),
        ]:
            d = sd.get(key)
            if d:
                deltas.append((label, d["point_estimate"], d["ci_lo"], d["ci_hi"]))
    yy = np.arange(len(deltas))
    ax_inset.errorbar(
        [d[1] for d in deltas], yy,
        xerr=[
            [d[1] - d[2] for d in deltas],
            [d[3] - d[1] for d in deltas],
        ],
        fmt="o", color="black", capsize=3, lw=1.0,
    )
    ax_inset.axvline(0, color="gray", lw=0.5, ls=":")
    ax_inset.set_yticks(yy)
    ax_inset.set_yticklabels([d[0] for d in deltas], fontsize=8)
    ax_inset.set_xlabel("Δ pass rate (bootstrap CI)")
    ax_inset.set_xlim(-0.15, 0.30)
    ax_inset.set_title("Paired deltas vs baseline", fontsize=9)
    ax_inset.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def fig_r4_deltas_only(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Standalone, full-width paired-delta plot for the main text."""
    deltas = []
    if STATS_JSON.exists():
        sd = json.loads(STATS_JSON.read_text())
        for key, label in [
            ("bootstrap_BARE_vs_MCP_FULL",          "MCP-Full vs BARE"),
            ("bootstrap_BARE_vs_MCP_NO_CRITIC",     "MCP-NoCritic vs BARE"),
            ("bootstrap_MCP_FULL_vs_MCP_NO_CRITIC", "MCP-Full vs MCP-NoCritic"),
        ]:
            d = sd.get(key)
            if d:
                deltas.append((label, d["point_estimate"], d["ci_lo"], d["ci_hi"]))
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    yy = np.arange(len(deltas))
    ax.errorbar(
        [d[1] for d in deltas], yy,
        xerr=[[d[1] - d[2] for d in deltas],
              [d[3] - d[1] for d in deltas]],
        fmt="o", color="black", markersize=6,
        capsize=4, lw=1.2,
    )
    for i, d in enumerate(deltas):
        ax.text(d[3] + 0.012, i,
                f"Δ = {d[1]*100:+.1f} pp\n[{d[2]*100:+.1f}, {d[3]*100:+.1f}]",
                va="center", fontsize=8)
    ax.axvline(0, color="gray", lw=0.7, ls=":")
    ax.set_yticks(yy)
    ax.set_yticklabels([d[0] for d in deltas], fontsize=10)
    ax.set_xlabel("change in pooled pass rate (percentage points)\n"
                  "filled marker = point estimate; bar = 95% task-bootstrap interval",
                  fontsize=9)
    ax.set_xlim(-0.18, 0.40)
    ax.set_xticks(np.arange(-0.15, 0.41, 0.05))
    ax.set_xticklabels([f"{int(100*t):+d}" for t in np.arange(-0.15, 0.41, 0.05)])
    ax.set_title("Pairwise differences in pass rate, 10 000-iteration task bootstrap",
                 fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_r5_pass_at_k(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Reliability metric: fraction of tasks passing >= k of 3 seeds, per condition.

    Grouped bar chart. x-axis groups by k in {1, 2, 3}; within each group, one bar
    per condition with the value labelled on top.
    """
    by_ct: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for r in rows:
        by_ct[(r["condition"], r["task_id"])].append(bool(r["passed"]))
    n_tasks = len({t for (_, t) in by_ct})

    ks = [1, 2, 3]
    rho_by_cond: dict[str, list[float]] = {}
    for cond in COND_ORDER:
        per_task_k = [sum(by_ct[(cond, t)])
                      for (cc, t) in by_ct if cc == cond]
        if not per_task_k:
            continue
        rho_by_cond[cond] = [
            sum(1 for k in per_task_k if k >= kk) / len(per_task_k)
            for kk in ks
        ]

    n_cond = len(rho_by_cond)
    bar_w = 0.78 / n_cond
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x_centres = np.arange(len(ks))
    for i, cond in enumerate(rho_by_cond):
        offs = (i - (n_cond - 1) / 2) * bar_w
        bars = ax.bar(
            x_centres + offs, rho_by_cond[cond], width=bar_w,
            color=COND_COLOR[cond], edgecolor="white",
            label=COND_LABEL[cond], linewidth=0.6,
        )
        for b, v in zip(bars, rho_by_cond[cond]):
            ax.text(
                b.get_x() + b.get_width() / 2, v + 0.015,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=8, color=COND_COLOR[cond], fontweight="bold",
            )

    ax.set_xticks(x_centres)
    ax.set_xticklabels(
        [r"$k\!=\!1$ (any seed)",
         r"$k\!=\!2$ (majority)",
         r"$k\!=\!3$ (all seeds)"],
        fontsize=9,
    )
    ax.set_xlabel("passes required out of 3 seeds", fontsize=10)
    ax.set_ylabel(f"share of tasks ({n_tasks} total)", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fontsize=9, frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rows = _load()
    if not rows:
        print("Run aggregate.py first.")
        return 1
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    # Main text:
    fig_r1_heatmap(rows, REPORT_OUT / "fig_R1_heatmap.png")
    fig_r3_critic_spawn(rows, REPORT_OUT / "fig_R3_critic_spawn.png")
    fig_r4_deltas_only(rows, REPORT_OUT / "fig_R4_deltas.png")
    fig_r5_pass_at_k(rows, REPORT_OUT / "fig_R5_pass_at_k.png")
    # Appendix (supplementary):
    fig_r2_tier_c_orders(rows, REPORT_OUT / "fig_SI_tier_c_orders.png")
    fig_r4_tier_bars(rows, REPORT_OUT / "fig_SI_tier_bars.png")
    print(f"Wrote 4 main-text + 2 SI figures to {REPORT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
