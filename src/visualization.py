import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("Qt5Agg")
    except Exception:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import RegularPolygon, FancyArrowPatch
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from typing import Optional, List, Dict
from mesh import HexMesh
from simulation import Simulation

_TISSUE_COLORS = [
    (0.39, 0.58, 0.93),
    (0.20, 0.80, 0.20),
    (0.86, 0.20, 0.20),
]
TISSUE_CMAP   = mcolors.LinearSegmentedColormap.from_list("tissue_stiffness", _TISSUE_COLORS, N=256)
ROTATION_CMAP = plt.cm.coolwarm
ERROR_CMAP    = plt.cm.hot_r
CONF_CMAP     = plt.cm.RdYlGn
ANISO_CMAP    = plt.cm.plasma


def plot_stiffness_comparison(sim: Simulation, show: bool = True) -> plt.Figure:
    mesh = sim.mesh
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.patch.set_facecolor("#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    _draw_hex_heatmap(
        axes[0], mesh,
        values  = {cid: c.stiffness for cid, c in mesh.cells.items()},
        title   = "True Stiffness",
        subtitle= "k_true (normalized)",
        vmin=0.5, vmax=10.0,
        cmap    = TISSUE_CMAP,
    )

    est_values = {}
    for cid, cell in mesh.cells.items():
        ests = [e for e in [cell.est_kx, cell.est_ky] if e is not None]
        est_values[cid] = float(np.mean(ests)) if ests else np.nan

    _draw_hex_heatmap(
        axes[1], mesh,
        values  = est_values,
        title   = "Estimated Stiffness",
        subtitle= "k̂ from kx=Fx/Δx, ky=Fy/Δy",
        vmin=0.5, vmax=10.0,
        cmap    = TISSUE_CMAP,
    )

    err_values = {}
    for cid, cell in mesh.cells.items():
        ests = [e for e in [cell.est_kx, cell.est_ky] if e is not None]
        if ests:
            err_values[cid] = abs(float(np.mean(ests)) - cell.stiffness)
        else:
            err_values[cid] = np.nan

    _draw_hex_heatmap(
        axes[2], mesh,
        values  = err_values,
        title   = "Estimation Error",
        subtitle= "|k̂ - k_true|",
        vmin=0.0, vmax=3.0,
        cmap    = ERROR_CMAP,
    )

    conf_values = {cid: c.confidence * 100.0 for cid, c in mesh.cells.items()}
    _draw_hex_heatmap(
        axes[3], mesh,
        values  = conf_values,
        title   = "Confidence",
        subtitle= "0–100%",
        vmin=0.0, vmax=100.0,
        cmap    = CONF_CMAP,
    )

    rot_values = {cid: np.degrees(c.rotation) for cid, c in mesh.cells.items()}
    _draw_hex_heatmap(
        axes[4], mesh,
        values  = rot_values,
        title   = "Rotational Response",
        subtitle= "θ (degrees) — Mode B",
        vmin=-15.0, vmax=15.0,
        cmap    = ROTATION_CMAP,
    )

    fig.suptitle(
        "Tissue Digital Twin — Full Stiffness Field Reconstruction",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if show:
        plt.show(block=False)
        plt.pause(0.05)

    return fig


def plot_anisotropy(sim: Simulation, show: bool = True) -> plt.Figure:
    mesh = sim.mesh
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    aniso_values = {
        cid: c.stiffness_kx / (c.stiffness_ky + 1e-9)
        for cid, c in mesh.cells.items()
    }
    _draw_hex_heatmap(
        axes[0], mesh,
        values  = aniso_values,
        title   = "Anisotropy Ratio (kx / ky)",
        subtitle= "1.0 = isotropic; >1 = fiber-aligned stiffer",
        vmin=0.5, vmax=4.0,
        cmap    = ANISO_CMAP,
    )

    for cid, cell in mesh.cells.items():
        cx, cy = cell.position
        angle_rad = np.radians(cell.fiber_angle)
        ratio     = min(cell.stiffness_kx / (cell.stiffness_ky + 1e-9), 4.0)
        half_len  = mesh.hex_radius * 0.38 * ratio

        dx = half_len * np.cos(angle_rad)
        dy = half_len * np.sin(angle_rad)

        hex_patch = RegularPolygon(
            (cx, cy), numVertices=6, radius=mesh.hex_radius * 0.90,
            orientation=0,
            facecolor=(*[c / 255 for c in cell.anisotropy_color], 0.55),
            edgecolor=(0.2, 0.2, 0.3, 0.8), linewidth=0.4,
        )
        axes[1].add_patch(hex_patch)

        axes[1].plot(
            [cx - dx, cx + dx], [cy - dy, cy + dy],
            color="white", linewidth=0.8, alpha=0.7, solid_capstyle="round",
        )

    all_pos = np.array([c.position for c in mesh.cells.values()])
    margin  = mesh.hex_radius * 1.5
    axes[1].set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    axes[1].set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    axes[1].set_aspect("equal")
    axes[1].invert_yaxis()
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("Fiber Orientation (glyphs)", color="white", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Line length ∝ anisotropy ratio", color="#aaaacc", fontsize=7)

    fig.suptitle(
        "Tissue Digital Twin — Anisotropic Tissue Model",
        color="white", fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if show:
        plt.show(block=False)
        plt.pause(0.05)

    return fig


def plot_twin_sync_pipeline(sim: Simulation, show: bool = True) -> plt.Figure:
    mesh    = sim.mesh
    forces, disps = sim.get_force_displacement_history()

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f1a")
    gs  = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.45)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    stages = [
        ("Physical\nState",          C("#50c8ff"), "Real tissue deformation"),
        ("Observed\nDeformation",    C("#80ff80"), "Δx, Δy, θ per cell"),
        ("Parameter\nEstimation",    C("#ffa040"), "kx, ky, kθ ± σ"),
        ("Twin\nUpdate",             C("#ff6060"), "Update virtual model"),
        ("Prediction",               C("#c080ff"), "Predict future state"),
        ("Prediction\nError",        C("#ff80ff"), "Compare to truth"),
        ("Convergence\nHistory",     C("#40e0d0"), "RMSE over iterations"),
        ("Twin\nAccuracy",           C("#ffdf80"), "% cells with conf>50%"),
    ]

    for i, (ax, (label, color, desc)) in enumerate(zip(axes, stages)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label, color=color, fontsize=8, fontweight="bold", pad=3)

    ax = axes[0]
    if forces:
        ax.scatter(forces, disps, c=[f / max(forces) for f in forces],
                   cmap="YlOrRd", s=25, alpha=0.8, zorder=3)
        ax.set_xlabel("|F|", color="white", fontsize=7)
        ax.set_ylabel("|Δu| px", color="white", fontsize=7)

    ax = axes[1]
    disp_mags = [np.linalg.norm(c.displacement) for c in mesh.cells.values()]
    if max(disp_mags) > 0:
        ax.hist(disp_mags, bins=15, color="#80ff80", alpha=0.7, edgecolor="#333355")
        ax.set_xlabel("|Δu| px", color="white", fontsize=7)
        ax.set_ylabel("Count", color="white", fontsize=7)

    ax = axes[2]
    unc_vals = [c.unc_kx for c in mesh.cells.values() if c.unc_kx is not None]
    if unc_vals:
        ax.hist(unc_vals, bins=15, color="#ffa040", alpha=0.7, edgecolor="#333355")
        ax.set_xlabel("σ_kx", color="white", fontsize=7)
        ax.set_ylabel("Count", color="white", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No estimates\nyet", ha="center", va="center",
                color="#666688", fontsize=9, transform=ax.transAxes)

    ax = axes[3]
    k_trues, k_ests = [], []
    for c in mesh.cells.values():
        ests = [e for e in [c.est_kx, c.est_ky] if e is not None]
        if ests:
            k_trues.append(c.stiffness)
            k_ests.append(float(np.mean(ests)))
    if k_trues:
        ax.scatter(k_trues, k_ests, s=15, color="#ff6060", alpha=0.7)
        lim = [0.5, 10.0]
        ax.plot(lim, lim, "w--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("k_true", color="white", fontsize=7)
        ax.set_ylabel("k̂", color="white", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No estimates\nyet", ha="center", va="center",
                color="#666688", fontsize=9, transform=ax.transAxes)

    ax = axes[4]
    k_est_vals = []
    for c in mesh.cells.values():
        ests = [e for e in [c.est_kx, c.est_ky] if e is not None]
        if ests:
            k_est_vals.append(float(np.mean(ests)))
    if k_est_vals and forces:
        f_range    = np.linspace(min(forces), max(forces), 30)
        k_mean     = float(np.mean(k_est_vals))
        pred_disp  = f_range / (k_mean + 1e-9) * 18.0
        ax.plot(f_range, pred_disp, color="#c080ff", linewidth=1.5)
        if disps:
            ax.scatter(forces, disps, color="#ffffff", s=12, alpha=0.6, zorder=3)
        ax.set_xlabel("|F|", color="white", fontsize=7)
        ax.set_ylabel("Predicted |Δu|", color="white", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Awaiting\nprediction", ha="center", va="center",
                color="#666688", fontsize=9, transform=ax.transAxes)

    ax = axes[5]
    errs_by_tissue = {"fat": [], "muscle": [], "tendon": []}
    for c in mesh.cells.values():
        ests = [e for e in [c.est_kx, c.est_ky] if e is not None]
        if ests:
            errs_by_tissue[c.tissue_type].append(abs(float(np.mean(ests)) - c.stiffness))
    labels, means, stds = [], [], []
    for ttype, errs in errs_by_tissue.items():
        if errs:
            labels.append(ttype[:3].upper())
            means.append(float(np.mean(errs)))
            stds.append(float(np.std(errs)))
    if labels:
        colors = ["#6495ed", "#32cd32", "#dc3232"][:len(labels)]
        bars   = ax.bar(labels, means, color=colors, alpha=0.8, edgecolor="#444466")
        ax.errorbar(range(len(labels)), means, yerr=stds,
                    fmt="none", color="white", capsize=4, linewidth=1)
        ax.set_ylabel("|k̂ - k_true|", color="white", fontsize=7)

    ax = axes[6]
    if len(forces) >= 2:
        rmse_history = []
        running_k = []
        for i, (f, d) in enumerate(zip(forces, disps)):
            if d > 0.5:
                k_est = f / d * DISP_SCALE_APPROX
                running_k.append(k_est)
            rmse_history.append(float(np.std(running_k)) if len(running_k) > 1 else float("nan"))
        ax.plot(range(len(rmse_history)), rmse_history, color="#40e0d0", linewidth=1.5)
        ax.set_xlabel("Probe #", color="white", fontsize=7)
        ax.set_ylabel("k̂ std dev", color="white", fontsize=7)
    else:
        ax.text(0.5, 0.5, "Probing...\nneed ≥2", ha="center", va="center",
                color="#666688", fontsize=9, transform=ax.transAxes)

    ax = axes[7]
    cells_with_est   = sum(1 for c in mesh.cells.values() if c.est_kx is not None)
    cells_high_conf  = sum(1 for c in mesh.cells.values() if c.confidence > 0.5)
    total_cells      = len(mesh.cells)
    coverage_pct     = 100.0 * cells_with_est  / total_cells
    accuracy_pct     = 100.0 * cells_high_conf / total_cells

    bar_labels = ["Coverage", "High\nConf"]
    bar_vals   = [coverage_pct, accuracy_pct]
    ax.bar(bar_labels, bar_vals, color=["#50c8ff", "#ffdf80"], alpha=0.8, edgecolor="#444466")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%", color="white", fontsize=7)
    for j, v in enumerate(bar_vals):
        ax.text(j, v + 2, f"{v:.0f}%", ha="center", color="white", fontsize=7)

    fig.suptitle(
        "Digital Twin Synchronization Pipeline",
        color="white", fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if show:
        plt.show(block=False)
        plt.pause(0.05)

    return fig

DISP_SCALE_APPROX = 18.0


def plot_force_displacement(
    forces:        List[float],
    displacements: List[float],
    show:          bool = True,
) -> Optional[plt.Figure]:
    if not forces:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")

    ax.scatter(forces, displacements, color="#7ecfff", s=40, alpha=0.7, zorder=3)
    ax.plot(forces, displacements, color="#4499cc", linewidth=1.2, alpha=0.5)

    if len(forces) > 2:
        f_arr = np.array(forces)
        d_arr = np.array(displacements)
        mask  = d_arr > 0.1
        if mask.sum() > 1:
            slope, _ = np.polyfit(d_arr[mask], f_arr[mask], 1)
            d_range  = np.linspace(0, max(d_arr), 50)
            ax.plot(d_range, d_range * slope, "r--", linewidth=1, alpha=0.6,
                    label=f"Fit: k≈{slope:.2f}")
            ax.legend(facecolor="#1a1a2e", edgecolor="#444466",
                      labelcolor="white", fontsize=8)

    ax.set_xlabel("Applied Force |F|",        color="white", fontsize=9)
    ax.set_ylabel("Max Displacement |Δu| (px)", color="white", fontsize=9)
    ax.set_title("Force vs Displacement History", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color="#333355", alpha=0.5, linewidth=0.5)

    plt.tight_layout()
    if show:
        plt.show(block=False)
        plt.pause(0.05)

    return fig


def plot_experiment_results(results, show: bool = True) -> plt.Figure:
    import pandas as pd
    df = results.df.copy()
    df_valid = df.dropna(subset=["kx_est", "ky_est"])

    if df_valid.empty:
        print("[Visualization] No valid experiment data to plot.")
        return None

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    fig.patch.set_facecolor("#0f0f1a")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    df_valid = df_valid.copy()
    df_valid["err_kx"] = (df_valid["kx_est"] - df_valid["kx_true"]).abs()
    df_valid["err_ky"] = (df_valid["ky_est"] - df_valid["ky_true"]).abs()
    df_valid["mean_err"] = (df_valid["err_kx"] + df_valid["err_ky"]) / 2.0

    ax = axes[0]
    tissues = ["fat", "muscle", "tendon"]
    x      = np.arange(len(tissues))
    width  = 0.35
    for mi, mode in enumerate(["A", "B"]):
        rmses  = []
        for tt in tissues:
            sub = df_valid[(df_valid["mode"] == mode) & (df_valid["tissue"] == tt)]
            if len(sub) > 0:
                rmse = float(np.sqrt((sub["err_kx"] ** 2).mean()))
                rmses.append(rmse)
            else:
                rmses.append(0.0)
        bars = ax.bar(x + mi * width, rmses, width, alpha=0.8,
                      color=["#50a0ff", "#50ff80"][mi],
                      label=f"Mode {mode}", edgecolor="#444466")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([t.capitalize() for t in tissues], color="white")
    ax.set_ylabel("RMSE (kx)", color="white", fontsize=8)
    ax.set_title("RMSE by Tissue Type", color="white", fontsize=9, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white", fontsize=7)

    ax = axes[1]
    errs_a = df_valid[df_valid["mode"] == "A"]["mean_err"].values
    errs_b = df_valid[df_valid["mode"] == "B"]["mean_err"].values
    if len(errs_a) and len(errs_b):
        bplot = ax.boxplot([errs_a, errs_b], labels=["Mode A", "Mode B"],
                           patch_artist=True, notch=False,
                           medianprops={"color": "white"})
        for patch, color in zip(bplot["boxes"], ["#50a0ff", "#50ff80"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for item in ["whiskers", "caps", "fliers"]:
            for line in bplot[item]:
                line.set_color("white")
        ax.set_ylabel("Mean |error|", color="white", fontsize=8)
        ax.set_title("Error Distribution\nA vs B", color="white", fontsize=9, fontweight="bold")
        ax.set_xticklabels(["Mode A", "Mode B"], color="white")

    ax = axes[2]
    for ti, tt in enumerate(tissues):
        sub = df_valid[df_valid["tissue"] == tt]["confidence"].values
        if len(sub):
            color = ["#6495ed", "#32cd32", "#dc3232"][ti]
            ax.hist(sub, bins=10, alpha=0.6, color=color, label=tt.capitalize(), density=True)
    ax.set_xlabel("Confidence %", color="white", fontsize=8)
    ax.set_ylabel("Density", color="white", fontsize=8)
    ax.set_title("Confidence by Tissue", color="white", fontsize=9, fontweight="bold")
    ax.legend(facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white", fontsize=7)

    ax = axes[3]
    df_b = df_valid[(df_valid["mode"] == "B")].dropna(subset=["kt_est"])
    if not df_b.empty:
        ax.scatter(df_b["k_true"], df_b["kt_est"], s=20,
                   c=df_b["confidence"], cmap="YlOrRd", alpha=0.7)
        lim = [0.5, 10.5]
        ax.plot(lim, lim, "w--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("k_true", color="white", fontsize=8)
        ax.set_ylabel("kθ_est (Mode B)", color="white", fontsize=8)
        ax.set_title("Rotational kθ vs k_true\n(Mode B)", color="white", fontsize=9, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Run Mode B\nexperiment first", ha="center", va="center",
                color="#666688", fontsize=9, transform=ax.transAxes)

    stats = results.summary_statistics()
    improvement = stats.get("improvement_pct")
    improvement_str = f"{improvement:.1f}%" if improvement is not None else "n/a"

    fig.suptitle(
        f"Experiment Results — Mode B improvement: {improvement_str}",
        color="white", fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if show:
        plt.show(block=False)
        plt.pause(0.05)

    return fig


def _draw_hex_heatmap(
    ax,
    mesh:     HexMesh,
    values:   dict,
    title:    str,
    subtitle: str,
    vmin:     float,
    vmax:     float,
    cmap,
) -> None:
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for cell_id, cell in mesh.cells.items():
        val = values.get(cell_id, np.nan)
        cx, cy = cell.position

        if np.isnan(val):
            color = (0.3, 0.3, 0.3, 0.5)
        else:
            color = cmap(norm(float(np.clip(val, vmin, vmax))))

        hex_patch = RegularPolygon(
            (cx, cy), numVertices=6, radius=mesh.hex_radius * 0.92,
            orientation=0,
            facecolor=color,
            edgecolor=(0.2, 0.2, 0.3, 0.8),
            linewidth=0.5,
        )
        ax.add_patch(hex_patch)

    all_pos = np.array([c.position for c in mesh.cells.values()])
    margin  = mesh.hex_radius * 1.5
    ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
    ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title,    color="white",   fontsize=9,  fontweight="bold", pad=3)
    ax.set_xlabel(subtitle, color="#aaaacc", fontsize=6.5)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=6)
    cbar.outline.set_edgecolor("#444466")


def render_plots_to_surface(sim: Simulation) -> bytes:
    import io
    fig = plot_stiffness_comparison(sim, show=False)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(),
                bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def C(hex_str: str):
    h = hex_str.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
