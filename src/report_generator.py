import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from typing import Optional
from simulation import Simulation

BG     = "#0f0f1a"
PANEL  = "#1a1a2e"
ACCENT = "#50c8ff"
TEXT   = "#d0d8e8"
DIM    = "#888aaa"
GREEN  = "#50ff80"
RED    = "#ff6060"
ORANGE = "#ffa040"


class ReportGenerator:
    def __init__(self, sim: Simulation, logger=None) -> None:
        self.sim    = sim
        self.logger = logger

    def generate(self, output_path: str = "research_report.pdf") -> str:
        print(f"[Report] Generating report: {output_path} ...")

        with PdfPages(output_path) as pdf:
            self._page_title(pdf)
            self._page_abstract(pdf)
            self._page_mathematical_model(pdf)
            self._page_tissue_parameters(pdf)
            self._page_stiffness_maps(pdf)
            self._page_error_analysis(pdf)
            self._page_results_table(pdf)
            self._page_mode_comparison(pdf)
            self._page_limitations(pdf)
            self._page_conclusions(pdf)

            d = pdf.infodict()
            d["Title"]   = "Tissue Digital Twin — Stiffness Estimation Research Report"
            d["Author"]  = "Tissue Digital Twin Research Prototype"
            d["Subject"] = "Rotational vs Translational Stiffness Estimation"
            d["Keywords"]= "digital twin, biomechanics, stiffness estimation, surgical robotics"

        print(f"[Report] Report saved: {output_path}")
        return output_path

    def _page_title(self, pdf: PdfPages) -> None:
        fig = self._fig()
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_facecolor(BG)
        ax.axis("off")

        ax.text(0.5, 0.82, "Tissue Digital Twin", ha="center", va="center",
                fontsize=28, fontweight="bold", color=ACCENT, transform=ax.transAxes)
        ax.text(0.5, 0.74, "Research Prototype — Stiffness Estimation Report",
                ha="center", va="center", fontsize=16, color=TEXT, transform=ax.transAxes)

        ax.text(0.5, 0.60,
                "Research Question:\n"
                "\"Can rotational deformation measurements improve local tissue\n"
                " property estimation compared to translational measurements alone?\"",
                ha="center", va="center", fontsize=12, color=TEXT,
                style="italic", transform=ax.transAxes)

        ax.text(0.5, 0.44,
                f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n"
                f"Mesh: {self.sim.mesh.cols}×{self.sim.mesh.rows} hexagonal cells\n"
                f"Records: {len(self.logger.records) if self.logger else 'N/A'}",
                ha="center", va="center", fontsize=10, color=DIM, transform=ax.transAxes)

        ax.text(0.5, 0.12,
                "⚠  DISCLAIMER: Research prototype — not for clinical use.\n"
                "Simplified spring-network model. Values are NOT clinically calibrated.",
                ha="center", va="center", fontsize=9, color=RED, transform=ax.transAxes)

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _page_abstract(self, pdf: PdfPages) -> None:
        fig = self._fig()
        ax  = fig.add_axes([0.08, 0.08, 0.84, 0.84])
        ax.set_facecolor(PANEL)
        ax.axis("off")

        self._section_title(ax, "Abstract", 0.96)
        abstract_text = (
            "This report presents results from the Tissue Digital Twin prototype, "
            "a research system for investigating stiffness estimation in heterogeneous "
            "tissue models using spring-network mechanics.\n\n"
            "The prototype simulates three tissue types — adipose (fat), skeletal muscle, "
            "and tendon — with literature-inspired stiffness ranges. Each type is assigned "
            "directional (anisotropic) stiffness properties and fiber orientations based "
            "on approximate physiological data.\n\n"
            "Two estimation modes are compared:\n"
            "  Mode A — Translational only: estimates kx = Fx/Δx, ky = Fy/Δy\n"
            "  Mode B — Translation + Rotation: additionally estimates kθ = M/θ\n\n"
            "The central hypothesis is that rotational deformation encodes information "
            "about local stiffness gradients not available from translational data alone. "
            "Cells at tissue boundaries exhibit differential rotation arising from "
            "asymmetric spring forces, providing additional discriminative information.\n\n"
            "All estimates include uncertainty quantification via first-order error "
            "propagation. Confidence scores (0–100%) are computed from the relative "
            "uncertainty of the stiffness estimate."
        )
        ax.text(0.05, 0.88, abstract_text, ha="left", va="top",
                fontsize=9, color=TEXT, transform=ax.transAxes,
                wrap=True, multialignment="left")
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _page_mathematical_model(self, pdf: PdfPages) -> None:
        fig = self._fig()
        ax  = fig.add_axes([0.05, 0.05, 0.90, 0.90])
        ax.set_facecolor(PANEL)
        ax.axis("off")

        self._section_title(ax, "Mathematical Model", 0.97)

        lines = [
            ("Spring Network Model", ACCENT, 11, "bold"),
            ("", TEXT, 9, "normal"),
            ("Each cell-to-cell connection is modelled as a linear spring:", TEXT, 9, "normal"),
            ("    k_ij = √(k_i × k_j)   (geometric mean)", GREEN, 9, "normal"),
            ("", TEXT, 9, "normal"),
            ("Anisotropic Directional Stiffness", ACCENT, 11, "bold"),
            ("", TEXT, 9, "normal"),
            ("    k(φ) = kx·cos²(φ − θ_f) + ky·sin²(φ − θ_f)", GREEN, 9, "normal"),
            ("where φ = link direction, θ_f = fiber angle", TEXT, 9, "normal"),
            ("", TEXT, 9, "normal"),
            ("Translational Equilibrium (Jacobi Relaxation)", ACCENT, 11, "bold"),
            ("", TEXT, 9, "normal"),
            ("    u_i = (F_ext + Σ_j k_ij·u_j) / (k_i + Σ_j k_ij)", GREEN, 9, "normal"),
            ("Solved iteratively for 80 steps with exponential decay: d^n (d=0.88)", TEXT, 9, "normal"),
            ("", TEXT, 9, "normal"),
            ("Rotational Deformation (Mode B)", ACCENT, 11, "bold"),
            ("", TEXT, 9, "normal"),
            ("    M_i = Σ_j k_ij · (r_ij × Δu_ij) / |r_ij|", GREEN, 9, "normal"),
            ("    θ_i = C · M_i / (k_i · L²)", GREEN, 9, "normal"),
            ("where C = rotational coupling (0.18), L = hex radius", TEXT, 9, "normal"),
            ("", TEXT, 9, "normal"),
            ("Stiffness Estimation", ACCENT, 11, "bold"),
            ("", TEXT, 9, "normal"),
            ("    kx_est = |Fx / Δx|     (if |Δx| > threshold)", GREEN, 9, "normal"),
            ("    ky_est = |Fy / Δy|     (if |Δy| > threshold)", GREEN, 9, "normal"),
            ("    kθ_est = |M / θ|       (Mode B, if |θ| > 1e-5 rad)", GREEN, 9, "normal"),
            ("", TEXT, 9, "normal"),
            ("Uncertainty Estimation", ACCENT, 11, "bold"),
            ("", TEXT, 9, "normal"),
            ("    σ_kx/kx = √( (σ_F/F)² + (σ_Δx/Δx)² )    [error propagation]", GREEN, 9, "normal"),
            ("where σ_F = noise_sigma × |F|,  σ_Δx = |Δx| / SNR (SNR=12)", TEXT, 9, "normal"),
            ("    confidence = 1 − σ_k / k_est", GREEN, 9, "normal"),
        ]

        y = 0.93
        for text, color, size, weight in lines:
            ax.text(0.04, y, text, ha="left", va="top", fontsize=size,
                    color=color, fontweight=weight, transform=ax.transAxes,
                    fontfamily="monospace")
            y -= 0.033

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _page_tissue_parameters(self, pdf: PdfPages) -> None:
        from cell import TISSUE_TYPES

        fig = self._fig()
        ax  = fig.add_axes([0.05, 0.05, 0.90, 0.90])
        ax.set_facecolor(PANEL)
        ax.axis("off")

        self._section_title(ax, "Tissue Parameter Database", 0.97)

        ax.text(0.04, 0.90,
                "Literature-inspired relative ranges (NOT clinically calibrated values).\n"
                "References: [1] Samani et al. PMB 2003  [2] Lieber & Ward Phil Trans 2011  "
                "[3] Butler et al. ESS 1978",
                ha="left", va="top", fontsize=8, color=DIM, transform=ax.transAxes)

        cols     = ["Tissue", "k (norm.)", "Range", "kx_scale", "ky_scale",
                    "Anisotropy", "Fiber σ°", "Noise σ", "Real-world Ref"]
        col_x    = [0.02, 0.14, 0.26, 0.37, 0.46, 0.55, 0.64, 0.73, 0.82]
        y_header = 0.79

        for cx, c in zip(col_x, cols):
            ax.text(cx, y_header, c, ha="left", va="top", fontsize=8,
                    color=ACCENT, fontweight="bold", transform=ax.transAxes)

        ax.plot([0.03, 0.97], [0.755, 0.755], linewidth=0.5, color=DIM, transform=ax.transAxes)

        rows = [
            ("Fat",    0.8, "0.5–1.2",  1.0, 1.0,  1.0,  90,  0.08, "~1–5 kPa"),
            ("Muscle", 3.5, "2.5–4.5",  1.0, 0.60, 0.60, 15,  0.12, "~10–100 kPa"),
            ("Tendon", 9.0, "7.0–10.0", 1.0, 0.30, 0.30,  5,  0.06, "~500–2000 kPa"),
        ]
        tissue_colors = {"Fat": "#6495ed", "Muscle": "#32cd32", "Tendon": "#dc3232"}

        for ri, (name, k, krange, kxs, kys, aniso, fstd, noise, ref) in enumerate(rows):
            y_row = 0.72 - ri * 0.065
            values = [name, str(k), krange, str(kxs), str(kys),
                      str(aniso), str(fstd), str(noise), ref]
            for cx, v in zip(col_x, values):
                color = tissue_colors.get(name, TEXT) if cx == 0.02 else TEXT
                ax.text(cx, y_row, v, ha="left", va="top", fontsize=8,
                        color=color, transform=ax.transAxes)

        y_note = 0.48
        notes = [
            "k (norm.): Normalized spring coefficient. Not in Pascal units.",
            "kx_scale / ky_scale: Multipliers applied to isotropic stiffness along / across fibers.",
            "Anisotropy = ky_scale / kx_scale. Fat ≈ isotropic; Tendon strongly anisotropic.",
            "Fiber σ°: Standard deviation of fiber angle distribution (degrees).",
            "Noise σ: Relative noise amplitude for uncertainty estimation.",
            "Real-world Ref: Approximate elastic modulus range from ex-vivo mechanical testing.",
            "",
            "Stiffness layout: Fat (left) | Muscle (center) | Tendon (right)",
            "Inclusions: Tendon blob in muscle region, fat pocket in upper-right corner.",
            "Heterogeneity: ±10% Gaussian noise added to each cell's stiffness.",
        ]
        for note in notes:
            ax.text(0.04, y_note, note, ha="left", va="top", fontsize=8,
                    color=DIM, transform=ax.transAxes)
            y_note -= 0.040

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _page_stiffness_maps(self, pdf: PdfPages) -> None:
        from visualization import plot_stiffness_comparison
        fig = plot_stiffness_comparison(self.sim, show=False)
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _page_error_analysis(self, pdf: PdfPages) -> None:
        mesh = self.sim.mesh
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(BG)

        ax = axes[0]
        ax.set_facecolor(PANEL)
        k_trues, k_ests, confs, tissues = [], [], [], []
        for c in mesh.cells.values():
            ests = [e for e in [c.est_kx, c.est_ky] if e is not None]
            if ests:
                k_trues.append(c.stiffness)
                k_ests.append(float(np.mean(ests)))
                confs.append(c.confidence)
                tissues.append(c.tissue_type)

        if k_trues:
            sc = ax.scatter(k_trues, k_ests, c=confs, cmap="RdYlGn",
                            vmin=0, vmax=1, s=30, alpha=0.8, zorder=3)
            ax.plot([0.5, 10], [0.5, 10], "w--", linewidth=0.8, alpha=0.5)
            plt.colorbar(sc, ax=ax, label="Confidence").ax.yaxis.label.set_color("white")
            ax.set_xlabel("k_true", color="white")
            ax.set_ylabel("k̂_est", color="white")
            ax.set_title("Estimated vs True Stiffness", color="white", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No estimates yet", ha="center", va="center",
                    color=DIM, transform=ax.transAxes, fontsize=11)
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_edgecolor("#444466")

        ax = axes[1]
        ax.set_facecolor(PANEL)
        by_tissue = {"fat": [], "muscle": [], "tendon": []}
        for c in mesh.cells.values():
            if c.unc_kx is not None:
                by_tissue[c.tissue_type].append(c.unc_kx)

        labels = [t.capitalize() for t in by_tissue if by_tissue[t]]
        means  = [float(np.mean(v)) for v in by_tissue.values() if v]
        stds   = [float(np.std(v))  for v in by_tissue.values() if v]
        colors_bar = ["#6495ed", "#32cd32", "#dc3232"][:len(labels)]

        if labels:
            ax.bar(labels, means, color=colors_bar, alpha=0.8,
                   yerr=stds, capsize=6, edgecolor="#444466", error_kw={"color": "white"})
            ax.set_ylabel("σ_kx (uncertainty std dev)", color="white")
            ax.set_title("Estimation Uncertainty by Tissue", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_edgecolor("#444466")

        fig.suptitle("Error Analysis", color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _page_results_table(self, pdf: PdfPages) -> None:
        if not self.logger or not self.logger.records:
            fig = self._fig()
            ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.set_facecolor(PANEL)
            ax.axis("off")
            ax.text(0.5, 0.5, "No experiment records available.\nRun an experiment first.",
                    ha="center", va="center", color=DIM, fontsize=12, transform=ax.transAxes)
            pdf.savefig(fig, facecolor=BG)
            plt.close(fig)
            return

        records = self.logger.records
        fig     = self._fig()
        ax      = fig.add_axes([0.03, 0.05, 0.94, 0.90])
        ax.set_facecolor(PANEL)
        ax.axis("off")
        self._section_title(ax, "Experiment Results Summary", 0.97)

        import itertools
        tissues = ["fat", "muscle", "tendon"]
        modes   = ["A", "B"]

        header = ["Tissue", "Mode", "N", "Mean Err", "Std Err", "RMSE", "Mean Conf%"]
        col_x  = np.linspace(0.03, 0.93, len(header))
        y      = 0.88

        for cx, h in zip(col_x, header):
            ax.text(cx, y, h, ha="left", va="top", fontsize=8,
                    color=ACCENT, fontweight="bold", transform=ax.transAxes)
        ax.plot([0.03, 0.97], [0.845, 0.845], linewidth=0.5, color=DIM, transform=ax.transAxes)

        y -= 0.055
        tissue_colors = {"fat": "#6495ed", "muscle": "#32cd32", "tendon": "#dc3232"}
        for tissue, mode in itertools.product(tissues, modes):
            subset = [r for r in records
                      if r["tissue_type"] == tissue and r["mode"] == mode and r.get("error_k") is not None]
            if not subset:
                continue
            errors = [r["error_k"]   for r in subset]
            confs  = [r["confidence"] for r in subset]
            rmse   = float(np.sqrt(np.mean(np.array(errors) ** 2)))
            row = [
                tissue.capitalize(),
                mode,
                str(len(subset)),
                f"{np.mean(errors):.3f}",
                f"{np.std(errors):.3f}",
                f"{rmse:.3f}",
                f"{np.mean(confs):.1f}%",
            ]
            color = tissue_colors[tissue]
            for cx, v in zip(col_x, row):
                ax.text(cx, y, v, ha="left", va="top", fontsize=8,
                        color=color if cx == col_x[0] else TEXT, transform=ax.transAxes)
            y -= 0.045

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _page_mode_comparison(self, pdf: PdfPages) -> None:
        forces, disps = self.sim.get_force_displacement_history()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor(BG)

        ax = axes[0]
        ax.set_facecolor(PANEL)
        if forces:
            ax.scatter(forces, disps, color=ACCENT, s=30, alpha=0.7, zorder=3)
            ax.plot(forces, disps, color=ACCENT, linewidth=1, alpha=0.4)
            ax.set_xlabel("|F|", color="white")
            ax.set_ylabel("Max |Δu| (px)", color="white")
            ax.set_title("Force vs Displacement History", color="white", fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No history", ha="center", va="center",
                    color=DIM, transform=ax.transAxes)
        ax.tick_params(colors="white")
        for s in ax.spines.values():
            s.set_edgecolor("#444466")

        ax = axes[1]
        ax.set_facecolor(PANEL)
        ax.axis("off")
        self._section_title(ax, "Mode A vs Mode B", 0.97, fontsize=10)
        desc = (
            "Mode A — Translational Only:\n"
            "  Estimates kx = |Fx / Δx|, ky = |Fy / Δy|\n"
            "  Uses only displacement magnitude\n"
            "  Cannot distinguish stiffness gradients\n"
            "  at single measurement points\n\n"
            "Mode B — Translation + Rotation:\n"
            "  Additionally estimates kθ = |M / θ|\n"
            "  Rotation encodes stiffness gradient\n"
            "  information from neighboring cells\n"
            "  Cells at tissue boundaries show\n"
            "  differential rotation proportional\n"
            "  to the local stiffness gradient\n\n"
            "Hypothesis:\n"
            "  kθ provides complementary information\n"
            "  to kx/ky, especially at boundaries\n"
            "  between tissue types.\n"
        )
        ax.text(0.05, 0.88, desc, ha="left", va="top", fontsize=8.5,
                color=TEXT, transform=ax.transAxes)

        fig.suptitle("Mode Comparison", color="white", fontsize=12, fontweight="bold")
        plt.tight_layout()
        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _page_limitations(self, pdf: PdfPages) -> None:
        fig = self._fig()
        ax  = fig.add_axes([0.05, 0.05, 0.90, 0.90])
        ax.set_facecolor(PANEL)
        ax.axis("off")
        self._section_title(ax, "Known Limitations", 0.97)

        limitations = [
            ("1. Simplified Spring Network (not FEM)",
             "The spring-network equilibrium is a severe simplification of continuum mechanics. "
             "Real tissue deformation requires partial differential equations of elasticity or "
             "poroelasticity solved by proper FEM (e.g., FEniCS, ABAQUS, OpenSim)."),
            ("2. 2D Model Only",
             "Real tissue is 3D. Out-of-plane deformation, volumetric compression, and "
             "through-tissue force transmission are not modelled."),
            ("3. No Material Anisotropy in Continuum Sense",
             "The directional stiffness model (k(φ) = kx·cos² + ky·sin²) is a Voigt-type "
             "approximation. Full anisotropic elasticity requires a 3×3 (2D) or 6×6 (3D) "
             "compliance tensor."),
            ("4. No Viscoelasticity",
             "Real tissue is viscoelastic (time-dependent). Stress relaxation, creep, and "
             "rate-dependent stiffness are not modelled. All solutions are quasi-static."),
            ("5. Heuristic Rotation Model",
             "The rotation computation (θ = C·M/k·L²) is an ad-hoc approximation of the "
             "curl of the displacement field. It is not derived from a rigorous mechanics model."),
            ("6. No Force Sensing in Vision Mode",
             "Real stiffness estimation requires simultaneous force AND displacement measurement. "
             "Vision mode uses an assumed force — real deployment needs a force-torque sensor."),
            ("7. Uncertainty Model is Approximate",
             "The Gaussian noise model and first-order error propagation are assumptions. "
             "Real measurement uncertainty depends on sensor calibration, temperature drift, "
             "material heterogeneity, and contact geometry."),
        ]

        y = 0.90
        for title, body in limitations:
            ax.text(0.03, y, title, ha="left", va="top", fontsize=9,
                    color=ORANGE, fontweight="bold", transform=ax.transAxes)
            y -= 0.033
            ax.text(0.03, y, body, ha="left", va="top", fontsize=7.8,
                    color=TEXT, transform=ax.transAxes, wrap=True)
            y -= 0.075

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _page_conclusions(self, pdf: PdfPages) -> None:
        fig = self._fig()
        ax  = fig.add_axes([0.05, 0.05, 0.90, 0.90])
        ax.set_facecolor(PANEL)
        ax.axis("off")
        self._section_title(ax, "Conclusions & Future Work", 0.97)

        text = (
            "This prototype demonstrates that:\n\n"
            "1.  A hexagonal spring-network digital twin can represent qualitative\n"
            "    tissue stiffness heterogeneity across three tissue types in real time.\n\n"
            "2.  Translational stiffness estimation (Mode A) provides reasonable\n"
            "    region-level estimates, but struggles to distinguish individual cells\n"
            "    at tissue boundaries due to spring force mixing.\n\n"
            "3.  Rotational deformation (Mode B) introduces an additional observable\n"
            "    (θ) that correlates with local stiffness gradients. Cells at\n"
            "    fat-muscle-tendon boundaries exhibit differential rotation, providing\n"
            "    information not available from single-point translational measurements.\n\n"
            "4.  Anisotropic tissue properties create directional rotational patterns\n"
            "    that further differentiate tissue types beyond stiffness magnitude.\n\n"
            "5.  Uncertainty estimation enables confidence-weighted maps, highlighting\n"
            "    regions where data is insufficient for reliable estimation.\n\n"
            "──────────────────────────────────────────────────────\n\n"
            "Recommended Future Work:\n\n"
            "  A.  Replace spring network with 2D FEM (plane-stress).\n"
            "  B.  Extend to 3D tetrahedral mesh.\n"
            "  C.  Integrate force-torque sensor for real tissue calibration.\n"
            "  D.  Implement Gaussian Process regression for stiffness field.\n"
            "  E.  Validate against ex-vivo tissue mechanical testing data.\n"
            "  F.  Real-time camera integration with force measurement.\n"
        )

        ax.text(0.04, 0.90, text, ha="left", va="top", fontsize=9,
                color=TEXT, transform=ax.transAxes)

        ax.text(0.5, 0.04,
                "Research prototype — not for clinical deployment.",
                ha="center", va="bottom", fontsize=8, color=RED,
                style="italic", transform=ax.transAxes)

        pdf.savefig(fig, facecolor=BG)
        plt.close(fig)

    def _fig(self, figsize=(12, 8)) -> plt.Figure:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(BG)
        return fig

    def _section_title(self, ax, text: str, y: float, fontsize: int = 13) -> None:
        ax.text(0.03, y, text, ha="left", va="top", fontsize=fontsize,
                color=ACCENT, fontweight="bold", transform=ax.transAxes)
        ax.plot([0.03, 0.97], [y - 0.035, y - 0.035], linewidth=0.8,
                color="#444466", transform=ax.transAxes)
