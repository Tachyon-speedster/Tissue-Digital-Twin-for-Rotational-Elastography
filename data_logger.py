import time
import json
import csv
import os
from typing import Optional, List, Dict, Any
import numpy as np
from cell import Cell
from simulation import Simulation


class DataLogger:

    def __init__(self, output_dir: str = "logs") -> None:
        self.output_dir  = output_dir
        self.records:    List[Dict[str, Any]] = []
        self.start_time: float = time.time()
        self.session_id: str   = time.strftime("%Y%m%d_%H%M%S")

        os.makedirs(output_dir, exist_ok=True)
        print(f"[DataLogger] Session {self.session_id} — output dir: {output_dir}/")

    def log_event(self, sim: Simulation, cell: Cell) -> None:
        if sim.source_cell is None:
            return

        elapsed = time.time() - self.start_time
        f       = sim.force
        disp    = cell.displacement

        ests    = [e for e in [cell.est_kx, cell.est_ky] if e is not None]
        k_mean  = float(np.mean(ests)) if ests else None
        err_k   = abs(k_mean - cell.stiffness) if k_mean is not None else None
        err_kx  = abs(cell.est_kx - cell.stiffness_kx) if cell.est_kx is not None else None
        err_ky  = abs(cell.est_ky - cell.stiffness_ky) if cell.est_ky is not None else None

        record = {
            "timestamp":    round(elapsed, 4),
            "iteration":    sim.iteration,
            "mode":         sim.mode,
            "cell_id":      cell.cell_id,
            "tissue_type":  cell.tissue_type,
            "Fx":           round(float(f[0]),    4),
            "Fy":           round(float(f[1]),    4),
            "|F|":          round(float(np.linalg.norm(f)), 4),
            "delta_x":      round(float(disp[0]), 4),
            "delta_y":      round(float(disp[1]), 4),
            "|delta_u|":    round(float(np.linalg.norm(disp)), 4),
            "theta_deg":    round(float(np.degrees(cell.rotation)), 4),
            "k_true":       round(cell.stiffness,    4),
            "kx_true":      round(cell.stiffness_kx, 4),
            "ky_true":      round(cell.stiffness_ky, 4),
            "fiber_angle":  round(cell.fiber_angle,  2),
            "kx_est":       round(cell.est_kx, 4) if cell.est_kx is not None else None,
            "ky_est":       round(cell.est_ky, 4) if cell.est_ky is not None else None,
            "kt_est":       round(cell.est_kt, 4) if cell.est_kt is not None else None,
            "unc_kx":       round(cell.unc_kx, 4) if cell.unc_kx is not None else None,
            "unc_ky":       round(cell.unc_ky, 4) if cell.unc_ky is not None else None,
            "unc_kt":       round(cell.unc_kt, 4) if cell.unc_kt is not None else None,
            "confidence":   round(cell.confidence_pct, 2),
            "error_k":      round(err_k,  4) if err_k  is not None else None,
            "error_kx":     round(err_kx, 4) if err_kx is not None else None,
            "error_ky":     round(err_ky, 4) if err_ky is not None else None,
        }
        self.records.append(record)

    def export_csv(self, filename: Optional[str] = None) -> str:
        if not self.records:
            print("[DataLogger] No records to export.")
            return ""

        if filename is None:
            filename = f"experiment_{self.session_id}.csv"

        path = os.path.join(self.output_dir, filename)
        fieldnames = list(self.records[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        print(f"[DataLogger] CSV exported: {path}  ({len(self.records)} rows)")
        return path

    def export_json(self, filename: Optional[str] = None) -> str:
        if not self.records:
            print("[DataLogger] No records to export.")
            return ""

        if filename is None:
            filename = f"experiment_{self.session_id}.json"

        path = os.path.join(self.output_dir, filename)

        # Compute per-tissue statistics
        tissue_stats: Dict[str, Dict] = {}
        for tissue in ["fat", "muscle", "tendon"]:
            subset = [r for r in self.records if r["tissue_type"] == tissue and r["error_k"] is not None]
            if subset:
                errors = [r["error_k"]   for r in subset]
                confs  = [r["confidence"] for r in subset]
                tissue_stats[tissue] = {
                    "n_records":     len(subset),
                    "mean_error_k":  round(float(np.mean(errors)),  4),
                    "std_error_k":   round(float(np.std(errors)),   4),
                    "rmse_k":        round(float(np.sqrt(np.mean(np.array(errors) ** 2))), 4),
                    "mean_confidence": round(float(np.mean(confs)), 2),
                }

        # Mode comparison
        mode_stats: Dict[str, Dict] = {}
        for mode in ["A", "B"]:
            subset = [r for r in self.records if r["mode"] == mode and r["error_k"] is not None]
            if subset:
                errors = [r["error_k"] for r in subset]
                mode_stats[mode] = {
                    "n_records":  len(subset),
                    "mean_error": round(float(np.mean(errors)), 4),
                    "std_error":  round(float(np.std(errors)),  4),
                    "rmse":       round(float(np.sqrt(np.mean(np.array(errors) ** 2))), 4),
                }

        improvement = None
        if "A" in mode_stats and "B" in mode_stats:
            rmse_a = mode_stats["A"]["rmse"]
            rmse_b = mode_stats["B"]["rmse"]
            if rmse_a > 0:
                improvement = round((rmse_a - rmse_b) / rmse_a * 100.0, 2)

        report = {
            "metadata": {
                "session_id":      self.session_id,
                "n_records":       len(self.records),
                "duration_s":      round(time.time() - self.start_time, 2),
                "research_question": (
                    "Can rotational deformation measurements improve local tissue "
                    "property estimation compared to translational measurements alone?"
                ),
                "disclaimer": (
                    "Research prototype. Not clinically validated. "
                    "Spring-network model, not FEM."
                ),
            },
            "tissue_statistics":  tissue_stats,
            "mode_statistics":    mode_stats,
            "mode_B_improvement_pct": improvement,
            "records":            self.records,
        }

        def _convert(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            return obj

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=_convert)

        print(f"[DataLogger] JSON report exported: {path}")
        return path

    def summary(self) -> str:
        n = len(self.records)
        if n == 0:
            return "No events logged yet."
        errors = [r["error_k"] for r in self.records if r["error_k"] is not None]
        mean_err = f"{np.mean(errors):.3f}" if errors else "n/a"
        confs    = [r["confidence"] for r in self.records]
        mean_conf = f"{np.mean(confs):.0f}%" if confs else "n/a"
        return f"Logged {n} events | mean_err={mean_err} | conf={mean_conf}"

    def clear(self) -> None:
        self.records.clear()
        print("[DataLogger] Log cleared.")
