
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from cell import Cell, ROTATIONAL_COUPLING, TISSUE_TYPES
from mesh import HexMesh

PROPAGATION_ITERATIONS = 40
DECAY_FACTOR           = 0.88
DISPLACEMENT_SCALE     = 18.0
FORCE_MAGNITUDE_MAX    = 20.0
MIN_DISP_THRESHOLD     = 0.5
NOISE_AMPLITUDE        = 0.025


class Simulation:

    def __init__(self, mesh: HexMesh) -> None:
        self.mesh            = mesh
        self.source_cell:    Optional[Cell] = None
        self.force:          np.ndarray = np.zeros(2)
        self.mode:           str = "B"
        self.use_anisotropy: bool = True
        self.iteration:      int = 0
        self.history:        List[Tuple[float, float]] = []

        self._n       = len(mesh.cells)
        self._cells   = [mesh.cells[i] for i in sorted(mesh.cells.keys())]
        self._id_to_idx = {c.cell_id: i for i, c in enumerate(self._cells)}

        self._edge_i:  np.ndarray
        self._edge_j:  np.ndarray
        self._k_ij:    np.ndarray
        self._sum_k:   np.ndarray
        self._stiff:   np.ndarray
        self._denom:   np.ndarray
        self._r_vecs:  np.ndarray
        self._r_mags:  np.ndarray

        self._dist_cache: Dict[int, np.ndarray] = {}
        self._precompute()

    def _precompute(self) -> None:
        edges_i, edges_j, k_vals = [], [], []
        for i, cell in enumerate(self._cells):
            for nbr_id in cell.neighbors:
                j   = self._id_to_idx[nbr_id]
                nbr = self._cells[j]
                if self.use_anisotropy:
                    ki  = self._dir_stiff(cell, nbr.position - cell.position)
                    kj  = self._dir_stiff(nbr,  cell.position - nbr.position)
                    k   = np.sqrt(ki * kj)
                else:
                    k   = np.sqrt(cell.stiffness * nbr.stiffness)
                edges_i.append(i)
                edges_j.append(j)
                k_vals.append(k)

        self._edge_i = np.array(edges_i, dtype=np.int32)
        self._edge_j = np.array(edges_j, dtype=np.int32)
        self._k_ij   = np.array(k_vals,  dtype=np.float64)
        self._stiff  = np.array([c.stiffness for c in self._cells], dtype=np.float64)
        self._sum_k  = np.zeros(self._n, dtype=np.float64)
        np.add.at(self._sum_k, self._edge_i, self._k_ij)
        self._denom  = self._stiff + self._sum_k

        pos          = np.array([c.position for c in self._cells], dtype=np.float64)
        self._r_vecs = pos[self._edge_j] - pos[self._edge_i]
        self._r_mags = np.linalg.norm(self._r_vecs, axis=1) + 1e-9
        self._dist_cache.clear()

    def _dir_stiff(self, cell: Cell, direction: np.ndarray) -> float:
        mag = np.linalg.norm(direction)
        if mag < 1e-9:
            return cell.stiffness
        phi     = np.arctan2(direction[1], direction[0])
        theta_f = np.radians(cell.fiber_angle)
        d       = phi - theta_f
        return cell.stiffness_kx * np.cos(d)**2 + cell.stiffness_ky * np.sin(d)**2

    def rebuild_springs(self) -> None:
        self._precompute()
        self._dist_cache.clear()

    def _bfs_dist_array(self, src_idx: int) -> np.ndarray:
        if src_idx in self._dist_cache:
            return self._dist_cache[src_idx]
        dist = np.full(self._n, 9999, dtype=np.int32)
        dist[src_idx] = 0
        queue = deque([src_idx])
        adj: Dict[int, List[int]] = {i: [] for i in range(self._n)}
        for ei, ej in zip(self._edge_i, self._edge_j):
            adj[ei].append(ej)
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if dist[nbr] == 9999:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        self._dist_cache[src_idx] = dist
        return dist

    def apply_force(self, cell_id: int, force: np.ndarray) -> None:
        self.mesh.reset_all()
        self.source_cell = self.mesh.cells[cell_id]
        self.force       = force.copy()
        self.iteration  += 1
        self.source_cell.force = force.copy()

        self._solve_displacement_fast()

        rotation_active = (self.mode == "B")
        for cell in self._cells:
            cell.use_rotation = rotation_active

        if self.mode == "B":
            self._compute_rotations_fast()

        for cell in self._cells:
            cell.update_estimates()

        max_disp = float(np.max(np.linalg.norm(
            np.array([c.displacement for c in self._cells]), axis=1)))
        self.history.append((float(np.linalg.norm(force)), max_disp))
        if len(self.history) > 40:
            self.history.pop(0)

    def _solve_displacement_fast(self) -> None:
        src_idx   = self._id_to_idx[self.source_cell.cell_id]
        k_src     = self.source_cell.stiffness

        init_disp_vis  = self.force * DISPLACEMENT_SCALE / k_src
        init_disp_phys = self.force / k_src

        dist_arr   = self._bfs_dist_array(src_idx)
        decay_arr  = DECAY_FACTOR ** dist_arr.astype(np.float64)

        disp_vis = np.zeros((self._n, 2), dtype=np.float64)
        disp_vis[src_idx] = init_disp_vis

        ei = self._edge_i
        ej = self._edge_j
        kv = self._k_ij

        for _ in range(PROPAGATION_ITERATIONS):
            weighted  = kv[:, None] * disp_vis[ej]
            spring    = np.zeros((self._n, 2))
            np.add.at(spring, ei, weighted)
            new_disp  = spring / self._denom[:, None]
            new_disp *= decay_arr[:, None]
            new_disp[src_idx] = init_disp_vis
            disp_vis  = new_disp

        mags   = np.linalg.norm(disp_vis, axis=1)
        sigmas = NOISE_AMPLITUDE * mags + 0.015
        rng    = np.random.default_rng(self.iteration)
        noise  = rng.normal(0.0, sigmas[:, None], size=disp_vis.shape)
        disp_vis += noise
        disp_vis[src_idx] = init_disp_vis

        for i, cell in enumerate(self._cells):
            cell.displacement = disp_vis[i].copy()

            if i == src_idx:
                cell.physics_disp = init_disp_phys.copy()
            else:
                cell.physics_disp = disp_vis[i] / DISPLACEMENT_SCALE
                cell.force = cell.stiffness * cell.physics_disp

    def _compute_rotations_fast(self) -> None:
        disp   = np.array([c.displacement for c in self._cells])
        ei     = self._edge_i
        ej     = self._edge_j
        kv     = self._k_ij
        r_vecs = self._r_vecs
        r_mags = self._r_mags
        L      = self.mesh.hex_radius

        delta_u = disp[ej] - disp[ei]
        cross   = r_vecs[:, 0] * delta_u[:, 1] - r_vecs[:, 1] * delta_u[:, 0]

        torque  = kv * cross / r_mags
        moment  = np.zeros(self._n)
        np.add.at(moment, ei, torque)

        theta   = ROTATIONAL_COUPLING * moment / (self._stiff * L + 1e-9)
        theta   = np.clip(theta, -0.6, 0.6)

        for i, cell in enumerate(self._cells):
            cell.rotation   = float(theta[i])
            cell.moment     = float(moment[i] * ROTATIONAL_COUPLING / L)

    def set_mode(self, mode: str) -> None:
        assert mode in ("A", "B")
        self.mode = mode
        if mode == "A":
            for cell in self._cells:
                cell.rotation = 0.0
                cell.moment   = 0.0
                cell.est_kt   = None
                cell.unc_kt   = None
        if self.source_cell is not None and np.linalg.norm(self.force) > 1e-6:
            self.apply_force(self.source_cell.cell_id, self.force)

    def set_anisotropy(self, enabled: bool) -> None:
        self.use_anisotropy = enabled
        self.rebuild_springs()
        if self.source_cell is not None and np.linalg.norm(self.force) > 1e-6:
            self.apply_force(self.source_cell.cell_id, self.force)

    def get_measurement_summary(self, cell: Cell) -> Dict:
        kx_str = (f"{cell.est_kx:.3f} ± {cell.unc_kx:.3f}"
                  if cell.est_kx is not None and cell.unc_kx is not None else "n/a")
        ky_str = (f"{cell.est_ky:.3f} ± {cell.unc_ky:.3f}"
                  if cell.est_ky is not None and cell.unc_ky is not None else "n/a")
        kt_str = (f"{cell.est_kt:.3f} ± {cell.unc_kt:.3f}"
                  if cell.est_kt is not None and cell.unc_kt is not None else "n/a")
        return {
            "Fx": float(self.force[0]), "Fy": float(self.force[1]),
            "delta_x": float(cell.displacement[0]),
            "delta_y": float(cell.displacement[1]),
            "theta_deg": float(np.degrees(cell.rotation)),
            "kx_est": cell.est_kx, "ky_est": cell.est_ky, "kt_est": cell.est_kt,
            "unc_kx": cell.unc_kx, "unc_ky": cell.unc_ky, "unc_kt": cell.unc_kt,
            "kx_str": kx_str, "ky_str": ky_str, "kt_str": kt_str,
            "k_true": cell.stiffness,
            "kx_true": cell.stiffness_kx, "ky_true": cell.stiffness_ky,
            "fiber_angle": cell.fiber_angle,
            "tissue": cell.tissue_type, "confidence": cell.confidence_pct,
        }

    def get_active_cell(self) -> Optional[Cell]:
        return self.source_cell

    def get_force_displacement_history(self) -> Tuple[List[float], List[float]]:
        if not self.history:
            return [], []
        return [h[0] for h in self.history], [h[1] for h in self.history]

    def clear(self) -> None:
        self.mesh.reset_all()
        self.source_cell = None
        self.force       = np.zeros(2)
        self.history.clear()

    def run_experiment(
        self,
        force_levels:     List[float]      = None,
        force_directions: List[np.ndarray] = None,
        target_cells:     List[int]        = None,
        modes:            List[str]        = None,
    ) -> "ExperimentResults":
        import pandas as pd
        if force_levels     is None: force_levels = [4.0, 8.0, 12.0, 16.0, 20.0]
        if force_directions is None:
            force_directions = [np.array([1.0, 0.0]),
                                np.array([0.0, 1.0]),
                                np.array([0.707, 0.707])]
        if target_cells     is None: target_cells = self._sample_representative_cells(12)
        if modes            is None: modes = ["A", "B"]
        records    = []
        saved_mode = self.mode
        for mode in modes:
            self.set_mode(mode)
            for cell_id in target_cells:
                cell = self.mesh.cells[cell_id]
                for fmag in force_levels:
                    for fdir in force_directions:
                        self.apply_force(cell_id, fdir * fmag)
                        records.append({
                            "mode": mode, "cell_id": cell_id,
                            "tissue": cell.tissue_type,
                            "force_mag": fmag,
                            "k_true": cell.stiffness,
                            "kx_true": cell.stiffness_kx, "ky_true": cell.stiffness_ky,
                            "fiber_angle": cell.fiber_angle,
                            "kx_est": cell.est_kx, "ky_est": cell.est_ky,
                            "kt_est": cell.est_kt,
                            "unc_kx": cell.unc_kx, "unc_ky": cell.unc_ky,
                            "unc_kt": cell.unc_kt,
                            "confidence": cell.confidence_pct,
                            "delta_x": float(cell.displacement[0]),
                            "delta_y": float(cell.displacement[1]),
                            "theta_deg": float(np.degrees(cell.rotation)),
                            "error_kx": abs(cell.est_kx - cell.stiffness_kx) if cell.est_kx else None,
                            "error_ky": abs(cell.est_ky - cell.stiffness_ky) if cell.est_ky else None,
                        })
        self.set_mode(saved_mode)
        self.clear()
        return ExperimentResults(pd.DataFrame(records))

    def _sample_representative_cells(self, n: int = 12) -> List[int]:
        by_tissue: Dict[str, List[int]] = {"fat": [], "muscle": [], "tendon": []}
        for cid, cell in self.mesh.cells.items():
            by_tissue[cell.tissue_type].append(cid)
        sampled = []
        per_type = max(1, n // 3)
        for cids in by_tissue.values():
            if cids:
                chosen = np.random.choice(cids, size=min(per_type, len(cids)), replace=False)
                sampled.extend(chosen.tolist())
        return sampled[:n]


class ExperimentResults:
    def __init__(self, df) -> None:
        self.df = df

    def summary_statistics(self) -> Dict:
        df = self.df.dropna(subset=["kx_est", "ky_est"]).copy()
        df["rmse_kx"] = (df["kx_est"] - df["kx_true"]) ** 2
        df["rmse_ky"] = (df["ky_est"] - df["ky_true"]) ** 2

        def rmse(arr):
            return float(np.sqrt(arr.mean())) if len(arr) else None

        rmse_a = rmse(df[df["mode"] == "A"][["rmse_kx", "rmse_ky"]].values.flatten())
        rmse_b = rmse(df[df["mode"] == "B"][["rmse_kx", "rmse_ky"]].values.flatten())
        improvement = ((rmse_a - rmse_b) / rmse_a * 100.0
                       if rmse_a and rmse_b and rmse_a > 0 else None)
        return {
            "n_total": len(self.df), "n_valid": len(df),
            "rmse_mode_A": rmse_a, "rmse_mode_B": rmse_b,
            "improvement_pct": improvement,
        }

    def export_csv(self, path: str) -> None:
        self.df.to_csv(path, index=False)
        print(f"[Experiment] CSV saved: {path}")

    def export_json(self, path: str) -> None:
        import json
        stats = self.summary_statistics()
        def _c(o):
            if isinstance(o, (np.floating, float)): return float(o)
            if isinstance(o, (np.integer, int)):    return int(o)
            if isinstance(o, dict): return {str(k): _c(v) for k, v in o.items()}
            return o
        with open(path, "w") as f:
            json.dump(_c(stats), f, indent=2)
        print(f"[Experiment] JSON saved: {path}")
