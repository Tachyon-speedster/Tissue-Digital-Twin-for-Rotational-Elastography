import numpy as np
from typing import Dict, List, Tuple, Optional
from cell import Cell, TISSUE_TYPES


class HexMesh:

    def __init__(
        self,
        cols:       int   = 14,
        rows:       int   = 10,
        hex_radius: float = 38.0,
        origin:     Tuple[float, float] = (80.0, 60.0),
    ) -> None:

        self.cols       = cols
        self.rows       = rows
        self.hex_radius = hex_radius
        self.origin     = np.array(origin, dtype=float)

        # Derived hex geometry
        self.hex_width  = 2.0 * hex_radius
        self.hex_height = np.sqrt(3.0) * hex_radius

        # Storage
        self.cells:      Dict[int, Cell]          = {}
        self.grid:       Dict[Tuple[int, int], int] = {}
        self.id_to_grid: Dict[int, Tuple[int, int]] = {}

        # Build
        self._build_grid()
        self._assign_tissue_map()
        self._assign_anisotropy()
        self._build_neighbors()


    def _col_row_to_pixel(self, col: int, row: int) -> np.ndarray:
        x = self.origin[0] + col * self.hex_width * 0.75
        y = self.origin[1] + row * self.hex_height + (col % 2) * self.hex_height * 0.5
        return np.array([x, y], dtype=float)

    def _build_grid(self) -> None:
        cell_id = 0
        for col in range(self.cols):
            for row in range(self.rows):
                pos  = self._col_row_to_pixel(col, row)
                cell = Cell(cell_id=cell_id, position=pos.copy())
                self.cells[cell_id]      = cell
                self.grid[(col, row)]    = cell_id
                self.id_to_grid[cell_id] = (col, row)
                cell_id += 1


    def _get_hex_neighbors(self, col: int, row: int) -> List[Tuple[int, int]]:
        if col % 2 == 0:
            directions = [(-1, 0), (+1, 0), (0, -1), (0, +1), (-1, -1), (+1, -1)]
        else:
            directions = [(-1, 0), (+1, 0), (0, -1), (0, +1), (-1, +1), (+1, +1)]
        return [(col + dc, row + dr) for dc, dr in directions]

    def _build_neighbors(self) -> None:
        for (col, row), cell_id in self.grid.items():
            neighbor_ids = []
            for nc, nr in self._get_hex_neighbors(col, row):
                if (nc, nr) in self.grid:
                    neighbor_ids.append(self.grid[(nc, nr)])
            self.cells[cell_id].neighbors = neighbor_ids


    def _assign_tissue_map(self) -> None:
        tendon_center      = np.array(self._col_row_to_pixel(7, 4), dtype=float)
        tendon_blob_radius = self.hex_width * 1.8

        fat_pocket_center  = np.array(self._col_row_to_pixel(11, 1), dtype=float)
        fat_pocket_radius  = self.hex_width * 1.5

        for (col, row), cell_id in self.grid.items():
            cell = self.cells[cell_id]
            pos  = cell.position

            dist_tendon = np.linalg.norm(pos - tendon_center)
            dist_fat    = np.linalg.norm(pos - fat_pocket_center)

            if dist_tendon < tendon_blob_radius:
                tissue = "tendon"
            elif dist_fat < fat_pocket_radius:
                tissue = "fat"
            elif col < 5:
                tissue = "fat"
            elif col < 10:
                tissue = "muscle"
            else:
                tissue = "tendon"

            tprops = TISSUE_TYPES[tissue]
            cell.tissue_type = tissue

            # Stiffness within literature-inspired range + ±10% noise
            k_base = tprops["stiffness"]
            k_lo, k_hi = tprops["stiffness_range"]
            noise  = np.random.uniform(-0.1, 0.1) * k_base
            cell.stiffness = float(np.clip(k_base + noise, k_lo, k_hi))

    def _assign_anisotropy(self) -> None:
        for (col, row), cell_id in self.grid.items():
            cell   = self.cells[cell_id]
            tprops = TISSUE_TYPES[cell.tissue_type]

            # Draw fiber angle from tissue distribution
            mu    = tprops["fiber_angle_mean"]
            sigma = tprops["fiber_angle_std"]
            # Add a small spatial gradient to avoid perfectly uniform regions
            spatial_gradient = col * 2.0 + row * 1.5   # degrees drift
            raw_angle = np.random.normal(mu, sigma) + spatial_gradient * 0.3
            cell.fiber_angle = float(raw_angle % 180.0)

            # Anisotropic stiffness components
            cell.stiffness_kx = cell.stiffness * tprops["kx_scale"]
            cell.stiffness_ky = cell.stiffness * tprops["ky_scale"]


    def get_cell_at_pixel(self, px: float, py: float) -> Optional[Cell]:
        best_cell = None
        best_dist = float("inf")
        query = np.array([px, py])
        for cell in self.cells.values():
            d = np.linalg.norm(cell.position - query)
            if d < best_dist:
                best_dist = d
                best_cell = cell
        if best_dist < self.hex_radius * 1.5:
            return best_cell
        return None

    def reset_all(self) -> None:
        for cell in self.cells.values():
            cell.reset_dynamics()

    def hex_vertices(
        self, cx: float, cy: float, scale: float = 1.0
    ) -> List[Tuple[float, float]]:
        r = self.hex_radius * scale
        return [
            (cx + r * np.cos(np.radians(60 * i)),
             cy + r * np.sin(np.radians(60 * i)))
            for i in range(6)
        ]

    def rotated_hex_vertices(
        self,
        cell:         Cell,
        scale:        float = 1.0,
        use_deformed: bool  = True,
    ) -> List[Tuple[float, float]]:
        pos    = cell.deformed_position if use_deformed else cell.position
        cx, cy = pos
        r      = self.hex_radius * scale
        theta  = cell.rotation
        return [
            (cx + r * np.cos(np.radians(60 * i) + theta),
             cy + r * np.sin(np.radians(60 * i) + theta))
            for i in range(6)
        ]


    def stiffness_array(self) -> np.ndarray:
        arr = np.zeros((self.rows, self.cols))
        for (col, row), cid in self.grid.items():
            arr[row, col] = self.cells[cid].stiffness
        return arr

    def estimated_stiffness_array(self) -> np.ndarray:
        arr = np.full((self.rows, self.cols), np.nan)
        for (col, row), cid in self.grid.items():
            c = self.cells[cid]
            ests = [e for e in [c.est_kx, c.est_ky] if e is not None]
            if ests:
                arr[row, col] = float(np.mean(ests))
        return arr

    def error_array(self) -> np.ndarray:
        arr = np.full((self.rows, self.cols), np.nan)
        for (col, row), cid in self.grid.items():
            c = self.cells[cid]
            ests = [e for e in [c.est_kx, c.est_ky] if e is not None]
            if ests:
                arr[row, col] = abs(float(np.mean(ests)) - c.stiffness)
        return arr

    def confidence_array(self) -> np.ndarray:
        arr = np.zeros((self.rows, self.cols))
        for (col, row), cid in self.grid.items():
            arr[row, col] = self.cells[cid].confidence
        return arr

    def displacement_magnitude_array(self) -> np.ndarray:
        arr = np.zeros((self.rows, self.cols))
        for (col, row), cid in self.grid.items():
            arr[row, col] = np.linalg.norm(self.cells[cid].displacement)
        return arr

    def rotation_array(self) -> np.ndarray:
        arr = np.zeros((self.rows, self.cols))
        for (col, row), cid in self.grid.items():
            arr[row, col] = np.degrees(self.cells[cid].rotation)
        return arr

    def anisotropy_array(self) -> np.ndarray:
        arr = np.ones((self.rows, self.cols))
        for (col, row), cid in self.grid.items():
            c = self.cells[cid]
            arr[row, col] = c.stiffness_kx / (c.stiffness_ky + 1e-9)
        return arr

    def all_cells_list(self) -> List[Cell]:
        return [self.cells[i] for i in sorted(self.cells.keys())]

    def __len__(self) -> int:
        return len(self.cells)

    def __repr__(self) -> str:
        return f"HexMesh({self.cols}×{self.rows}, {len(self.cells)} cells, hex_r={self.hex_radius})"
