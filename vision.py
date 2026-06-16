import numpy as np
import cv2
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


# ── Stream URL helpers ────────────────────────────────────────────────────────

def build_stream_url(source_type: str, ip: str = "192.168.1.100") -> str:
    """
    Build the video stream URL for common Android webcam apps.

    Parameters
    ----------
    source_type : One of 'droidcam', 'ipwebcam', 'rtsp', 'iriun', 'device'
    ip          : Local IP address of the smartphone on the same WiFi network

    Returns
    -------
    URL string or integer device index
    """
    urls = {
        "droidcam":  f"http://{ip}:4747/video",
        "ipwebcam":  f"http://{ip}:8080/video",
        "rtsp":      f"rtsp://{ip}:8554/stream",
        "iriun":     "/dev/video0",       # Iriun typically maps here on Linux
        "device0":   0,
        "device1":   1,
    }
    return urls.get(source_type, 0)


# ── Feature tracking ──────────────────────────────────────────────────────────

@dataclass
class TrackedPoint:
    """A single feature point being tracked across frames."""
    pt_id:        int
    ref_position: np.ndarray    # (x, y) in reference frame
    cur_position: np.ndarray    # (x, y) in current frame
    displacement: np.ndarray    # cur - ref
    is_active:    bool = True


class OpticalFlowTracker:
    """
    Tracks feature points using Lucas-Kanade optical flow.

    Workflow:
    ---------
    1. Call set_reference(frame) to capture the rest state.
    2. Call update(frame) each new frame to get displacement field.
    3. Call get_displacement_field() to get per-cell displacement estimates.
    """

    # LK optical flow parameters
    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    # Shi-Tomasi feature detection parameters
    FEATURE_PARAMS = dict(
        maxCorners=120,
        qualityLevel=0.02,
        minDistance=12,
        blockSize=7,
    )

    def __init__(self) -> None:
        self.ref_frame_gray: Optional[np.ndarray] = None
        self.ref_points:     Optional[np.ndarray] = None    # shape (N, 1, 2)
        self.tracked_points: List[TrackedPoint] = []
        self.current_frame:  Optional[np.ndarray] = None
        self.has_reference:  bool = False

        # Grid for visualization
        self.frame_shape: Optional[Tuple[int, int]] = None

    def set_reference(self, frame: np.ndarray) -> int:
        """
        Set the reference (rest) frame and detect initial feature points.

        Parameters
        ----------
        frame : BGR frame from camera

        Returns
        -------
        Number of feature points detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        self.ref_frame_gray = gray
        self.frame_shape = gray.shape
        self.current_frame = frame.copy()

        # Detect corners as initial tracking points
        corners = cv2.goodFeaturesToTrack(gray, **self.FEATURE_PARAMS)

        if corners is None or len(corners) == 0:
            self.ref_points = None
            self.has_reference = False
            return 0

        self.ref_points = corners.copy()

        # Initialize tracked point records
        self.tracked_points = []
        for i, pt in enumerate(corners):
            pos = pt[0]  # (x, y)
            self.tracked_points.append(TrackedPoint(
                pt_id=i,
                ref_position=pos.copy(),
                cur_position=pos.copy(),
                displacement=np.zeros(2),
            ))

        self.has_reference = True
        return len(corners)

    def update(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Track feature points in a new frame using Lucas-Kanade optical flow.

        Parameters
        ----------
        frame : New BGR frame

        Returns
        -------
        Dict with keys:
          'displacements'     : (N, 2) array of displacement vectors
          'positions'         : (N, 2) array of current point positions
          'ref_positions'     : (N, 2) array of reference positions
          'rotations'         : (M,) array of local rotation estimates
          'active_mask'       : (N,) boolean mask of successfully tracked points
        """
        if not self.has_reference or self.ref_points is None:
            return self._empty_result()

        self.current_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Forward track: ref → cur
        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.ref_frame_gray, gray, self.ref_points, None, **self.LK_PARAMS
        )

        if cur_pts is None:
            return self._empty_result()

        active = status.ravel().astype(bool)

        # Update tracked point records
        displacements = []
        positions = []
        ref_positions = []

        for i, tp in enumerate(self.tracked_points):
            if i < len(active) and active[i]:
                cur_pos = cur_pts[i][0]
                disp = cur_pos - tp.ref_position
                tp.cur_position = cur_pos
                tp.displacement = disp
                tp.is_active = True
            else:
                tp.is_active = False

            displacements.append(tp.displacement)
            positions.append(tp.cur_position)
            ref_positions.append(tp.ref_position)

        displacements = np.array(displacements)
        positions = np.array(positions)
        ref_positions = np.array(ref_positions)

        # Estimate local rotations from displacement field curl
        rotations = self._estimate_local_rotations(ref_positions, displacements, active)

        return {
            'displacements': displacements,
            'positions':     positions,
            'ref_positions': ref_positions,
            'rotations':     rotations,
            'active_mask':   active,
        }

    def _estimate_local_rotations(
        self,
        positions: np.ndarray,
        displacements: np.ndarray,
        active: np.ndarray,
        n_clusters: int = 5,
        search_radius: float = 60.0,
    ) -> np.ndarray:
        """
        Estimate local rotation from displacement field using rigid body fitting.

        For each cluster of nearby points, fit a rigid body transformation
        (translation + rotation) and extract the rotation component.

        Method: For a rigid body motion, if we have points p_i → p_i + d_i,
        we can estimate rotation θ by minimizing:
            Σ_i |R(θ) * (p_i - centroid) - (p_i + d_i - centroid - T)|²

        Here we use the curl of the displacement field as an approximation:
            θ_local ≈ (1/2) * (∂u_y/∂x - ∂u_x/∂y)

        This is the infinitesimal rotation tensor component.
        """
        active_pos = positions[active]
        active_disp = displacements[active]

        if len(active_pos) < 4:
            return np.zeros(n_clusters)

        # Sample cluster centers from active points
        cluster_indices = np.linspace(0, len(active_pos) - 1, n_clusters, dtype=int)
        rotations = []

        for idx in cluster_indices:
            center = active_pos[idx]

            # Find points within search radius
            dists = np.linalg.norm(active_pos - center, axis=1)
            nearby = dists < search_radius

            if nearby.sum() < 3:
                rotations.append(0.0)
                continue

            pts = active_pos[nearby]
            disps = active_disp[nearby]

            # Fit affine transformation: disp ≈ A * (pt - centroid)
            # Extract antisymmetric part → rotation
            centroid = pts.mean(axis=0)
            P = pts - centroid     # (N, 2)
            D = disps              # (N, 2)

            # Least-squares: D = P * A^T  → A = (P^T P)^(-1) P^T D
            try:
                PtP = P.T @ P
                if abs(np.linalg.det(PtP)) < 1e-6:
                    rotations.append(0.0)
                    continue
                A = np.linalg.inv(PtP) @ (P.T @ D)
                # Rotation component: θ = (A[1,0] - A[0,1]) / 2
                theta = (A[1, 0] - A[0, 1]) / 2.0
                rotations.append(float(np.clip(theta, -0.3, 0.3)))
            except np.linalg.LinAlgError:
                rotations.append(0.0)

        return np.array(rotations)

    def _empty_result(self) -> Dict[str, np.ndarray]:
        return {
            'displacements': np.zeros((0, 2)),
            'positions':     np.zeros((0, 2)),
            'ref_positions': np.zeros((0, 2)),
            'rotations':     np.zeros(0),
            'active_mask':   np.zeros(0, dtype=bool),
        }

    def draw_tracking_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw feature points and displacement vectors on the frame.

        Parameters
        ----------
        frame : BGR frame to annotate (in-place)

        Returns
        -------
        Annotated frame
        """
        vis = frame.copy()

        for tp in self.tracked_points:
            if not tp.is_active:
                continue

            ref = tuple(tp.ref_position.astype(int))
            cur = tuple(tp.cur_position.astype(int))
            disp_mag = np.linalg.norm(tp.displacement)

            # Color by displacement magnitude: blue=small, green=medium, red=large
            t = min(disp_mag / 30.0, 1.0)
            if t < 0.5:
                s = t / 0.5
                color = (int(200 * (1 - s)), int(200 * s), 200)
            else:
                s = (t - 0.5) / 0.5
                color = (0, int(200 * (1 - s)), int(200 * s))

            # Reference point (hollow circle)
            cv2.circle(vis, ref, 3, (180, 180, 180), 1)

            # Current position (filled dot)
            cv2.circle(vis, cur, 4, color, -1)

            # Displacement arrow
            if disp_mag > 1.5:
                cv2.arrowedLine(vis, ref, cur, color, 1, tipLength=0.3)

        # Status text
        n_active = sum(1 for tp in self.tracked_points if tp.is_active)
        cv2.putText(
            vis, f"Tracking: {n_active}/{len(self.tracked_points)} pts",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 100), 1
        )

        if self.has_reference:
            cv2.putText(
                vis, "REF SET — deform object",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 100), 1
            )
        else:
            cv2.putText(
                vis, "Press R to set reference frame",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 100, 100), 1
            )

        return vis


# ── Deformation mapper ────────────────────────────────────────────────────────

class DeformationMapper:
    """
    Maps tracked optical flow displacements onto the hexagonal mesh cells.

    Uses inverse-distance weighting (IDW) to interpolate displacement
    from sparse tracked points to each cell centroid.

    IDW formula:
        u_cell = Σ_i (w_i * u_i) / Σ_i w_i
        w_i = 1 / (d_i^p + ε)

    where d_i is the distance from tracked point i to the cell centroid,
    p is the power parameter (typically 2), and ε prevents division by zero.
    """

    def __init__(
        self,
        search_radius: float = 80.0,
        idw_power:     float = 2.0,
    ) -> None:
        self.search_radius = search_radius
        self.idw_power = idw_power

        # Coordinate normalization (camera frame → mesh frame)
        self.cam_to_mesh_scale: float = 1.0
        self.cam_to_mesh_offset: np.ndarray = np.zeros(2)

    def set_coordinate_mapping(
        self,
        cam_bounds: Tuple[float, float, float, float],   # (x0, y0, x1, y1)
        mesh_bounds: Tuple[float, float, float, float],
    ) -> None:
        """
        Set up the coordinate transform from camera pixels to mesh pixels.

        Parameters
        ----------
        cam_bounds  : (xmin, ymin, xmax, ymax) in camera frame
        mesh_bounds : (xmin, ymin, xmax, ymax) in mesh frame
        """
        cam_w  = cam_bounds[2] - cam_bounds[0]
        cam_h  = cam_bounds[3] - cam_bounds[1]
        mesh_w = mesh_bounds[2] - mesh_bounds[0]
        mesh_h = mesh_bounds[3] - mesh_bounds[1]

        self.cam_to_mesh_scale = min(mesh_w / cam_w, mesh_h / cam_h)
        self.cam_to_mesh_offset = np.array([
            mesh_bounds[0] - cam_bounds[0] * self.cam_to_mesh_scale,
            mesh_bounds[1] - cam_bounds[1] * self.cam_to_mesh_scale,
        ])

    def cam_to_mesh(self, cam_pt: np.ndarray) -> np.ndarray:
        """Convert a camera-frame point to mesh-frame coordinates."""
        return cam_pt * self.cam_to_mesh_scale + self.cam_to_mesh_offset

    def update_mesh_from_tracking(
        self,
        mesh,            # HexMesh — import avoided for circular imports
        tracking_result: dict,
        scale_factor:    float = 1.0,
    ) -> None:
        """
        Update cell displacements and rotations from optical flow tracking.

        Parameters
        ----------
        mesh            : The HexMesh to update
        tracking_result : Dict from OpticalFlowTracker.update()
        scale_factor    : Multiplier to amplify small real-world displacements
                          for visualization on the digital twin
        """
        positions  = tracking_result['positions']
        disps      = tracking_result['displacements']
        active     = tracking_result['active_mask']

        if active.sum() == 0:
            return

        active_pos   = positions[active]
        active_disps = disps[active]

        # Convert camera coordinates to mesh coordinates
        mesh_pos = np.array([
            self.cam_to_mesh(p) for p in active_pos
        ])

        for cell in mesh.cells.values():
            cx, cy = cell.position
            cell_pt = np.array([cx, cy])

            # Find nearby tracked points
            dists = np.linalg.norm(mesh_pos - cell_pt, axis=1)
            nearby = dists < self.search_radius

            if nearby.sum() == 0:
                # No nearby data → leave cell at rest
                continue

            nearby_dists = dists[nearby] + 1e-6
            nearby_disps = active_disps[nearby]

            # IDW weights
            weights = 1.0 / (nearby_dists ** self.idw_power)
            weights /= weights.sum()

            # Weighted displacement
            interp_disp = (weights[:, None] * nearby_disps).sum(axis=0)
            cell.displacement = interp_disp * scale_factor

        # Estimate stiffness from measured deformation
        for cell in mesh.cells.values():
            cell.update_estimates()


# ── Camera capture ────────────────────────────────────────────────────────────

class CameraCapture:
    """
    Manages the video capture from a smartphone or webcam.

    Usage:
    ------
    cap = CameraCapture("http://192.168.1.100:4747/video")
    if cap.open():
        frame = cap.read_frame()
        ...
    cap.release()
    """

    def __init__(self, source) -> None:
        """
        Parameters
        ----------
        source : URL string, device path, or integer device index
        """
        self.source = source
        self._cap: Optional[cv2.VideoCapture] = None
        self.is_open: bool = False
        self.frame_width:  int = 0
        self.frame_height: int = 0
        self.fps:          float = 30.0

    def open(self) -> bool:
        """
        Open the video source.

        Returns
        -------
        True if successfully opened, False otherwise.
        """
        try:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                print(f"[CameraCapture] Failed to open: {self.source}")
                self.is_open = False
                return False

            self.frame_width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps          = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.is_open = True
            print(f"[CameraCapture] Opened: {self.source} "
                  f"({self.frame_width}×{self.frame_height} @ {self.fps:.1f}fps)")
            return True

        except Exception as e:
            print(f"[CameraCapture] Exception opening source: {e}")
            self.is_open = False
            return False

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read one frame. Returns None on failure.
        Resizes to 640×480 for consistent processing.
        """
        if not self.is_open or self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            return None

        # Resize for consistent processing speed
        frame = cv2.resize(frame, (640, 480))
        return frame

    def release(self) -> None:
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
        self.is_open = False

    def __del__(self) -> None:
        self.release()


# ── Setup instructions ────────────────────────────────────────────────────────

SETUP_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════════╗
║       Smartphone Camera Setup — Ubuntu Linux                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  OPTION 1: DroidCam (Recommended — WiFi or USB)                      ║
║  ─────────────────────────────────────────────                        ║
║  Android: Install "DroidCam" from Play Store                         ║
║  Ubuntu:  sudo apt install v4l2loopback-dkms                         ║
║           Download DroidCam Linux client from dev47apps.com          ║
║           Run: droidcam-cli wifi <phone_ip> 4747                     ║
║  Stream:  http://<phone_ip>:4747/video                               ║
║                                                                      ║
║  OPTION 2: IP Webcam (WiFi — no PC client needed)                    ║
║  ─────────────────────────────────────────────────                    ║
║  Android: Install "IP Webcam" from Play Store                        ║
║  Start server on phone, note the IP:port shown                       ║
║  Stream:  http://<phone_ip>:8080/video                               ║
║                                                                      ║
║  OPTION 3: Iriun Webcam (USB)                                        ║
║  ──────────────────────────────                                       ║
║  Android: Install "Iriun Webcam" from Play Store                     ║
║  Ubuntu:  Install Iriun Linux app from iriun.com                     ║
║  Connect phone via USB, enable USB debugging                         ║
║  Stream:  /dev/video0 (or /dev/video1)                               ║
║                                                                      ║
║  IMPORTANT: Phone and PC must be on the same WiFi network            ║
║  for WiFi options. Disable phone sleep/lock during streaming.        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
