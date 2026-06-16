import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Protocol, runtime_checkable
from abc import ABC, abstractmethod


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class Frame:
    """A single captured frame with metadata."""
    bgr:        np.ndarray       # Raw BGR image
    timestamp:  float            # Seconds since capture start
    frame_idx:  int
    width:      int
    height:     int


@dataclass
class TrackedPoint:
    """A feature point tracked across frames."""
    pt_id:        int
    ref_pos:      np.ndarray     # (x, y) in reference frame
    cur_pos:      np.ndarray     # (x, y) in current frame
    displacement: np.ndarray     # cur_pos - ref_pos
    is_active:    bool = True
    confidence:   float = 1.0    # tracking confidence (0–1)


@dataclass
class DisplacementField:
    """
    Sparse or dense displacement field from tracking.

    positions     : (N, 2) tracked point positions in camera frame
    displacements : (N, 2) displacement vectors
    rotations     : (N,) local rotation estimates (radians)
    active_mask   : (N,) boolean — which points are reliably tracked
    frame_idx     : source frame index
    """
    positions:      np.ndarray
    displacements:  np.ndarray
    rotations:      np.ndarray
    active_mask:    np.ndarray
    frame_idx:      int = 0


@dataclass
class ForceMeasurement:
    """
    Force measurement from a sensor or user input.

    In future hardware integration, this would come from a load cell
    or force-torque sensor synchronized with the camera.

    For now, force_vector is set by the user via the UI force slider.
    """
    force_vector: np.ndarray     # [Fx, Fy] in Newtons (or normalized units)
    timestamp:    float
    source:       str = "assumed"   # 'assumed' | 'load_cell' | 'fts'


# ── Abstract interfaces ───────────────────────────────────────────────────────

class CameraSourceBase(ABC):
    """Abstract base for camera sources (USB, WiFi, RTSP)."""

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def read_frame(self) -> Optional[Frame]: ...

    @abstractmethod
    def release(self) -> None: ...

    @property
    @abstractmethod
    def is_open(self) -> bool: ...


class FeatureDetectorBase(ABC):
    """Abstract base for feature detection strategies."""

    @abstractmethod
    def detect(self, gray: np.ndarray) -> np.ndarray:
        """Detect feature points. Returns (N, 2) array of positions."""
        ...


class DeformationTrackerBase(ABC):
    """Abstract base for deformation tracking algorithms."""

    @abstractmethod
    def set_reference(self, frame: Frame) -> int:
        """Set reference frame. Returns number of tracked points."""
        ...

    @abstractmethod
    def update(self, frame: Frame) -> DisplacementField:
        """Track deformation in new frame."""
        ...

    @property
    @abstractmethod
    def has_reference(self) -> bool: ...


class ForceSensorBase(ABC):
    """Abstract base for force measurement sources."""

    @abstractmethod
    def read(self) -> ForceMeasurement: ...


# ── Concrete implementations ──────────────────────────────────────────────────

class OpenCVCameraSource(CameraSourceBase):
    """
    OpenCV-based camera source supporting:
      - USB webcam (integer index)
      - WiFi stream (URL string)
      - RTSP stream (URL string)
      - V4L2 device path (string)

    Resizes all frames to (target_w, target_h) for consistent processing.
    """

    def __init__(
        self,
        source,               # int or str
        target_w:   int = 640,
        target_h:   int = 480,
        buffer_size: int = 1,   # Minimize buffer to reduce latency
    ) -> None:
        self.source    = source
        self.target_w  = target_w
        self.target_h  = target_h
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open  = False
        self._frame_idx = 0
        import time
        self._start_time = time.time()

    def open(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                return False
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._is_open = True
            print(f"[OpenCVCameraSource] Opened: {self.source} "
                  f"({int(self._cap.get(3))}×{int(self._cap.get(4))})")
            return True
        except Exception as e:
            print(f"[OpenCVCameraSource] Error: {e}")
            return False

    def read_frame(self) -> Optional[Frame]:
        if not self._is_open or self._cap is None:
            return None
        ret, bgr = self._cap.read()
        if not ret:
            return None
        import time
        bgr = cv2.resize(bgr, (self.target_w, self.target_h))
        self._frame_idx += 1
        return Frame(
            bgr=bgr,
            timestamp=time.time() - self._start_time,
            frame_idx=self._frame_idx,
            width=self.target_w,
            height=self.target_h,
        )

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open


class ShiTomasiFeatureDetector(FeatureDetectorBase):
    """
    Shi-Tomasi corner detector (goodFeaturesToTrack).

    Suitable for deformable objects with texture (gel pads, rubber, skin).
    For smoother surfaces, consider FAST or ORB.
    """

    def __init__(
        self,
        max_corners:   int   = 150,
        quality_level: float = 0.02,
        min_distance:  int   = 12,
        block_size:    int   = 7,
    ) -> None:
        self._params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )

    def detect(self, gray: np.ndarray) -> np.ndarray:
        corners = cv2.goodFeaturesToTrack(gray, **self._params)
        if corners is None:
            return np.zeros((0, 2))
        return corners.reshape(-1, 2)


class ArucoMarkerDetector(FeatureDetectorBase):
    """
    ArUco marker detector for precise, robust tracking.

    Requires physical ArUco markers printed and placed on the tissue surface.
    More robust than optical flow under large deformations and occlusion.

    Usage: print an ArUco dictionary (e.g., DICT_4X4_50) and adhere
    markers to a gel pad or rubber tissue phantom.
    """

    def __init__(self, dictionary_id: int = None) -> None:
        try:
            import cv2.aruco as aruco
            if dictionary_id is None:
                dictionary_id = aruco.DICT_4X4_50
            self._dictionary = aruco.getPredefinedDictionary(dictionary_id)
            self._detector   = aruco.ArucoDetector(
                self._dictionary,
                aruco.DetectorParameters(),
            )
            self._available  = True
        except (ImportError, AttributeError):
            print("[ArucoMarkerDetector] ArUco not available — install opencv-contrib-python")
            self._available = False

    def detect(self, gray: np.ndarray) -> np.ndarray:
        if not self._available:
            return np.zeros((0, 2))
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None:
            return np.zeros((0, 2))
        # Use marker center points as tracking locations
        centers = np.array([c[0].mean(axis=0) for c in corners])
        return centers


class LucasKanadeTracker(DeformationTrackerBase):
    """
    Lucas-Kanade pyramidal optical flow tracker.

    Tracks features from a reference frame to all subsequent frames.
    Computes displacement = current_position - reference_position.
    """

    LK_PARAMS = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def __init__(self, feature_detector: FeatureDetectorBase = None) -> None:
        self._detector    = feature_detector or ShiTomasiFeatureDetector()
        self._ref_gray:   Optional[np.ndarray] = None
        self._ref_pts:    Optional[np.ndarray] = None   # shape (N, 1, 2)
        self._tracked:    List[TrackedPoint]   = []
        self._has_ref     = False

    def set_reference(self, frame: Frame) -> int:
        gray      = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
        gray      = cv2.GaussianBlur(gray, (5, 5), 0)
        self._ref_gray = gray

        pts = self._detector.detect(gray)
        if len(pts) == 0:
            self._has_ref = False
            return 0

        self._ref_pts = pts.reshape(-1, 1, 2).astype(np.float32)
        self._tracked = [
            TrackedPoint(
                pt_id=i,
                ref_pos=pt.copy(),
                cur_pos=pt.copy(),
                displacement=np.zeros(2),
            )
            for i, pt in enumerate(pts)
        ]
        self._has_ref = True
        return len(pts)

    def update(self, frame: Frame) -> DisplacementField:
        if not self._has_ref or self._ref_pts is None:
            return self._empty_field()

        gray = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        cur_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._ref_gray, gray, self._ref_pts, None, **self.LK_PARAMS
        )

        if cur_pts is None:
            return self._empty_field()

        active   = status.ravel().astype(bool)
        positions    = []
        displacements = []
        rotations    = []

        for i, tp in enumerate(self._tracked):
            if i < len(active) and active[i]:
                cur_pos         = cur_pts[i][0]
                disp            = cur_pos - tp.ref_pos
                tp.cur_pos      = cur_pos
                tp.displacement = disp
                tp.is_active    = True
            else:
                tp.is_active = False

            positions.append(tp.cur_pos)
            displacements.append(tp.displacement)
            rotations.append(0.0)   # per-point rotation estimated in post-processing

        positions     = np.array(positions)
        displacements = np.array(displacements)
        rotations     = np.array(rotations)

        # Estimate local rotation at each point from nearest-neighbour curl
        _estimate_local_rotations(positions, displacements, rotations, active)

        return DisplacementField(
            positions     = positions,
            displacements = displacements,
            rotations     = rotations,
            active_mask   = active,
            frame_idx     = frame.frame_idx,
        )

    @property
    def has_reference(self) -> bool:
        return self._has_ref

    def _empty_field(self) -> DisplacementField:
        z = np.zeros((0, 2))
        return DisplacementField(
            positions=z, displacements=z, rotations=np.zeros(0),
            active_mask=np.zeros(0, dtype=bool),
        )


class AssumedForceSensor(ForceSensorBase):
    """
    Placeholder force sensor that returns a user-specified assumed force.

    In a real system, this would be replaced by a load cell or
    force-torque sensor (e.g., ATI Mini45, Nano43) synchronized
    with the camera frame timestamps.
    """

    def __init__(self, force_vector: np.ndarray = None) -> None:
        self.force_vector = force_vector if force_vector is not None else np.array([8.0, 0.0])
        import time
        self._t0 = time.time()

    def set_force(self, fx: float, fy: float) -> None:
        self.force_vector = np.array([fx, fy])

    def read(self) -> ForceMeasurement:
        import time
        return ForceMeasurement(
            force_vector=self.force_vector.copy(),
            timestamp=time.time() - self._t0,
            source="assumed",
        )


# ── Digital Twin Updater ──────────────────────────────────────────────────────

class TwinUpdater:
    """
    Maps a DisplacementField from camera tracking to a HexMesh.

    Uses Inverse Distance Weighting (IDW) interpolation:
        u_cell = Σ_i w_i · u_i / Σ_i w_i,   w_i = 1 / d_i^p

    Parameters
    ----------
    search_radius : Max distance from cell centroid to include a tracked point
    idw_power     : IDW power parameter (p=2 is standard)
    coordinate_mapping : dict with keys 'cam_bounds', 'mesh_bounds'
    """

    def __init__(
        self,
        search_radius:         float = 80.0,
        idw_power:             float = 2.0,
        displacement_scale:    float = 1.5,
    ) -> None:
        self.search_radius      = search_radius
        self.idw_power          = idw_power
        self.displacement_scale = displacement_scale

        # Coordinate transform
        self._scale:  float = 1.0
        self._offset: np.ndarray = np.zeros(2)

    def set_coordinate_mapping(
        self,
        cam_bounds:  Tuple,
        mesh_bounds: Tuple,
    ) -> None:
        """Set the camera-to-mesh coordinate transform."""
        cam_w  = cam_bounds[2]  - cam_bounds[0]
        cam_h  = cam_bounds[3]  - cam_bounds[1]
        mesh_w = mesh_bounds[2] - mesh_bounds[0]
        mesh_h = mesh_bounds[3] - mesh_bounds[1]
        self._scale  = min(mesh_w / cam_w, mesh_h / cam_h)
        self._offset = np.array([
            mesh_bounds[0] - cam_bounds[0] * self._scale,
            mesh_bounds[1] - cam_bounds[1] * self._scale,
        ])

    def cam_to_mesh(self, pt: np.ndarray) -> np.ndarray:
        return pt * self._scale + self._offset

    def update_mesh(self, mesh, field: DisplacementField) -> None:
        """
        Update all mesh cell displacements from the DisplacementField.

        Only active tracked points are used in the IDW interpolation.
        Cells with no nearby active points are left at their current
        displacement (not reset), allowing partial coverage.
        """
        if field.active_mask.sum() == 0:
            return

        active_pos  = field.positions[field.active_mask]
        active_disp = field.displacements[field.active_mask]
        active_rot  = field.rotations[field.active_mask]

        # Convert camera positions to mesh coordinates
        mesh_pos = np.array([self.cam_to_mesh(p) for p in active_pos])

        for cell in mesh.cells.values():
            cell_pos = cell.position
            dists    = np.linalg.norm(mesh_pos - cell_pos, axis=1)
            nearby   = dists < self.search_radius

            if nearby.sum() == 0:
                continue

            nd     = dists[nearby] + 1e-6
            nw     = 1.0 / nd ** self.idw_power
            nw    /= nw.sum()

            nd_disp = active_disp[nearby]
            nd_rot  = active_rot[nearby]

            cell.displacement = (nw[:, None] * nd_disp).sum(axis=0) * self.displacement_scale
            cell.rotation     = float((nw * nd_rot).sum())

        for cell in mesh.cells.values():
            cell.update_estimates()


# ── Vision Pipeline (Facade) ──────────────────────────────────────────────────

class VisionPipeline:
    """
    High-level facade that chains together:
        Camera → Preprocessor → Tracker → TwinUpdater

    Usage:
    ------
        pipeline = VisionPipeline(
            source=OpenCVCameraSource("http://192.168.1.100:8080/video"),
            tracker=LucasKanadeTracker(),
            updater=TwinUpdater(),
            force_sensor=AssumedForceSensor(),
        )
        pipeline.open()
        pipeline.set_reference()

        while True:
            result = pipeline.step(mesh)  # updates mesh in place
            annotated_frame = result['annotated']
            # display annotated_frame with Pygame
    """

    def __init__(
        self,
        source:       CameraSourceBase,
        tracker:      DeformationTrackerBase,
        updater:      TwinUpdater,
        force_sensor: ForceSensorBase = None,
    ) -> None:
        self.source       = source
        self.tracker      = tracker
        self.updater      = updater
        self.force_sensor = force_sensor or AssumedForceSensor()
        self._last_frame: Optional[Frame] = None

    def open(self) -> bool:
        return self.source.open()

    def set_reference(self) -> int:
        """Capture current frame as reference (rest) state."""
        frame = self.source.read_frame()
        if frame is None:
            return 0
        self._last_frame = frame
        n = self.tracker.set_reference(frame)
        print(f"[VisionPipeline] Reference set: {n} tracked points")
        return n

    def step(self, mesh) -> Dict:
        """
        Capture one frame, track, and update the digital twin mesh.

        Returns dict with:
          'annotated'   : BGR frame with tracking overlay
          'field'       : DisplacementField
          'force'       : ForceMeasurement
          'n_active'    : int number of active tracked points
        """
        frame = self.source.read_frame()
        if frame is None:
            return {"annotated": None, "field": None, "force": None, "n_active": 0}

        self._last_frame = frame
        field = self.tracker.update(frame)
        force = self.force_sensor.read()

        if field.active_mask.sum() > 0:
            self.updater.update_mesh(mesh, field)

        annotated = self._draw_overlay(frame.bgr, field)

        return {
            "annotated": annotated,
            "field":     field,
            "force":     force,
            "n_active":  int(field.active_mask.sum()),
        }

    def release(self) -> None:
        self.source.release()

    def _draw_overlay(self, bgr: np.ndarray, field: DisplacementField) -> np.ndarray:
        """Draw tracking points and displacement vectors on the frame."""
        out = bgr.copy()
        if len(field.positions) == 0:
            return out

        for i, (pos, disp, active) in enumerate(
            zip(field.positions, field.displacements, field.active_mask)
        ):
            px, py = int(pos[0]), int(pos[1])
            if active:
                end = (px + int(disp[0] * 2), py + int(disp[1] * 2))
                cv2.arrowedLine(out, (px, py), end, (0, 255, 0), 1, tipLength=0.3)
                cv2.circle(out, (px, py), 3, (0, 200, 255), -1)
            else:
                cv2.circle(out, (px, py), 2, (0, 0, 180), -1)

        # Status text
        n_active = int(field.active_mask.sum())
        cv2.putText(out, f"Tracked: {n_active}/{len(field.positions)}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
        return out


# ── Utility functions ─────────────────────────────────────────────────────────

def _estimate_local_rotations(
    positions:     np.ndarray,
    displacements: np.ndarray,
    rotations:     np.ndarray,
    active_mask:   np.ndarray,
    k_neighbors:   int = 6,
) -> None:
    """
    Estimate per-point rotation from the curl of the local displacement field.

    For each active point i, find k nearest neighbours and compute:
        θ_i ≈ 0.5 * (∂uy/∂x - ∂ux/∂y)

    Using finite difference from nearest neighbours.
    Modifies rotations array in-place.
    """
    if len(positions) < 3:
        return

    from scipy.spatial import KDTree
    active_pos  = positions[active_mask]
    active_disp = displacements[active_mask]
    active_idx  = np.where(active_mask)[0]

    if len(active_pos) < 3:
        return

    tree = KDTree(active_pos)
    _, nn_idx = tree.query(active_pos, k=min(k_neighbors + 1, len(active_pos)))

    for i, orig_i in enumerate(active_idx):
        neighbors = nn_idx[i][1:]   # exclude self
        if len(neighbors) < 2:
            continue

        # Finite-difference curl estimate
        r_vecs = active_pos[neighbors] - active_pos[i]
        du_vecs = active_disp[neighbors] - active_disp[i]

        # θ ≈ mean(r × Δu) / mean(|r|²)
        cross = r_vecs[:, 0] * du_vecs[:, 1] - r_vecs[:, 1] * du_vecs[:, 0]
        r_sq  = (r_vecs ** 2).sum(axis=1)
        valid = r_sq > 1e-6
        if valid.sum() > 0:
            rotations[orig_i] = float(cross[valid].mean() / r_sq[valid].mean())


# ── Factory helpers ───────────────────────────────────────────────────────────

def build_pipeline_from_config(config: Dict) -> VisionPipeline:
    """
    Build a VisionPipeline from a configuration dictionary.

    Example config:
        {
            "source": "ipwebcam",
            "ip": "192.168.1.100",
            "detector": "shi_tomasi",
            "tracker": "lucas_kanade",
            "search_radius": 80.0,
            "displacement_scale": 1.5,
            "assumed_force": [8.0, 0.0],
        }
    """
    from vision import build_stream_url
    source_type = config.get("source", "ipwebcam")
    ip          = config.get("ip", "192.168.1.100")
    url         = build_stream_url(source_type, ip)

    camera      = OpenCVCameraSource(url)

    detector_name = config.get("detector", "shi_tomasi")
    if detector_name == "aruco":
        detector = ArucoMarkerDetector()
    else:
        detector = ShiTomasiFeatureDetector()

    tracker  = LucasKanadeTracker(feature_detector=detector)
    updater  = TwinUpdater(
        search_radius=config.get("search_radius", 80.0),
        displacement_scale=config.get("displacement_scale", 1.5),
    )
    force_vec = np.array(config.get("assumed_force", [8.0, 0.0]))
    sensor   = AssumedForceSensor(force_vec)

    return VisionPipeline(camera, tracker, updater, sensor)
