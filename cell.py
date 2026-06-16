
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


TISSUE_TYPES = {
    "fat": {
        "stiffness":         0.8,
        "stiffness_range":   (0.5, 1.2),
        "kx_scale":          1.0,
        "ky_scale":          1.0,
        "anisotropy_ratio":  1.0,
        "fiber_angle_mean":  0.0,
        "fiber_angle_std":   90.0,
        "color_label":       "soft",
        "rgb":               (100, 149, 237),
        "noise_sigma":       0.08,
        "description":       "Adipose tissue (~1–5 kPa)",
    },
    "muscle": {
        "stiffness":         3.5,
        "stiffness_range":   (2.5, 4.5),
        "kx_scale":          1.0,
        "ky_scale":          0.60,
        "anisotropy_ratio":  0.60,
        "fiber_angle_mean":  45.0,
        "fiber_angle_std":   15.0,
        "color_label":       "medium",
        "rgb":               (50, 205, 50),
        "noise_sigma":       0.12,
        "description":       "Skeletal muscle (~10–100 kPa, passive)",
    },
    "tendon": {
        "stiffness":         9.0,
        "stiffness_range":   (7.0, 10.0),
        "kx_scale":          1.0,
        "ky_scale":          0.30,
        "anisotropy_ratio":  0.30,
        "fiber_angle_mean":  0.0,
        "fiber_angle_std":   5.0,
        "color_label":       "hard",
        "rgb":               (220, 50, 50),
        "noise_sigma":       0.06,
        "description":       "Tendon / collagen (~500–2000 kPa)",
    },
}

ROTATIONAL_COUPLING  = 0.18
MEASUREMENT_SNR      = 12.0
MIN_DISP_THRESHOLD   = 1e-6
GRADIENT_ALPHA       = 0.0
MIN_ROTATION_FOR_FUSION = 0.005


@dataclass
class Cell:
    cell_id:       int
    position:      np.ndarray
    tissue_type:   str   = "muscle"
    stiffness:     float = 3.5
    stiffness_kx:  float = 3.5
    stiffness_ky:  float = 3.5
    fiber_angle:   float = 0.0

    neighbors:     List[int] = field(default_factory=list)

    displacement:  np.ndarray = field(default_factory=lambda: np.zeros(2))
    physics_disp:  np.ndarray = field(default_factory=lambda: np.zeros(2))
    rotation:      float = 0.0
    force:         np.ndarray = field(default_factory=lambda: np.zeros(2))
    moment:        float = 0.0

    use_rotation:  bool  = False

    est_kx:        Optional[float] = None
    est_ky:        Optional[float] = None
    unc_kx:        Optional[float] = None
    unc_ky:        Optional[float] = None

    est_kt:        Optional[float] = None
    unc_kt:        Optional[float] = None

    est_kx_fused:  Optional[float] = None
    unc_kx_fused:  Optional[float] = None

    confidence:    float = 0.0

    highlighted:   bool = False

    def reset_dynamics(self) -> None:
        self.displacement[:] = 0.0
        self.physics_disp[:] = 0.0
        self.rotation        = 0.0
        self.force[:]        = 0.0
        self.moment          = 0.0
        self.use_rotation    = False
        self.est_kx          = None
        self.est_ky          = None
        self.unc_kx          = None
        self.unc_ky          = None
        self.est_kt          = None
        self.unc_kt          = None
        self.est_kx_fused    = None
        self.unc_kx_fused    = None
        self.confidence      = 0.0

    def update_estimates(self, min_disp: float = MIN_DISP_THRESHOLD) -> None:
        dx, dy      = self.physics_disp
        noise_sigma = TISSUE_TYPES[self.tissue_type]["noise_sigma"]

        if abs(dx) > min_disp:
            self.est_kx = abs(self.force[0] / dx)
            sig_F = noise_sigma * abs(self.force[0]) if abs(self.force[0]) > 1e-9 else noise_sigma
            sig_d = abs(dx) / MEASUREMENT_SNR
            rel   = np.sqrt((sig_F / (abs(self.force[0]) + 1e-12)) ** 2 +
                            (sig_d / abs(dx)) ** 2)
            self.unc_kx = self.est_kx * rel
        else:
            self.est_kx = None
            self.unc_kx = None

        if abs(dy) > min_disp:
            self.est_ky = abs(self.force[1] / dy)
            sig_F = noise_sigma * abs(self.force[1]) if abs(self.force[1]) > 1e-9 else noise_sigma
            sig_d = abs(dy) / MEASUREMENT_SNR
            rel   = np.sqrt((sig_F / (abs(self.force[1]) + 1e-12)) ** 2 +
                            (sig_d / abs(dy)) ** 2)
            self.unc_ky = self.est_ky * rel
        else:
            self.est_ky = None
            self.unc_ky = None

        min_rot = 1e-5
        if abs(self.rotation) > min_rot and abs(self.moment) > 1e-8:
            self.est_kt = abs(self.moment / self.rotation)
            sig_M = noise_sigma * abs(self.moment)
            sig_r = abs(self.rotation) / MEASUREMENT_SNR
            rel   = np.sqrt((sig_M / (abs(self.moment) + 1e-12)) ** 2 +
                            (sig_r / abs(self.rotation)) ** 2)
            self.unc_kt = self.est_kt * rel
        else:
            self.est_kt = None
            self.unc_kt = None

        if (self.use_rotation
                and self.est_kx  is not None
                and self.est_kt  is not None
                and self.unc_kx  is not None
                and self.unc_kt  is not None
                and abs(self.rotation) > MIN_ROTATION_FOR_FUSION):

            var_t = self.unc_kx ** 2
            var_r = self.unc_kt ** 2

            w_t = 1.0 / (var_t + 1e-12)
            w_r = 1.0 / (var_r + 1e-12)
            kx_fused  = (w_t * self.est_kx + w_r * self.est_kt) / (w_t + w_r)
            unc_fused = 1.0 / np.sqrt(w_t + w_r)

            if abs(self.rotation) > MIN_ROTATION_FOR_FUSION:
                gradient_correction = GRADIENT_ALPHA * self.rotation
                gradient_correction = np.clip(gradient_correction, -0.20, 0.20)
                kx_fused = kx_fused * (1.0 + gradient_correction)

            self.est_kx_fused = float(kx_fused)
            self.unc_kx_fused = float(unc_fused)

        else:
            self.est_kx_fused = self.est_kx
            self.unc_kx_fused = self.unc_kx

        if self.use_rotation and self.est_kt is not None:
            ests = [e for e in [self.est_kx_fused, self.est_ky] if e is not None]
            uncs = [u for u in [self.unc_kx_fused, self.unc_ky] if u is not None]
        else:
            ests = [e for e in [self.est_kx, self.est_ky] if e is not None]
            uncs = [u for u in [self.unc_kx, self.unc_ky] if u is not None]

        if ests and uncs:
            rel_unc = np.mean(uncs) / (np.mean(ests) + 1e-9)
            self.confidence = float(np.clip(1.0 - rel_unc, 0.0, 1.0))
        else:
            self.confidence = 0.0

    @property
    def deformed_position(self) -> np.ndarray:
        return self.position + self.displacement

    @property
    def stiffness_color(self) -> Tuple[int, int, int]:
        return _stiffness_to_rgb(self.stiffness)

    @property
    def estimated_stiffness_color(self) -> Tuple[int, int, int]:
        k = self.est_kx_fused if self.est_kx_fused is not None else self.est_kx
        if k is None:
            return (80, 80, 80)
        return _stiffness_to_rgb(float(k))

    @property
    def confidence_color(self) -> Tuple[int, int, int]:
        c = self.confidence
        if c < 0.5:
            t = c / 0.5
            return (220, int(220 * t), 0)
        t = (c - 0.5) / 0.5
        return (int(220 * (1 - t)), 220, 0)

    @property
    def anisotropy_color(self) -> Tuple[int, int, int]:
        ratio = self.stiffness_kx / (self.stiffness_ky + 1e-9)
        t = np.clip((ratio - 1.0) / 3.0, 0.0, 1.0)
        return (int(80 + 160 * t), int(120 - 80 * t), int(200 - 150 * t))

    @property
    def confidence_pct(self) -> float:
        return self.confidence * 100.0

    def __repr__(self) -> str:
        return (f"Cell(id={self.cell_id}, type={self.tissue_type}, "
                f"k={self.stiffness:.3f}, kx_fused={self.est_kx_fused}, "
                f"conf={self.confidence_pct:.0f}%)")


def _stiffness_to_rgb(k: float) -> Tuple[int, int, int]:
    k_min, k_max = 0.5, 10.0
    t = np.clip((k - k_min) / (k_max - k_min), 0.0, 1.0)
    if t < 0.5:
        s = t / 0.5
        return (int(100 * (1 - s)), int(50 + 155 * s), int(237 * (1 - s)))
    s = (t - 0.5) / 0.5
    return (int(50 + 170 * s), int(205 * (1 - s)), int(50 * (1 - s)))
