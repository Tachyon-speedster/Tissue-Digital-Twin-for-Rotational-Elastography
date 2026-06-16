import pygame
import numpy as np
import math
from typing import Optional, Tuple, Dict
from mesh import HexMesh
from simulation import Simulation
from cell import Cell, TISSUE_TYPES

C_BG          = (10,  10,  22)
C_PANEL       = (18,  18,  36)
C_BORDER      = (40,  40,  80)
C_TEXT        = (200, 210, 230)
C_TEXT_DIM    = (100, 110, 140)
C_ACCENT      = (80,  160, 240)
C_ACCENT2     = (120, 240, 160)
C_ACCENT3     = (240, 120, 80)
C_WARN        = (240, 80,  80)
C_HEX_EDGE    = (50,  55,  90)
C_HEX_EDGE_HI = (180, 200, 255)
C_SELECTED    = (255, 220, 60)
C_FORCE_ARROW = (255, 100, 60)
C_DISP_ARROW  = (60,  200, 255)
C_ROT_RING    = (255, 180, 50)
C_FIBER       = (200, 200, 255)
C_CONF_HI     = (80,  220, 80)
C_CONF_LO     = (220, 80,  80)
C_CONF_MID    = (220, 200, 50)

WINDOW_W   = 1300
WINDOW_H   = 800
VIEWPORT_W = 920
PANEL_W    = WINDOW_W - VIEWPORT_W
STATUS_H   = 36
VIEWPORT_H = WINDOW_H - STATUS_H

FORCE_MAGNITUDE_MIN = 1.0
FORCE_MAGNITUDE_MAX = 20.0

HEATMAP_MODES  = ["off", "stiffness_true", "stiffness_est", "error", "confidence", "anisotropy"]
HEATMAP_LABELS = {
    "off":            "Tissue Type",
    "stiffness_true": "True Stiffness",
    "stiffness_est":  "Est. Stiffness",
    "error":          "Est. Error",
    "confidence":     "Confidence",
    "anisotropy":     "Anisotropy",
}


class UIRenderer:
    def __init__(self, sim: Simulation, logger=None) -> None:
        pygame.init()
        pygame.display.set_caption("Tissue Digital Twin v2")
        self.screen   = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock    = pygame.time.Clock()

        self.viewport  = self.screen.subsurface((0, 0, VIEWPORT_W, VIEWPORT_H))
        self.panel     = self.screen.subsurface((VIEWPORT_W, 0, PANEL_W, VIEWPORT_H))
        self.statusbar = self.screen.subsurface((0, VIEWPORT_H, WINDOW_W, STATUS_H))

        self.font_lg  = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_md  = pygame.font.SysFont("monospace", 11)
        self.font_sm  = pygame.font.SysFont("monospace", 9)
        self.font_xl  = pygame.font.SysFont("monospace", 15, bold=True)

        self.sim    = sim
        self.mesh   = sim.mesh
        self.logger = logger

        self.heatmap_mode      = "off"
        self.show_displacement = True
        self.show_rotation     = True
        self.show_fiber_glyphs = False
        self.show_confidence   = False
        self.force_magnitude   = 8.0
        self.force_presets     = [
            np.array([ 1.0,  0.0]),
            np.array([ 0.707, 0.707]),
            np.array([ 0.0,  1.0]),
            np.array([-0.707, 0.707]),
            np.array([-1.0,  0.0]),
            np.array([ 0.0, -1.0]),
        ]
        self._force_preset_idx = 0
        self.force_direction   = self.force_presets[0].copy()

        self.vision_mode    = False
        self.vision_frame   = None
        self.vision_surface = None

        self.slider_dragging  = False
        self._slider_pending  = False
        self.selected_cell:   Optional[Cell] = None

        self._notification = ""
        self._notif_timer  = 0

        self._slider_rect      = pygame.Rect(0, 0, 1, 1)
        self._btn_a            = pygame.Rect(0, 0, 1, 1)
        self._btn_b            = pygame.Rect(0, 0, 1, 1)
        self._btn_heatmap      = pygame.Rect(0, 0, 1, 1)
        self._btn_exp          = pygame.Rect(0, 0, 1, 1)
        self._btn_report       = pygame.Rect(0, 0, 1, 1)
        self._btn_aniso        = pygame.Rect(0, 0, 1, 1)
        self._btn_export       = pygame.Rect(0, 0, 1, 1)
        self._btn_export_json  = pygame.Rect(0, 0, 1, 1)

    def draw(self) -> None:
        self.screen.fill(C_BG)
        self._draw_viewport()
        self._draw_panel()
        self._draw_statusbar()
        pygame.display.flip()

    def _draw_viewport(self) -> None:
        self.viewport.fill(C_BG)
        if self.vision_mode and self.vision_surface is not None:
            self._draw_vision_split()
        else:
            self._draw_hex_mesh()
            self._draw_force_arrow()
            self._draw_overlays()
            self._draw_mode_label()
        if self._notif_timer > 0:
            self._draw_notification()
            self._notif_timer -= 1

    def _draw_hex_mesh(self) -> None:
        for cell in self.mesh.all_cells_list():
            self._draw_cell(cell)
        if self.show_fiber_glyphs:
            self._draw_fiber_glyphs()

    def _draw_cell(self, cell: Cell, use_deformed: bool = True) -> None:
        color     = self._cell_color(cell)
        verts     = self.mesh.rotated_hex_vertices(cell, scale=0.91, use_deformed=use_deformed)
        verts_int = [(int(v[0]), int(v[1])) for v in verts]

        pygame.draw.polygon(self.viewport, color, verts_int)

        if self.show_confidence and cell.confidence > 0:
            cf     = self._confidence_color(cell.confidence)
            inner  = self.mesh.rotated_hex_vertices(cell, scale=0.55, use_deformed=use_deformed)
            pygame.draw.polygon(self.viewport, cf,
                                [(int(v[0]), int(v[1])) for v in inner])

        edge_color = (C_SELECTED if cell == self.selected_cell
                      else C_HEX_EDGE_HI if cell.highlighted
                      else C_HEX_EDGE)
        edge_w = 2 if cell == self.selected_cell else 1
        pygame.draw.polygon(self.viewport, edge_color, verts_int, edge_w)

        cx, cy = cell.deformed_position if use_deformed else cell.position
        surf   = self.font_sm.render(cell.tissue_type[0].upper(), True, (180, 180, 180))
        self.viewport.blit(surf, (int(cx) - 4, int(cy) - 5))

    def _cell_color(self, cell: Cell) -> Tuple[int, int, int]:
        m = self.heatmap_mode
        if m == "stiffness_true":
            return cell.stiffness_color
        elif m == "stiffness_est":
            return cell.estimated_stiffness_color
        elif m == "error":
            ests = [e for e in [cell.est_kx, cell.est_ky] if e is not None]
            if ests:
                t = np.clip(abs(float(np.mean(ests)) - cell.stiffness) / 3.0, 0.0, 1.0)
                return (int(255 * t), int(60 * (1 - t)), 30)
            return (50, 50, 80)
        elif m == "confidence":
            return cell.confidence_color
        elif m == "anisotropy":
            return cell.anisotropy_color
        else:
            base     = TISSUE_TYPES[cell.tissue_type]["rgb"]
            bright   = min(1.0, 0.55 + 0.45 * np.linalg.norm(cell.displacement) / 25.0)
            return tuple(int(c * bright) for c in base)

    def _confidence_color(self, conf: float) -> Tuple[int, int, int]:
        if conf < 0.5:
            t = conf / 0.5
            return (C_CONF_LO[0], int(C_CONF_LO[1] + (C_CONF_MID[1] - C_CONF_LO[1]) * t), 0)
        t = (conf - 0.5) / 0.5
        return (int(C_CONF_MID[0] * (1 - t)), C_CONF_HI[1], 0)

    def _draw_fiber_glyphs(self) -> None:
        for cell in self.mesh.all_cells_list():
            pos   = cell.deformed_position
            cx, cy = int(pos[0]), int(pos[1])
            angle  = math.radians(cell.fiber_angle)
            ratio  = min(cell.stiffness_kx / (cell.stiffness_ky + 1e-9), 3.0)
            half   = int(self.mesh.hex_radius * 0.35 * ratio)
            dx, dy = int(half * math.cos(angle)), int(half * math.sin(angle))
            pygame.draw.line(self.viewport, C_FIBER,
                             (cx - dx, cy - dy), (cx + dx, cy + dy), 1)

    def _draw_force_arrow(self) -> None:
        if self.selected_cell is None:
            return
        pos    = self.selected_cell.deformed_position
        cx, cy = int(pos[0]), int(pos[1])
        fx, fy = self.sim.force * 4.0
        ex, ey = cx + int(fx), cy + int(fy)
        if abs(fx) + abs(fy) < 1:
            return
        pygame.draw.line(self.viewport, C_FORCE_ARROW, (cx, cy), (ex, ey), 3)
        angle = math.atan2(ey - cy, ex - cx)
        for da in [0.4, -0.4]:
            hx = ex - int(12 * math.cos(angle + da))
            hy = ey - int(12 * math.sin(angle + da))
            pygame.draw.line(self.viewport, C_FORCE_ARROW, (ex, ey), (hx, hy), 2)
        surf = self.font_sm.render(f"F={np.linalg.norm(self.sim.force):.1f}", True, C_FORCE_ARROW)
        self.viewport.blit(surf, (ex + 4, ey - 8))

    def _draw_overlays(self) -> None:
        for cell in self.mesh.all_cells_list():
            mag = np.linalg.norm(cell.displacement)
            if mag < 0.5:
                continue
            if self.show_displacement:
                self._draw_disp_arrow(cell)
            if self.show_rotation and self.sim.mode == "B" and abs(cell.rotation) > 0.01:
                self._draw_rot_indicator(cell)

    def _draw_disp_arrow(self, cell: Cell) -> None:
        pos  = cell.position
        disp = cell.displacement
        mag  = np.linalg.norm(disp)
        if mag > 30:
            disp = disp / mag * 30
        start = (int(pos[0]), int(pos[1]))
        end   = (int(pos[0] + disp[0]), int(pos[1] + disp[1]))
        pygame.draw.line(self.viewport, C_DISP_ARROW, start, end, 1)
        pygame.draw.circle(self.viewport, C_DISP_ARROW, end, 2)

    def _draw_rot_indicator(self, cell: Cell) -> None:
        pos   = cell.deformed_position
        cx, cy = int(pos[0]), int(pos[1])
        r      = int(self.mesh.hex_radius * 0.50)
        theta  = cell.rotation
        n_seg  = 10
        span   = min(abs(theta) * 4, math.pi)
        start  = -math.pi / 2
        segs   = [
            (cx + int(r * math.cos(start + span * i / n_seg * (1 if theta > 0 else -1))),
             cy + int(r * math.sin(start + span * i / n_seg * (1 if theta > 0 else -1))))
            for i in range(n_seg + 1)
        ]
        if len(segs) >= 2:
            pygame.draw.lines(self.viewport, C_ROT_RING, False, segs, 1)
            pygame.draw.circle(self.viewport, C_ROT_RING, segs[-1], 2)

    def _draw_mode_label(self) -> None:
        mode  = self.sim.mode
        color = C_ACCENT if mode == "A" else C_ACCENT2
        self.viewport.blit(
            self.font_lg.render("MODE A — Translational Only" if mode == "A"
                                else "MODE B — Translation + Rotation", True, color), (10, 8))
        self.viewport.blit(
            self.font_sm.render(f"MAP: {HEATMAP_LABELS[self.heatmap_mode]}", True, C_ACCENT3),
            (10, 26))
        if self.slider_dragging:
            surf = self.font_sm.render(
                f"Force: {self.force_magnitude:.1f} N  (release to apply)",
                True, C_ACCENT3)
            self.viewport.blit(surf, (10, 44))

    def _draw_vision_split(self) -> None:
        half_w = VIEWPORT_W // 2
        if self.vision_surface is not None:
            scaled = pygame.transform.scale(self.vision_surface, (half_w, VIEWPORT_H))
            self.viewport.blit(scaled, (0, 0))
        pygame.draw.line(self.viewport, C_BORDER, (half_w, 0), (half_w, VIEWPORT_H), 2)
        twin  = pygame.Surface((half_w, VIEWPORT_H))
        twin.fill(C_BG)
        for cell in self.mesh.all_cells_list():
            base  = TISSUE_TYPES[cell.tissue_type]["rgb"]
            mag   = np.linalg.norm(cell.displacement)
            bright = min(1.0, 0.55 + 0.45 * mag / 25.0)
            color = tuple(int(c * bright) for c in base)
            verts = self.mesh.rotated_hex_vertices(cell, scale=0.91, use_deformed=True)
            vi    = [(int(v[0]), int(v[1])) for v in verts]
            pygame.draw.polygon(twin, color, vi)
            pygame.draw.polygon(twin, C_HEX_EDGE, vi, 1)
        self.viewport.blit(twin, (half_w, 0))
        self.viewport.blit(self.font_lg.render("REAL OBJECT",  True, C_ACCENT),  (8, 8))
        self.viewport.blit(self.font_lg.render("DIGITAL TWIN", True, C_ACCENT2), (half_w + 8, 8))

    def _draw_notification(self) -> None:
        surf = self.font_md.render(self._notification, True, C_ACCENT3)
        r    = surf.get_rect(center=(VIEWPORT_W // 2, VIEWPORT_H - 40))
        bg   = pygame.Rect(r.x - 8, r.y - 4, r.w + 16, r.h + 8)
        pygame.draw.rect(self.viewport, C_PANEL, bg, border_radius=4)
        pygame.draw.rect(self.viewport, C_BORDER, bg, 1, border_radius=4)
        self.viewport.blit(surf, r)

    def _draw_panel(self) -> None:
        self.panel.fill(C_PANEL)
        pygame.draw.line(self.panel, C_BORDER, (0, 0), (0, VIEWPORT_H), 2)
        y = 8

        y = self._ps(y, "TISSUE DIGITAL TWIN", header=True)
        y = self._pd(y)

        y = self._ps(y, "APPLIED FORCE")
        y = self._pkv(y, "Fx",  f"{self.sim.force[0]:+.2f}")
        y = self._pkv(y, "Fy",  f"{self.sim.force[1]:+.2f}")
        y = self._pkv(y, "|F|", f"{np.linalg.norm(self.sim.force):.2f}")
        y = self._pd(y)

        cell = self.selected_cell
        if cell is not None:
            y = self._ps(y, f"CELL #{cell.cell_id} ({cell.tissue_type.upper()})")
            y = self._pkv(y, "k_true",  f"{cell.stiffness:.3f}",   val_color=C_ACCENT2)
            y = self._pkv(y, "kx_true", f"{cell.stiffness_kx:.3f}")
            y = self._pkv(y, "ky_true", f"{cell.stiffness_ky:.3f}")
            y = self._pkv(y, "fiber",   f"{cell.fiber_angle:.1f}°")
            y = self._pd(y)

            y = self._ps(y, "MEASURED")
            y = self._pkv(y, "Δx",   f"{cell.displacement[0]:+.2f} px")
            y = self._pkv(y, "Δy",   f"{cell.displacement[1]:+.2f} px")
            y = self._pkv(y, "|Δu|", f"{np.linalg.norm(cell.displacement):.2f} px")
            theta_str = (f"{math.degrees(cell.rotation):.2f}°"
                         if self.sim.mode == "B" else "--")
            y = self._pkv(y, "θ", theta_str)
            y = self._pd(y)

            mode_label = "ESTIMATED k — MODE B FUSED" if self.sim.mode == "B" else "ESTIMATED k — MODE A"
            y = self._ps(y, mode_label)
            if self.sim.mode == "B" and cell.est_kx_fused is not None:
                y = self._draw_est_row(y, "kx⊕kθ", cell.est_kx_fused,
                                       cell.unc_kx_fused, cell.stiffness_kx)
                y = self._draw_est_row(y, "kx_raw", cell.est_kx, cell.unc_kx, cell.stiffness_kx)
            else:
                y = self._draw_est_row(y, "kx", cell.est_kx, cell.unc_kx, cell.stiffness_kx)
            y = self._draw_est_row(y, "ky", cell.est_ky, cell.unc_ky, cell.stiffness_ky)
            y = self._draw_est_row(y, "kθ", cell.est_kt, cell.unc_kt, cell.stiffness)
            y = self._pd(y)
            y = self._draw_conf_bar(y, cell.confidence)
            y = self._pd(y)
        else:
            y = self._ps(y, "→ Click a cell to inspect")
            y = self._pd(y)

        y = self._ps(y, "FORCE MAGNITUDE")
        y = self._draw_slider(y)

        y = self._pd(y)
        y = self._ps(y, "MODE")
        y = self._draw_mode_buttons(y)

        y = self._pd(y)
        y = self._draw_heatmap_btn(y)

        y = self._pd(y)
        y = self._ps(y, "DISPLAY")
        y = self._draw_toggle(y, "show_displacement",  "Displacement arrows")
        y = self._draw_toggle(y, "show_rotation",      "Rotation arcs")
        y = self._draw_toggle(y, "show_fiber_glyphs",  "Fiber glyphs (G)")
        y = self._draw_toggle(y, "show_confidence",    "Confidence inner (Q)")
        y += 2

        y = self._pd(y)
        y = self._draw_action_buttons(y)

        if self.logger is not None:
            y = self._pd(y)
            y = self._ps(y, "DATA LOGGER")
            y = self._pkv(y, "Events", str(len(self.logger.records)), val_color=C_ACCENT3)
            y = self._draw_export_buttons(y)

        y = self._pd(y)
        y = self._ps(y, "KEYS")
        for key, desc in [
            ("Click", "Select + apply force"),
            ("F",     "Cycle force dir"),
            ("A/B",   "Mode A / B"),
            ("M",     "Cycle heatmap"),
            ("H",     "Analysis plots"),
            ("X",     "Run experiment"),
            ("P",     "PDF report"),
            ("N",     "Anisotropy on/off"),
            ("G/Q",   "Glyphs / Conf overlay"),
            ("C",     "Clear"),
            ("Esc",   "Quit"),
        ]:
            self.panel.blit(
                self.font_sm.render(f"  {key:<6}{desc}", True, C_TEXT_DIM), (8, y))
            y += 13

    def _draw_est_row(self, y, label, est, unc, true_k=None) -> int:
        if est is not None and unc is not None:
            val_str = f"{est:.3f}±{unc:.3f}"
            if true_k is not None:
                frac = abs(est - true_k) / (true_k + 1e-9)
                vc   = C_ACCENT2 if frac < 0.2 else C_ACCENT3 if frac < 0.5 else C_WARN
            else:
                vc = C_TEXT
        else:
            val_str, vc = "n/a", C_TEXT_DIM
        lbl = self.font_sm.render(f"  {label:<5}", True, C_TEXT_DIM)
        val = self.font_sm.render(val_str, True, vc)
        self.panel.blit(lbl, (8, y))
        self.panel.blit(val, (8 + lbl.get_width(), y))
        return y + 13

    def _draw_conf_bar(self, y, confidence) -> int:
        bx, bw, bh = 10, PANEL_W - 20, 9
        pygame.draw.rect(self.panel, C_BORDER, (bx, y + 13, bw, bh), border_radius=3)
        fw = int(bw * confidence)
        if fw > 0:
            r = int(220 * (1 - confidence) + 50 * confidence)
            g = int(220 * confidence)
            pygame.draw.rect(self.panel, (r, g, 30), (bx, y + 13, fw, bh), border_radius=3)
        self.panel.blit(self.font_sm.render(f"Confidence: {confidence*100:.0f}%",
                                             True, C_TEXT), (8, y + 1))
        return y + 28

    def _draw_heatmap_btn(self, y) -> int:
        lbl  = f"MAP: {HEATMAP_LABELS[self.heatmap_mode]}  [M]"
        bw, bh = PANEL_W - 20, 20
        bx, by = 10, y + 2
        pygame.draw.rect(self.panel, C_ACCENT3, (bx, by, bw, bh), border_radius=3)
        s = self.font_sm.render(lbl, True, C_BG)
        self.panel.blit(s, (bx + bw // 2 - s.get_width() // 2, by + 3))
        self._btn_heatmap = pygame.Rect(bx, by, bw, bh)
        return y + 26

    def _draw_action_buttons(self, y) -> int:
        bw = (PANEL_W - 24) // 2
        bh = 20
        for i, (label, color, attr) in enumerate([
            ("Run Experiment", (60, 120, 200), "_btn_exp"),
            ("PDF Report",     (80, 160, 80),  "_btn_report"),
        ]):
            bx = 8 + i * (bw + 8)
            by = y + 2
            pygame.draw.rect(self.panel, color, (bx, by, bw, bh), border_radius=3)
            s = self.font_sm.render(label, True, (230, 230, 230))
            self.panel.blit(s, (bx + bw // 2 - s.get_width() // 2, by + 3))
            setattr(self, attr, pygame.Rect(bx, by, bw, bh))
        bx3, by3 = 8, y + 28
        ac = (140, 60, 200) if self.sim.use_anisotropy else (60, 60, 90)
        pygame.draw.rect(self.panel, ac, (bx3, by3, bw * 2 + 8, bh), border_radius=3)
        lbl = "Anisotropy: ON  [N]" if self.sim.use_anisotropy else "Anisotropy: OFF [N]"
        s   = self.font_sm.render(lbl, True, (230, 230, 230))
        self.panel.blit(s, (bx3 + (bw * 2 + 8) // 2 - s.get_width() // 2, by3 + 3))
        self._btn_aniso = pygame.Rect(bx3, by3, bw * 2 + 8, bh)
        return y + 54

    def _draw_export_buttons(self, y) -> int:
        bw = (PANEL_W - 24) // 2
        bh = 18
        for i, (label, color, attr) in enumerate([
            ("Export CSV",  (70, 110, 70),  "_btn_export"),
            ("Export JSON", (70, 70, 110),  "_btn_export_json"),
        ]):
            bx = 8 + i * (bw + 8)
            by = y + 2
            pygame.draw.rect(self.panel, color, (bx, by, bw, bh), border_radius=3)
            s = self.font_sm.render(label, True, (220, 220, 220))
            self.panel.blit(s, (bx + bw // 2 - s.get_width() // 2, by + 3))
            setattr(self, attr, pygame.Rect(bx, by, bw, bh))
        return y + 24

    def _draw_toggle(self, y, attr, label) -> int:
        val     = getattr(self, attr)
        color   = C_ACCENT2 if val else C_TEXT_DIM
        box_col = C_ACCENT2 if val else C_BORDER
        pygame.draw.rect(self.panel, box_col, (10, y + 1, 10, 10), border_radius=2)
        if val:
            pygame.draw.line(self.panel, C_BG, (11, y + 6), (13, y + 9), 2)
            pygame.draw.line(self.panel, C_BG, (13, y + 9), (18, y + 3), 2)
        self.panel.blit(self.font_sm.render(label, True, color), (26, y))
        return y + 15

    def _ps(self, y, text, header=False) -> int:
        color = C_ACCENT if header else C_ACCENT2
        font  = self.font_xl if header else self.font_lg
        surf  = font.render(text, True, color)
        self.panel.blit(surf, (8, y))
        return y + surf.get_height() + 2

    def _pkv(self, y, key, value, val_color=None) -> int:
        vc  = val_color or C_TEXT
        ks  = self.font_md.render(f"  {key:<8}", True, C_TEXT_DIM)
        vs  = self.font_md.render(value, True, vc)
        self.panel.blit(ks, (8, y))
        self.panel.blit(vs, (8 + ks.get_width(), y))
        return y + 14

    def _pd(self, y) -> int:
        pygame.draw.line(self.panel, C_BORDER, (8, y + 3), (PANEL_W - 8, y + 3), 1)
        return y + 8

    def _draw_slider(self, y) -> int:
        sx, sy = 12, y + 14
        sw, sh = PANEL_W - 24, 8
        pygame.draw.rect(self.panel, C_BORDER, (sx, sy, sw, sh), border_radius=4)
        t  = (self.force_magnitude - FORCE_MAGNITUDE_MIN) / (FORCE_MAGNITUDE_MAX - FORCE_MAGNITUDE_MIN)
        pygame.draw.rect(self.panel, C_ACCENT, (sx, sy, int(sw * t), sh), border_radius=4)
        pygame.draw.circle(self.panel, C_ACCENT, (sx + int(sw * t), sy + sh // 2), 8)
        val_surf = self.font_sm.render(f"{self.force_magnitude:.1f} N", True, C_TEXT)
        self.panel.blit(val_surf, (sx + sw // 2 - 18, sy + sh + 4))
        if self.slider_dragging:
            hint = self.font_sm.render("release to apply", True, C_ACCENT3)
            self.panel.blit(hint, (sx + sw // 2 - hint.get_width() // 2, sy + sh + 18))
        self._slider_rect = pygame.Rect(sx, sy - 8, sw, sh + 16)
        return y + 46

    def _draw_mode_buttons(self, y) -> int:
        bw = (PANEL_W - 28) // 2
        bh = 22
        by = y + 3
        for mode, bx, label in [("A", 8, "MODE A"), ("B", 8 + bw + 8, "MODE B")]:
            active = self.sim.mode == mode
            pygame.draw.rect(self.panel, C_ACCENT if active else C_BORDER,
                             (bx, by, bw, bh), border_radius=4)
            s = self.font_md.render(label, True, C_BG if active else C_TEXT_DIM)
            self.panel.blit(s, (bx + bw // 2 - s.get_width() // 2, by + 3))
        self._btn_a = pygame.Rect(8, by, bw, bh)
        self._btn_b = pygame.Rect(8 + bw + 8, by, bw, bh)
        return y + bh + 8

    def _draw_statusbar(self) -> None:
        self.statusbar.fill((12, 12, 28))
        pygame.draw.line(self.statusbar, C_BORDER, (0, 0), (WINDOW_W, 0), 1)

        mode_col = C_ACCENT3 if self.vision_mode else C_ACCENT2
        self.statusbar.blit(
            self.font_md.render(f"● {'VISION' if self.vision_mode else 'SIM'}", True, mode_col),
            (10, 10))

        fps    = self.clock.get_fps()
        n_est  = sum(1 for c in self.mesh.cells.values() if c.est_kx is not None)
        confs  = [c.confidence_pct for c in self.mesh.cells.values() if c.est_kx is not None]
        avg_c  = f"{np.mean(confs):.0f}%" if confs else "--"
        center = f"Cells:{len(self.mesh)}  Est:{n_est}  Conf:{avg_c}  FPS:{fps:.0f}"
        s2 = self.font_md.render(center, True, C_TEXT_DIM)
        self.statusbar.blit(s2, (WINDOW_W // 2 - s2.get_width() // 2, 10))

        disc = self.font_sm.render("Research prototype — not for clinical use", True, C_TEXT_DIM)
        self.statusbar.blit(disc, (WINDOW_W - disc.get_width() - 10, 12))

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.QUIT:
            return False

        if event.type == pygame.KEYDOWN:
            return self._handle_keydown(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mousedown(event)

        if event.type == pygame.MOUSEMOTION:
            if self.slider_dragging and event.pos[0] >= VIEWPORT_W:
                self._update_slider_visual(event.pos[0] - VIEWPORT_W)

        if event.type == pygame.MOUSEBUTTONUP:
            if self.slider_dragging and self._slider_pending:
                if self.selected_cell is not None:
                    self.sim.apply_force(self.selected_cell.cell_id,
                                         self.force_direction * self.force_magnitude)
                    if self.logger is not None:
                        self.logger.log_event(self.sim, self.selected_cell)
            self.slider_dragging  = False
            self._slider_pending  = False

        return True

    def _handle_keydown(self, event):
        k = event.key
        if k == pygame.K_ESCAPE:
            return False
        elif k == pygame.K_a:
            self.sim.set_mode("A")
            self._notify("Mode A: Translational only")
        elif k == pygame.K_b:
            self.sim.set_mode("B")
            self._notify("Mode B: Translation + Rotation")
        elif k == pygame.K_f:
            self._force_preset_idx = (self._force_preset_idx + 1) % len(self.force_presets)
            self.force_direction   = self.force_presets[self._force_preset_idx].copy()
            self._notify(f"Force: ({self.force_direction[0]:+.1f}, {self.force_direction[1]:+.1f})")
            if self.selected_cell is not None:
                self.sim.apply_force(self.selected_cell.cell_id,
                                     self.force_direction * self.force_magnitude)
        elif k == pygame.K_c:
            self.sim.clear()
            self.selected_cell = None
            self._notify("Simulation cleared")
        elif k == pygame.K_h:
            self._notify("Opening plots...")
            return "SHOW_PLOTS"
        elif k == pygame.K_x:
            self._notify("Running experiment...")
            return "RUN_EXPERIMENT"
        elif k == pygame.K_p:
            self._notify("Generating PDF...")
            return "GEN_REPORT"
        elif k == pygame.K_v:
            self.vision_mode = not self.vision_mode
            self._notify("Vision ON" if self.vision_mode else "Vision OFF")
        elif k == pygame.K_r:
            return "SET_REFERENCE"
        elif k == pygame.K_m:
            self._cycle_heatmap()
        elif k == pygame.K_n:
            self.sim.set_anisotropy(not self.sim.use_anisotropy)
            self._notify(f"Anisotropy: {'ON' if self.sim.use_anisotropy else 'OFF'}")
        elif k == pygame.K_g:
            self.show_fiber_glyphs = not self.show_fiber_glyphs
            self._notify(f"Fiber glyphs: {'ON' if self.show_fiber_glyphs else 'OFF'}")
        elif k == pygame.K_q:
            self.show_confidence = not self.show_confidence
        return True

    def _handle_mousedown(self, event) -> None:
        mx, my = event.pos

        if mx < VIEWPORT_W and my < VIEWPORT_H:
            if event.button == 1:
                cell = self.mesh.get_cell_at_pixel(float(mx), float(my))
                if cell is not None:
                    if self.selected_cell is not None:
                        self.selected_cell.highlighted = False
                    self.selected_cell     = cell
                    cell.highlighted       = True
                    self.sim.apply_force(cell.cell_id,
                                         self.force_direction * self.force_magnitude)
                    self._notify(f"Cell #{cell.cell_id} ({cell.tissue_type})")
                    if self.logger is not None:
                        self.logger.log_event(self.sim, cell)
            elif event.button == 3:
                if self.selected_cell is not None:
                    self.selected_cell.highlighted = False
                    self.selected_cell = None
                self.sim.clear()

        elif mx >= VIEWPORT_W:
            px = mx - VIEWPORT_W
            if self._slider_rect.collidepoint(px, my):
                self.slider_dragging = True
                self._slider_pending = True
                self._update_slider_visual(px)
            if self._btn_a.collidepoint(px, my):
                self.sim.set_mode("A")
                self._notify("Mode A")
            if self._btn_b.collidepoint(px, my):
                self.sim.set_mode("B")
                self._notify("Mode B")
            if self._btn_heatmap.collidepoint(px, my):
                self._cycle_heatmap()
            if self._btn_aniso.collidepoint(px, my):
                self.sim.set_anisotropy(not self.sim.use_anisotropy)
                self._notify(f"Anisotropy: {'ON' if self.sim.use_anisotropy else 'OFF'}")
            if self._btn_export.collidepoint(px, my) and self.logger:
                self.logger.export_csv()
                self._notify("CSV saved to logs/")
            if self._btn_export_json.collidepoint(px, my) and self.logger:
                self.logger.export_json()
                self._notify("JSON saved to logs/")

    def _update_slider_visual(self, panel_x: int) -> None:
        sx, sw = 12, PANEL_W - 24
        t = np.clip((panel_x - sx) / sw, 0.0, 1.0)
        self.force_magnitude = FORCE_MAGNITUDE_MIN + t * (FORCE_MAGNITUDE_MAX - FORCE_MAGNITUDE_MIN)
        self._slider_pending = True

    def _cycle_heatmap(self) -> None:
        idx = HEATMAP_MODES.index(self.heatmap_mode)
        self.heatmap_mode = HEATMAP_MODES[(idx + 1) % len(HEATMAP_MODES)]
        self._notify(f"Map: {HEATMAP_LABELS[self.heatmap_mode]}")

    def update_vision_frame(self, bgr_frame: np.ndarray) -> None:
        rgb  = bgr_frame[:, :, ::-1]
        surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        self.vision_surface = surf
        self.vision_frame   = bgr_frame

    def _notify(self, msg: str, duration: int = 100) -> None:
        self._notification = msg
        self._notif_timer  = duration

    def tick(self, fps: int = 60) -> float:
        return self.clock.tick(fps)
