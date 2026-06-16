import sys
import os
import argparse
import numpy as np

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

from mesh import HexMesh
from simulation import Simulation
from ui import UIRenderer
from vision import CameraCapture, OpticalFlowTracker, DeformationMapper, build_stream_url
from data_logger import DataLogger
from report_generator import ReportGenerator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tissue Digital Twin v2")
    p.add_argument("--vision",      action="store_true")
    p.add_argument("--source",      default="ipwebcam",
                   choices=["droidcam", "ipwebcam", "rtsp", "device0", "device1"])
    p.add_argument("--ip",          default="192.168.1.100")
    p.add_argument("--rows",        type=int, default=10)
    p.add_argument("--cols",        type=int, default=14)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--experiment",  action="store_true",
                   help="Run automated experiment and exit (no GUI)")
    p.add_argument("--report",      action="store_true",
                   help="Generate PDF report and exit (no GUI)")
    p.add_argument("--no-aniso",    action="store_true",
                   help="Disable anisotropic tissue model")
    p.add_argument("--log-dir",     default="logs")
    return p.parse_args()


def print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║        TISSUE DIGITAL TWIN — Research Prototype v2                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Click a hex cell → apply force and see stiffness estimates              ║
║  A/B = mode | M = heatmap | H = plots | X = experiment | P = PDF        ║
║  N = anisotropy | G = fiber glyphs | C = clear | Esc = quit             ║
║                                                                          ║
║  DISCLAIMER: Not a clinical tool. Simplified spring-network model.      ║
╚══════════════════════════════════════════════════════════════════════════╝
""")



def _drain_events(ui: UIRenderer, sim: Simulation, logger: DataLogger):
    actions = []
    for event in pygame.event.get():
        result = ui.handle_event(event)
        if result is False:
            return False
        if isinstance(result, str):
            actions.append(result)
    return actions


def run_simulation_mode(ui: UIRenderer, sim: Simulation, logger: DataLogger) -> None:
    running = True
    while running:
        # Drain all events first — never block on a single event
        result = _drain_events(ui, sim, logger)
        if result is False:
            break

        # Handle action signals
        for action in (result or []):
            if action == "SHOW_PLOTS":
                _open_analysis_plots(sim)
            elif action == "RUN_EXPERIMENT":
                _run_ui_experiment(sim, logger, ui)
            elif action == "GEN_REPORT":
                _generate_report(sim, logger, ui)

        # Draw current state
        ui.draw()
        # Cap at 60 fps — prevents CPU spinning
        ui.tick(60)


def run_vision_mode(ui, sim, cam, tracker, mapper, logger) -> None:
    running       = True
    frame_counter = 0

    while running:
        result = _drain_events(ui, sim, logger)
        if result is False:
            break

        for action in (result or []):
            if action == "SHOW_PLOTS":
                _open_analysis_plots(sim)
            elif action == "RUN_EXPERIMENT":
                _run_ui_experiment(sim, logger, ui)
            elif action == "GEN_REPORT":
                _generate_report(sim, logger, ui)
            elif action == "SET_REFERENCE":
                if ui.vision_frame is not None:
                    n = tracker.set_reference(ui.vision_frame)
                    ui._notify(f"Reference set: {n} feature points")
                else:
                    ui._notify("No camera frame yet")

        # Camera — process every 2nd frame to keep UI responsive
        if cam.is_open:
            frame = cam.read_frame()
            if frame is not None:
                frame_counter += 1
                if frame_counter % 2 == 0 and tracker.has_reference:
                    flow = tracker.update(frame)
                    if flow["active_mask"].sum() > 0:
                        mapper.update_mesh_from_tracking(sim.mesh, flow, scale_factor=1.5)
                ui.update_vision_frame(tracker.draw_tracking_overlay(frame))

        ui.draw()
        ui.tick(60)



def _open_analysis_plots(sim: Simulation) -> None:
    from visualization import (plot_stiffness_comparison, plot_force_displacement,
                               plot_anisotropy, plot_twin_sync_pipeline)
    try:
        plot_stiffness_comparison(sim, show=True)
        plot_anisotropy(sim, show=True)
        plot_twin_sync_pipeline(sim, show=True)
        forces, disps = sim.get_force_displacement_history()
        if forces:
            plot_force_displacement(forces, disps, show=True)
        print("[Analysis] Close the matplotlib windows to resume simulation.")
    except Exception as e:
        print(f"[Analysis] Error: {e}")


def _run_ui_experiment(sim: Simulation, logger: DataLogger, ui: UIRenderer) -> None:
    ui._notify("Running experiment… see terminal")
    ui.draw()
    pygame.display.flip()
    try:
        os.makedirs("logs", exist_ok=True)
        results = sim.run_experiment()
        stats   = results.summary_statistics()
        imp     = stats.get("improvement_pct")
        imp_str = f"{imp:.1f}%" if imp is not None else "n/a"
        results.export_csv("logs/experiment_results.csv")
        results.export_json("logs/experiment_report.json")
        print(f"[Experiment] RMSE A={stats.get('rmse_mode_A') or 'n/a'}  "
              f"B={stats.get('rmse_mode_B') or 'n/a'}  "
              f"improvement={imp_str}")
        ui._notify(f"Experiment done — B improvement: {imp_str}")
        from visualization import plot_experiment_results
        plot_experiment_results(results, show=True)
    except Exception as e:
        print(f"[Experiment] Error: {e}")
        ui._notify("Experiment failed — see terminal")


def _generate_report(sim: Simulation, logger: DataLogger, ui: UIRenderer) -> None:
    ui._notify("Generating PDF… (~5 s)")
    ui.draw()
    pygame.display.flip()
    try:
        gen  = ReportGenerator(sim, logger)
        path = gen.generate("research_report.pdf")
        ui._notify(f"Saved: {path}")
        print(f"[Report] PDF saved: {path}")
    except Exception as e:
        print(f"[Report] Error: {e}")
        ui._notify("Report failed — see terminal")



def run_headless_experiment(sim: Simulation, logger: DataLogger) -> None:
    print("[Experiment] Running automated experiment…")
    results = sim.run_experiment(
        force_levels     = [4.0, 8.0, 12.0, 16.0, 20.0],
        force_directions = [np.array([1.0, 0.0]),
                            np.array([0.0, 1.0]),
                            np.array([0.707, 0.707])],
        modes = ["A", "B"],
    )
    stats = results.summary_statistics()
    print(f"\n  Records  : {stats['n_total']}  (valid: {stats['n_valid']})")
    print(f"  RMSE A   : {stats['rmse_mode_A']:.4f}" if stats['rmse_mode_A'] else "  RMSE A: n/a")
    print(f"  RMSE B   : {stats['rmse_mode_B']:.4f}" if stats['rmse_mode_B'] else "  RMSE B: n/a")
    imp = stats.get("improvement_pct")
    print(f"  B improv : {imp:.2f}%" if imp is not None else "  B improv: n/a")
    os.makedirs("logs", exist_ok=True)
    results.export_csv("logs/experiment_results.csv")
    results.export_json("logs/experiment_report.json")

    try:
        from visualization import plot_experiment_results
        import matplotlib.pyplot as plt
        plot_experiment_results(results, show=True)
        plt.show(block=True)
    except Exception as e:
        print(f"[Experiment] Plot skipped: {e}")



def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    print_banner()

    print(f"[Init] Building {args.cols}×{args.rows} mesh…")
    mesh = HexMesh(cols=args.cols, rows=args.rows, hex_radius=38.0, origin=(80.0, 60.0))
    sim  = Simulation(mesh)
    if args.no_aniso:
        sim.set_anisotropy(False)
    print(f"[Init] {len(mesh)} cells  |  anisotropy={'ON' if sim.use_anisotropy else 'OFF'}")

    logger = DataLogger(output_dir=args.log_dir)

    if args.experiment:
        run_headless_experiment(sim, logger)
        return

    if args.report:
        gen = ReportGenerator(sim, logger)
        gen.generate("research_report.pdf")
        return

    ui = UIRenderer(sim, logger=logger)
    ui.vision_mode = args.vision

    cam = tracker = mapper = None
    if args.vision:
        url = build_stream_url(args.source, args.ip)
        print(f"[Vision] Connecting: {url}")
        cam = CameraCapture(url)
        if cam.open():
            tracker = OpticalFlowTracker()
            mapper  = DeformationMapper(search_radius=80.0)
            mapper.set_coordinate_mapping(
                cam_bounds  = (0, 0, 640, 480),
                mesh_bounds = (80, 60,
                               80 + mesh.cols * mesh.hex_width * 0.75,
                               60 + mesh.rows * mesh.hex_height),
            )
            print("[Vision] Camera open. Press R to set reference frame.")
        else:
            print("[Vision] Could not open camera — falling back to simulation mode.")
            ui.vision_mode = False
            cam = None

    try:
        if ui.vision_mode and cam is not None:
            run_vision_mode(ui, sim, cam, tracker, mapper, logger)
        else:
            run_simulation_mode(ui, sim, logger)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        if logger.records:
            print(f"[Main] Auto-saving {len(logger.records)} log records…")
            os.makedirs(args.log_dir, exist_ok=True)
            logger.export_csv()
            logger.export_json()
        if cam is not None:
            cam.release()
        pygame.quit()
        print("[Main] Session ended.")


if __name__ == "__main__":
    main()
