import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_float_list(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def tok(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return s.replace("-", "m").replace(".", "p")


def write_trapezoid_basin_obj(
    out_obj: Path,
    length_x: float,
    half_outer: float,
    half_bench: float,
    half_bottom: float,
    top_elev: float,
):
    # Mesh axes are solver-world axes:
    # X length, Y elevation, Z width
    x0 = 0.0
    x1 = float(length_x)
    profile = [
        (-half_outer, top_elev),
        (-half_bench, top_elev),
        (-half_bottom, 0.0),
        (half_bottom, 0.0),
        (half_bench, top_elev),
        (half_outer, top_elev),
    ]  # (z, y)

    verts = []
    for x in (x0, x1):
        for z, y in profile:
            verts.append((x, y, z))

    n = len(profile)

    def vid(ix, ip):
        return ix * n + ip + 1

    faces = []
    # Side strips along X. This gives two top long planes, two side slopes, one bottom plane.
    for i in range(n - 1):
        a = vid(0, i)
        b = vid(0, i + 1)
        c = vid(1, i + 1)
        d = vid(1, i)
        faces.append((a, b, c))
        faces.append((a, c, d))

    out_obj.parent.mkdir(parents=True, exist_ok=True)
    with out_obj.open("w", encoding="utf-8") as f:
        f.write("# trapezoid stilling basin shell\n")
        f.write("# axes: X=length, Y=elevation, Z=width\n")
        f.write(
            f"# params: L={length_x}, half_outer={half_outer}, half_bench={half_bench}, half_bottom={half_bottom}, top_elev={top_elev}\n"
        )
        for x, y, z in verts:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")


def build_scene_dict(
    model_rel_path: str,
    basin_length: float,
    tailwater_depth: float,
    inflow_speed: float,
):
    # Basin geometric constants
    half_outer = 3.0
    half_bench = 1.9
    half_bottom = 0.7
    top_elev = 1.1
    particle_radius = 0.02
    padding = 4.0 * particle_radius
    x_off, y_off, z_off = 3.0 * padding, 0.1, 3.5

    # Domain follows basin length
    domain_end = [x_off + basin_length + 0.08, 2.4, z_off + half_outer + padding]

    # Tailwater in pit (stage-1)
    tw_start_y = y_off + 0.12
    tw_end_y = tw_start_y + tailwater_depth
    fluid_tailwater = {
        "objectId": 0,
        "start": [x_off + 0.2, tw_start_y, z_off - half_bottom + 0.05],
        "end": [x_off + basin_length - 0.2, tw_end_y, z_off + half_bottom - 0.05],
        "translation": [0.0, 0.0, 0.0],
        "scale": [1, 1, 1],
        "velocity": [0.0, 0.0, 0.0],
        "density": 1000.0,
        "color": [70, 145, 245],
    }

    # Delayed side-platform inflow (stage-2), flowing toward pit center (negative Z)
    fluid_inflow = {
        "objectId": 1,
        "start": [x_off + 0.35 * basin_length, y_off + top_elev + 0.05, z_off + half_bench + 0.08],
        "end": [x_off + 0.65 * basin_length, y_off + top_elev + 0.22, z_off + half_outer - 0.08],
        "translation": [0.0, 0.0, 0.0],
        "scale": [1, 1, 1],
        "velocity": [0.0, -0.35, -inflow_speed],
        "density": 1000.0,
        "color": [40, 105, 230],
        "startTime": 0.8,
    }

    # Thin static stopper near x-min to suppress premature outflow toward x=0.
    rigid_stopper = {
        "objectId": 200,
        "start": [x_off, y_off, 0.0],
        "end": [x_off + 0.12, y_off + top_elev + 0.4, domain_end[2]],
        "translation": [0.0, 0.0, 0.0],
        "scale": [1, 1, 1],
        "velocity": [0.0, 0.0, 0.0],
        "density": 1200.0,
        "color": [165, 165, 165],
        "isDynamic": False,
    }

    scene = {
        "DesignParameters": {
            "description": "Trapezoid-section stilling basin hydraulic-jump scan scene",
            "basinLength": basin_length,
            "tailwaterDepth": tailwater_depth,
            "inflowSpeedTowardPit": inflow_speed,
            "crossSection": "trapezoid (two upper long planes + two side slopes + one bottom plane)",
        },
        "Configuration": {
            "domainStart": [0.0, 0.0, 0.0],
            "domainEnd": domain_end,
            "particleRadius": 0.02,
            "numberOfStepsPerRenderUpdate": 2,
            "density0": 1000,
            "simulationMethod": 4,
            "gravitation": [0.0, -9.81, 0.0],
            "timeStepSize": 0.00025,
            "stiffness": 50000,
            "exponent": 7,
            "boundaryHandlingMethod": 0,
            "exportFrame": False,
            "exportPly": False,
            "exportObj": False,
        },
        "RigidBodies": [
            {
                "objectId": 100,
                "geometryFile": model_rel_path,
                "translation": [x_off, y_off, z_off],
                "rotationAxis": [0, 1, 0],
                "rotationAngle": 0,
                "scale": [1, 1, 1],
                "velocity": [0.0, 0.0, 0.0],
                "density": 1200.0,
                "color": [180, 180, 180],
                "isDynamic": False,
                "voxelMode": "fill",
                "repairHoles": True,
            }
        ],
        "RigidBlocks": [rigid_stopper],
        "FluidBlocks": [fluid_tailwater, fluid_inflow],
    }
    return scene


def run():
    parser = argparse.ArgumentParser(description="Scan trapezoid-basin hydraulic-jump parameters")
    parser.add_argument("--output_root", default="scan_runs/trapezoid_basin_scan")
    parser.add_argument("--basin_lengths", default="2.0,2.6")
    parser.add_argument("--tailwater_depths", default="0.14,0.22")
    parser.add_argument("--inflow_speeds", default="3.5,4.0")
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--substeps_override", type=int, default=8)
    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    scenes_dir = out_root / "scenes"
    models_dir = out_root / "models"
    frames_root = out_root / "frames"
    videos_dir = out_root / "videos"
    for d in (scenes_dir, models_dir, frames_root, videos_dir):
        d.mkdir(parents=True, exist_ok=True)

    basin_lengths = parse_float_list(args.basin_lengths)
    tailwater_depths = parse_float_list(args.tailwater_depths)
    inflow_speeds = parse_float_list(args.inflow_speeds)
    cases = list(itertools.product(basin_lengths, tailwater_depths, inflow_speeds))
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    # Generate one model per basin length
    model_map = {}
    for L in basin_lengths:
        model_name = f"trapezoid_basin_L{tok(L)}.obj"
        model_path = models_dir / model_name
        write_trapezoid_basin_obj(
            out_obj=model_path,
            length_x=L,
            half_outer=3.0,
            half_bench=1.9,
            half_bottom=0.7,
            top_elev=1.1,
        )
        model_map[L] = model_path

    rows = []
    for i, (L, tw, v) in enumerate(cases, start=1):
        case_name = f"L{tok(L)}_tw{tok(tw)}_v{tok(v)}"
        scene_path = scenes_dir / f"{case_name}.json"
        frame_dir = frames_root / case_name
        video_path = videos_dir / f"{case_name}.mp4"

        model_rel = str(model_map[L].as_posix())
        if not model_rel.startswith("./"):
            model_rel = "./" + model_rel
        scene = build_scene_dict(model_rel_path=model_rel, basin_length=L, tailwater_depth=tw, inflow_speed=v)
        scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-u",
            "run_simulation_gaussian.py",
            "--scene_file",
            str(scene_path),
            "--frames",
            str(args.frames),
            "--fps",
            str(args.fps),
            "--output_dir",
            str(frame_dir),
            "--video_path",
            str(video_path),
            "--camera_pos",
            f"{0.6 + L + 3.0},0.95,3.5",
            "--camera_lookat",
            f"{0.6 + 0.5 * L},0.95,3.5",
            "--camera_up",
            "0,1,0",
            "--fov",
            "60",
            "--sigma_scale",
            "1.2",
            "--alpha_scale",
            "0.65",
            "--substeps_override",
            str(args.substeps_override),
        ]

        if args.dry_run:
            print(f"[{i}/{len(cases)}] DRY-RUN {case_name}")
            rc = 0
            status = "dry_run"
        else:
            print(f"[{i}/{len(cases)}] RUN {case_name}")
            proc = subprocess.run(cmd)
            rc = proc.returncode
            status = "ok" if rc == 0 else "failed"
            if rc != 0:
                print(f"Case failed: {case_name}, rc={rc}")

        rows.append(
            {
                "case_name": case_name,
                "status": status,
                "return_code": rc,
                "basin_length": L,
                "tailwater_depth": tw,
                "inflow_speed": v,
                "scene_file": str(scene_path),
                "model_file": str(model_map[L]),
                "frames_dir": str(frame_dir),
                "video_file": str(video_path),
            }
        )

    manifest = out_root / "scan_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"Finished. total={len(rows)} ok={ok} dry_run={args.dry_run}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    run()
