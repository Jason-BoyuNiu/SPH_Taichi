import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import taichi as ti

from config_builder import SimConfig
from particle_system import ParticleSystem


ti.init(arch=ti.gpu, device_memory_fraction=0.5)


def parse_vec3(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated numbers, got: {text}")
    return np.array([float(v) for v in parts], dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def look_at_basis(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up))
    cam_up = normalize(np.cross(right, forward))
    return np.stack([right, cam_up, forward], axis=0)


def world_to_camera(points_world: np.ndarray, eye: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (points_world - eye[None, :]) @ basis.T


def kinetic_to_white_red_colors(
    velocity: np.ndarray,
    is_fluid: np.ndarray,
    fallback_colors: np.ndarray,
    high_percentile: float = 95.0,
) -> np.ndarray:
    # Kinetic energy per unit mass: 0.5 * |v|^2
    ke = 0.5 * np.sum(velocity * velocity, axis=1)
    out = fallback_colors.copy()

    fluid_idx = np.where(is_fluid)[0]
    if fluid_idx.size == 0:
        return out

    fluid_ke = ke[fluid_idx]
    ke_hi = np.percentile(fluid_ke, high_percentile)
    ke_hi = max(float(ke_hi), 1e-8)
    t = np.clip(fluid_ke / ke_hi, 0.0, 1.0)

    # White (low) -> Red (high): [1,1,1] -> [1,0,0]
    out[fluid_idx, 0] = 1.0
    out[fluid_idx, 1] = 1.0 - t
    out[fluid_idx, 2] = 1.0 - t
    return out


def gaussian_splat_render(
    points_world: np.ndarray,
    colors: np.ndarray,
    camera_eye: np.ndarray,
    camera_lookat: np.ndarray,
    camera_up: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
    particle_radius: float,
    sigma_scale: float,
    alpha_scale: float,
    background: np.ndarray,
    near: float,
    min_sigma: float,
    max_sigma: float,
    max_kernel_radius: int,
) -> np.ndarray:
    basis = look_at_basis(camera_eye, camera_lookat, camera_up)
    points_cam = world_to_camera(points_world, camera_eye, basis)

    z = points_cam[:, 2]
    valid = z > near
    if not np.any(valid):
        return np.tile(background[None, None, :], (height, width, 1)).astype(np.float32)

    p = points_cam[valid]
    c = colors[valid]
    z = z[valid]

    f = 0.5 * height / math.tan(math.radians(fov_deg * 0.5))
    u = f * (p[:, 0] / z) + 0.5 * width
    v = 0.5 * height - f * (p[:, 1] / z)

    sigma = np.clip(f * particle_radius / (z + 1e-8) * sigma_scale, min_sigma, max_sigma)
    radius_px = np.minimum(np.ceil(3.0 * sigma).astype(np.int32), max_kernel_radius)

    in_view = (
        (u >= -max_kernel_radius)
        & (u < width + max_kernel_radius)
        & (v >= -max_kernel_radius)
        & (v < height + max_kernel_radius)
    )
    if not np.any(in_view):
        return np.tile(background[None, None, :], (height, width, 1)).astype(np.float32)

    u = u[in_view]
    v = v[in_view]
    sigma = sigma[in_view]
    radius_px = radius_px[in_view]
    c = c[in_view]

    accum_rgb = np.zeros((height, width, 3), dtype=np.float32)
    accum_alpha = np.zeros((height, width), dtype=np.float32)

    for i in range(u.shape[0]):
        r = int(radius_px[i])
        if r < 1:
            continue
        cx = int(round(u[i]))
        cy = int(round(v[i]))
        x0 = max(0, cx - r)
        x1 = min(width, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(height, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        xs = np.arange(x0, x1, dtype=np.float32) - u[i]
        ys = np.arange(y0, y1, dtype=np.float32) - v[i]
        dx2 = xs[None, :] * xs[None, :]
        dy2 = ys[:, None] * ys[:, None]
        g = np.exp(-(dx2 + dy2) / (2.0 * sigma[i] * sigma[i]))

        a = alpha_scale * g
        accum_alpha[y0:y1, x0:x1] += a
        accum_rgb[y0:y1, x0:x1] += a[..., None] * c[i]

    safe_alpha = np.maximum(accum_alpha, 1e-6)
    rgb = accum_rgb / safe_alpha[..., None]
    opacity = 1.0 - np.exp(-accum_alpha)
    frame = rgb * opacity[..., None] + background[None, None, :] * (1.0 - opacity[..., None])
    frame = np.clip(frame, 0.0, 1.0)
    frame = np.power(frame, 1.0 / 2.2)
    return frame.astype(np.float32)


def encode_video_with_opencv(frames_dir: Path, fps: int, output_video: Path) -> bool:
    frame_files = sorted(frames_dir.glob("*.png"))
    if len(frame_files) == 0:
        print(f"No PNG frames found in: {frames_dir}")
        return False

    first = cv2.imread(str(frame_files[0]))
    if first is None:
        print(f"Failed to read first frame: {frame_files[0]}")
        return False

    h, w = first.shape[:2]
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        print(f"Failed to open VideoWriter: {output_video}")
        return False

    written = 0
    for frame_file in frame_files:
        img = cv2.imread(str(frame_file))
        if img is None:
            continue
        if img.shape[0] != h or img.shape[1] != w:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)
        written += 1
    writer.release()

    if written == 0:
        print(f"No readable frames in: {frames_dir}")
        return False

    print(f"Video encoded with OpenCV: {output_video} (frames={written}, fps={fps}, size={w}x{h})")
    return True


def main():
    parser = argparse.ArgumentParser(description="SPH simulation + 3D Gaussian splatting renderer")
    parser.add_argument("--scene_file", required=True, help="Path to scene json")
    parser.add_argument("--frames", type=int, default=120, help="How many rendered frames to output")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for output video")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera_pos", default="5.5,2.5,4.0")
    parser.add_argument("--camera_lookat", default="-1.0,0.0,0.0")
    parser.add_argument("--camera_up", default="0.0,1.0,0.0")
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--background", default="0,0,0", help="RGB in [0,1], e.g. 0.02,0.03,0.05")
    parser.add_argument("--sigma_scale", type=float, default=1.6, help="Controls Gaussian footprint size")
    parser.add_argument("--alpha_scale", type=float, default=0.75, help="Controls opacity per splat")
    parser.add_argument("--min_sigma", type=float, default=0.6)
    parser.add_argument("--max_sigma", type=float, default=6.0)
    parser.add_argument("--max_kernel_radius", type=int, default=18, help="Clamp per-particle kernel radius in pixels")
    parser.add_argument(
        "--kinetic_percentile",
        type=float,
        default=95.0,
        help="Fluid kinetic-energy percentile mapped to pure red (lower value => more red overall)",
    )
    parser.add_argument(
        "--max_render_particles",
        type=int,
        default=60000,
        help="Subsample particles when scene is very large",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="", help="Directory for rendered PNG frames")
    parser.add_argument("--video_path", default="", help="Output mp4 path; encoded from PNG frames using OpenCV")
    parser.add_argument(
        "--invisible_objects",
        default="",
        help="Comma-separated object IDs to hide, e.g. 1,2",
    )
    parser.add_argument(
        "--substeps_override",
        type=int,
        default=84,
        help="If >0, overrides numberOfStepsPerRenderUpdate from scene config",
    )
    args = parser.parse_args()

    scene_path = Path(args.scene_file)
    scene_name = scene_path.stem

    config = SimConfig(scene_file_path=str(scene_path))
    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    if args.substeps_override > 0:
        substeps = args.substeps_override

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{scene_name}_gaussian_frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = Path(args.video_path) if args.video_path else Path(f"{scene_name}_gaussian.mp4")

    invisible_objects = set()
    if args.invisible_objects.strip():
        invisible_objects = {int(x.strip()) for x in args.invisible_objects.split(",") if x.strip()}

    camera_eye = parse_vec3(args.camera_pos)
    camera_lookat = parse_vec3(args.camera_lookat)
    camera_up = parse_vec3(args.camera_up)
    background = np.clip(parse_vec3(args.background), 0.0, 1.0)

    ps = ParticleSystem(config, GGUI=False)
    solver = ps.build_solver()
    solver.initialize()

    rng = np.random.default_rng(args.seed)

    for frame_idx in range(args.frames):
        for _ in range(substeps):
            solver.step()

        n = ps.particle_num[None]
        pos = ps.x.to_numpy()[:n].astype(np.float32)
        vel = ps.v.to_numpy()[:n].astype(np.float32)
        mat = ps.material.to_numpy()[:n]
        base_col = (ps.color.to_numpy()[:n].astype(np.float32) / 255.0).clip(0.0, 1.0)
        obj_id = ps.object_id.to_numpy()[:n]

        if invisible_objects:
            mask = ~np.isin(obj_id, list(invisible_objects))
            pos = pos[mask]
            vel = vel[mask]
            mat = mat[mask]
            base_col = base_col[mask]

        if args.max_render_particles > 0 and pos.shape[0] > args.max_render_particles:
            sample_idx = rng.choice(pos.shape[0], size=args.max_render_particles, replace=False)
            pos = pos[sample_idx]
            vel = vel[sample_idx]
            mat = mat[sample_idx]
            base_col = base_col[sample_idx]

        col = kinetic_to_white_red_colors(
            velocity=vel,
            is_fluid=(mat == ps.material_fluid),
            fallback_colors=base_col,
            high_percentile=args.kinetic_percentile,
        )

        frame = gaussian_splat_render(
            points_world=pos,
            colors=col,
            camera_eye=camera_eye,
            camera_lookat=camera_lookat,
            camera_up=camera_up,
            fov_deg=args.fov,
            width=args.width,
            height=args.height,
            particle_radius=ps.particle_radius,
            sigma_scale=args.sigma_scale,
            alpha_scale=args.alpha_scale,
            background=background,
            near=args.near,
            min_sigma=args.min_sigma,
            max_sigma=args.max_sigma,
            max_kernel_radius=args.max_kernel_radius,
        )

        frame_file = output_dir / f"{frame_idx:06d}.png"
        # ti.tools.imwrite expects image layout as (W, H, C) and writes Y bottom-up.
        # Renderer output is (H, W, C) with top-left as origin, so we flip Y then swap axes.
        frame_for_taichi = np.swapaxes(frame[::-1, :, :], 0, 1)
        ti.tools.imwrite(frame_for_taichi, str(frame_file))
        print(f"[{frame_idx + 1}/{args.frames}] wrote {frame_file}")

    encoded = encode_video_with_opencv(output_dir, args.fps, output_video)
    if encoded:
        print(f"Video ready: {output_video}")
    else:
        print("OpenCV encoding failed; PNG frame sequence is still available:")
        print(output_dir)
        print(f"Use OpenCV or ffmpeg manually on {output_dir}")


if __name__ == "__main__":
    main()
