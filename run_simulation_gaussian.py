import argparse
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import taichi as ti

from config_builder import SimConfig
from particle_system import ParticleSystem

from tqdm import tqdm


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


def normalize_last_dim(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def build_background_image(
    width: int,
    height: int,
    base_color: np.ndarray,
    top_color: np.ndarray,
    bottom_color: np.ndarray,
    mode: str,
) -> np.ndarray:
    yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    grad = bottom_color[None, None, :] * (1.0 - yy) + top_color[None, None, :] * yy
    img = np.tile(grad, (1, width, 1))

    if mode == "plain":
        img[...] = base_color[None, None, :]
        return img.astype(np.float32)

    xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]

    if mode == "checker":
        cell = 0.07
        checker = (((xx / cell).astype(np.int32) + (yy / cell).astype(np.int32)) % 2).astype(np.float32)
        checker = checker * 0.12 + 0.88
        img = img * checker
    else:
        # "studio": subtle procedural pattern to make refraction visible.
        band = 0.06 * np.sin(xx * 24.0 + yy * 11.0)
        vignette_x = (xx - 0.5) ** 2
        vignette_y = (yy - 0.45) ** 2
        vignette = 1.0 - 0.45 * (vignette_x + vignette_y)
        img = img * (1.0 + band) * np.clip(vignette, 0.55, 1.0)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def blur_box_3x3(src: np.ndarray, passes: int) -> np.ndarray:
    out = src.astype(np.float32).copy()
    for _ in range(max(0, passes)):
        out = (
            out
            + np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
            + np.roll(np.roll(out, 1, axis=0), 1, axis=1)
            + np.roll(np.roll(out, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(out, -1, axis=0), 1, axis=1)
            + np.roll(np.roll(out, -1, axis=0), -1, axis=1)
        ) / 9.0
    return out


def sample_image_bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    u = np.clip(u, 0.0, w - 1.0)
    v = np.clip(v, 0.0, h - 1.0)

    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)

    fu = (u - x0).astype(np.float32)
    fv = (v - y0).astype(np.float32)

    c00 = image[y0, x0]
    c10 = image[y0, x1]
    c01 = image[y1, x0]
    c11 = image[y1, x1]

    c0 = c00 * (1.0 - fu[..., None]) + c10 * fu[..., None]
    c1 = c01 * (1.0 - fu[..., None]) + c11 * fu[..., None]
    return c0 * (1.0 - fv[..., None]) + c1 * fv[..., None]


def sample_environment(reflect_dir: np.ndarray, sun_power: float) -> np.ndarray:
    # Camera-space procedural environment: sky-ground gradient + sharp sun lobe.
    up_t = np.clip(0.5 * (reflect_dir[..., 1] + 1.0), 0.0, 1.0)
    sky = np.array([0.50, 0.73, 0.98], dtype=np.float32)
    horizon = np.array([0.92, 0.92, 0.90], dtype=np.float32)
    ground = np.array([0.34, 0.30, 0.26], dtype=np.float32)
    base = (
        (ground[None, None, :] * (1.0 - up_t[..., None]) + horizon[None, None, :] * up_t[..., None]) * 0.55
        + sky[None, None, :] * np.power(up_t[..., None], 2.0) * 0.45
    )

    sun_dir = normalize(np.array([0.25, 0.84, -0.48], dtype=np.float32))
    sun_dot = np.clip(np.sum(reflect_dir * sun_dir[None, None, :], axis=-1), 0.0, 1.0)
    sun = np.power(sun_dot, sun_power)[..., None] * np.array([1.0, 0.98, 0.90], dtype=np.float32)[None, None, :]
    return np.clip(base + 1.8 * sun, 0.0, 2.0).astype(np.float32)


def gaussian_splat_render(
    points_world: np.ndarray,
    colors: np.ndarray,
    normals_world: np.ndarray,
    camera_eye: np.ndarray,
    camera_lookat: np.ndarray,
    camera_up: np.ndarray,
    fov_deg: float,
    width: int,
    height: int,
    particle_radius: float,
    sigma_scale: float,
    alpha_scale: float,
    background_image: np.ndarray,
    near: float,
    min_sigma: float,
    max_sigma: float,
    max_kernel_radius: int,
    absorption_coeff: np.ndarray,
    refraction_strength: float,
    normal_strength: float,
    specular_strength: float,
    fresnel_f0: float,
    sun_power: float,
    opacity_gain: float,
    thickness_blur_passes: int,
    normal_source: str,
    particle_normal_mix: float,
) -> np.ndarray:
    basis = look_at_basis(camera_eye, camera_lookat, camera_up)
    points_cam = world_to_camera(points_world, camera_eye, basis)

    z = points_cam[:, 2]
    valid = z > near
    if not np.any(valid):
        return np.power(np.clip(background_image, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)

    p = points_cam[valid]
    c = colors[valid]
    n_world = normals_world[valid]
    z = z[valid]
    n_cam = normalize_last_dim(n_world @ basis.T)

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
        return np.power(np.clip(background_image, 0.0, 1.0), 1.0 / 2.2).astype(np.float32)

    u = u[in_view]
    v = v[in_view]
    sigma = sigma[in_view]
    radius_px = radius_px[in_view]
    c = c[in_view]
    n_cam = n_cam[in_view]

    accum_thickness = np.zeros((height, width), dtype=np.float32)
    accum_tint = np.zeros((height, width, 3), dtype=np.float32)
    accum_depth = np.zeros((height, width), dtype=np.float32)
    accum_normal = np.zeros((height, width, 3), dtype=np.float32)

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

        # Pass 1: additive blending for thickness.
        a = alpha_scale * g
        accum_thickness[y0:y1, x0:x1] += a
        accum_tint[y0:y1, x0:x1] += a[..., None] * c[i]
        accum_depth[y0:y1, x0:x1] += a * z[i]
        accum_normal[y0:y1, x0:x1] += a[..., None] * n_cam[i]

    tau = accum_thickness
    mask = tau > 1e-6
    safe_tau = np.maximum(tau, 1e-6)

    tint = accum_tint / safe_tau[..., None]
    mean_depth = accum_depth / safe_tau

    smooth_depth = blur_box_3x3(mean_depth, thickness_blur_passes)
    gx = np.zeros_like(smooth_depth, dtype=np.float32)
    gy = np.zeros_like(smooth_depth, dtype=np.float32)
    gx[:, 1:-1] = 0.5 * (smooth_depth[:, 2:] - smooth_depth[:, :-2])
    gy[1:-1, :] = 0.5 * (smooth_depth[2:, :] - smooth_depth[:-2, :])
    gx *= mask
    gy *= mask

    # Screen-space normal from depth gradient.
    nx = -gx * normal_strength
    ny = -gy * normal_strength
    nz = -np.ones_like(nx, dtype=np.float32)
    n_len = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    normal_screen = np.stack([nx / n_len, ny / n_len, nz / n_len], axis=-1)

    normal_particle = normalize_last_dim(accum_normal)
    if normal_source == "particle":
        normal = normal_particle
    elif normal_source == "screen":
        normal = normal_screen
    else:
        mix = np.clip(particle_normal_mix, 0.0, 1.0)
        normal = normalize_last_dim(mix * normal_particle + (1.0 - mix) * normal_screen)

    # Refraction: sample shifted background image.
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    refraction_mag = refraction_strength * np.clip(1.0 - np.exp(-tau), 0.0, 1.0)
    offset_scale = min(width, height) * refraction_mag
    sample_u = xs + normal[..., 0] * offset_scale
    sample_v = ys - normal[..., 1] * offset_scale
    bg_refracted = sample_image_bilinear(background_image, sample_u, sample_v)

    # Beer-Lambert transmittance.
    transmittance = np.exp(-tau[..., None] * absorption_coeff[None, None, :])
    diffuse = transmittance * bg_refracted + (1.0 - transmittance) * tint

    # Fresnel + procedural environment-map specular.
    f = 0.5 * height / math.tan(math.radians(fov_deg * 0.5))
    ray_x = (xs + 0.5 - 0.5 * width) / f
    ray_y = -(ys + 0.5 - 0.5 * height) / f
    ray_z = np.ones((height, width), dtype=np.float32)
    ray_len = np.sqrt(ray_x * ray_x + ray_y * ray_y + ray_z * ray_z) + 1e-6
    ray_dir = np.stack([ray_x / ray_len, ray_y / ray_len, ray_z / ray_len], axis=-1)
    view_dir = -ray_dir

    # Orient normal towards the viewer to avoid back-face flips from noisy estimates.
    flip = np.sum(normal * view_dir, axis=-1, keepdims=True) < 0.0
    normal = np.where(flip, -normal, normal)
    ndotv = np.clip(np.sum(normal * view_dir, axis=-1), 0.0, 1.0)
    fresnel = fresnel_f0 + (1.0 - fresnel_f0) * np.power(1.0 - ndotv, 5.0)
    reflect_dir = 2.0 * ndotv[..., None] * normal - view_dir
    reflect_len = np.linalg.norm(reflect_dir, axis=-1, keepdims=True) + 1e-6
    reflect_dir = reflect_dir / reflect_len

    env_spec = sample_environment(reflect_dir, sun_power)
    specular = specular_strength * fresnel[..., None] * env_spec

    fluid_rgb = diffuse + specular
    opacity = 1.0 - np.exp(-tau * opacity_gain)
    opacity *= mask.astype(np.float32)
    frame = fluid_rgb * opacity[..., None] + background_image * (1.0 - opacity[..., None])
    frame = np.clip(frame, 0.0, 1.0)
    frame = np.power(frame, 1.0 / 2.2)
    return frame.astype(np.float32)


def try_encode_video_with_ffmpeg(frames_dir: Path, fps: int, output_video: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False

    output_video.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.png"),
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        str(output_video),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("ffmpeg encode failed:")
        print(proc.stderr[-2000:])
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="SPH simulation + 3D Gaussian splatting renderer")
    parser.add_argument("--scene_file", required=True, help="Path to scene json")
    parser.add_argument("--frames", type=int, default=240, help="How many rendered frames to output")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for output video")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera_pos", default="5.5,2.5,4.0")
    parser.add_argument("--camera_lookat", default="-1.0,0.0,0.0")
    parser.add_argument("--camera_up", default="0.0,1.0,0.0")
    parser.add_argument("--fov", type=float, default=70.0)
    parser.add_argument("--near", type=float, default=0.05)
    parser.add_argument("--background", default="0,0,0", help="RGB in [0,1], e.g. 0.02,0.03,0.05")
    parser.add_argument("--background_top", default="0.55,0.72,0.95", help="Gradient top RGB in [0,1]")
    parser.add_argument("--background_bottom", default="0.11,0.13,0.16", help="Gradient bottom RGB in [0,1]")
    parser.add_argument(
        "--background_mode",
        default="studio",
        choices=["plain", "studio", "checker"],
        help="Background pattern for better refraction cues",
    )
    parser.add_argument("--sigma_scale", type=float, default=1.6, help="Controls Gaussian footprint size")
    parser.add_argument("--alpha_scale", type=float, default=0.75, help="Controls thickness contribution per splat")
    parser.add_argument("--min_sigma", type=float, default=0.6)
    parser.add_argument("--max_sigma", type=float, default=6.0)
    parser.add_argument("--max_kernel_radius", type=int, default=18, help="Clamp per-particle kernel radius in pixels")
    parser.add_argument("--absorption", default="2.4,1.2,0.35", help="Beer-Lambert absorption coeff RGB")
    parser.add_argument("--refraction_strength", type=float, default=0.02, help="Normal-based UV distortion amount")
    parser.add_argument("--normal_strength", type=float, default=95.0, help="Amplifies depth-gradient normal")
    parser.add_argument("--specular_strength", type=float, default=0.85, help="Environment specular amount")
    parser.add_argument("--fresnel_f0", type=float, default=0.02, help="Base Fresnel reflectance")
    parser.add_argument("--sun_power", type=float, default=384.0, help="Procedural sun highlight sharpness")
    parser.add_argument("--opacity_gain", type=float, default=1.4, help="Opacity growth against accumulated thickness")
    parser.add_argument("--thickness_blur_passes", type=int, default=1, help="Smoothing passes before normal extraction")
    parser.add_argument(
        "--normal_source",
        default="hybrid",
        choices=["particle", "screen", "hybrid"],
        help="Source of refraction/specular normal",
    )
    parser.add_argument("--particle_normal_mix", type=float, default=0.75, help="Hybrid blend weight for particle normal")
    parser.add_argument("--normal_radius_scale", type=float, default=1.0, help="Neighbor radius scale for particle normal estimation")
    parser.add_argument("--normal_min_neighbors", type=int, default=10, help="Minimum neighbor count to accept particle normal")
    parser.add_argument(
        "--max_render_particles",
        type=int,
        default=60000,
        help="Subsample particles when scene is very large",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="", help="Directory for rendered PNG frames")
    parser.add_argument("--video_path", default="", help="Output mp4 path; ffmpeg is used if available")
    parser.add_argument(
        "--include_solids",
        action="store_true",
        help="Render solid/boundary particles together with fluid particles",
    )
    parser.add_argument(
        "--invisible_objects",
        default="",
        help="Comma-separated object IDs to hide, e.g. 1,2",
    )
    parser.add_argument(
        "--substeps_override",
        type=int,
        default=100,
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
    background_top = np.clip(parse_vec3(args.background_top), 0.0, 1.0)
    background_bottom = np.clip(parse_vec3(args.background_bottom), 0.0, 1.0)
    absorption = np.maximum(parse_vec3(args.absorption), 0.0)

    background_image = build_background_image(
        width=args.width,
        height=args.height,
        base_color=background,
        top_color=background_top,
        bottom_color=background_bottom,
        mode=args.background_mode,
    )

    ps = ParticleSystem(config, GGUI=False)
    solver = ps.build_solver()
    solver.initialize()

    rng = np.random.default_rng(args.seed)

    for frame_idx in tqdm(range(args.frames), desc="Rendering frames"):
        for _ in range(substeps):
            solver.step()

        # Rebuild neighbor grid with latest particle positions before extracting normals.
        ps.initialize_particle_system()
        n = ps.particle_num[None]
        pos = ps.x.to_numpy()[:n].astype(np.float32)
        col = (ps.color.to_numpy()[:n].astype(np.float32) / 255.0).clip(0.0, 1.0)
        obj_id = ps.object_id.to_numpy()[:n]
        mat = ps.material.to_numpy()[:n]
        ps.compute_surface_normals(args.normal_radius_scale, args.normal_min_neighbors)
        nrm = ps.normal.to_numpy()[:n].astype(np.float32)

        if not args.include_solids:
            fluid_mask = mat == 1
            pos = pos[fluid_mask]
            col = col[fluid_mask]
            obj_id = obj_id[fluid_mask]
            nrm = nrm[fluid_mask]

        if invisible_objects:
            mask = ~np.isin(obj_id, list(invisible_objects))
            pos = pos[mask]
            col = col[mask]
            nrm = nrm[mask]

        if args.max_render_particles > 0 and pos.shape[0] > args.max_render_particles:
            sample_idx = rng.choice(pos.shape[0], size=args.max_render_particles, replace=False)
            pos = pos[sample_idx]
            col = col[sample_idx]
            nrm = nrm[sample_idx]

        frame = gaussian_splat_render(
            points_world=pos,
            colors=col,
            normals_world=nrm,
            camera_eye=camera_eye,
            camera_lookat=camera_lookat,
            camera_up=camera_up,
            fov_deg=args.fov,
            width=args.width,
            height=args.height,
            particle_radius=ps.particle_radius,
            sigma_scale=args.sigma_scale,
            alpha_scale=args.alpha_scale,
            background_image=background_image,
            near=args.near,
            min_sigma=args.min_sigma,
            max_sigma=args.max_sigma,
            max_kernel_radius=args.max_kernel_radius,
            absorption_coeff=absorption,
            refraction_strength=args.refraction_strength,
            normal_strength=args.normal_strength,
            specular_strength=args.specular_strength,
            fresnel_f0=args.fresnel_f0,
            sun_power=args.sun_power,
            opacity_gain=args.opacity_gain,
            thickness_blur_passes=args.thickness_blur_passes,
            normal_source=args.normal_source,
            particle_normal_mix=args.particle_normal_mix,
        )

        frame_file = output_dir / f"{frame_idx:06d}.png"
        ti.tools.imwrite(frame, str(frame_file))

    encoded = try_encode_video_with_ffmpeg(output_dir, args.fps, output_video)
    if encoded:
        print(f"Video encoded: {output_video}")
    else:
        print(f"ffmpeg not available or encoding failed; PNG frame sequence is still available in {output_dir}")
        print(f"Use ffmpeg manually, for example:\nffmpeg -y -framerate {args.fps} -i {output_dir / '%06d.png'} -pix_fmt yuv420p -vcodec libx264 {output_video}")


if __name__ == "__main__":
    main()
