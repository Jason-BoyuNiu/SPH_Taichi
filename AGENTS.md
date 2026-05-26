# Agent Instructions for SPH Taichi

This project is a high-performance Smooth Particle Hydrodynamics (SPH) simulator — a fork of [erizmr/SPH_Taichi](https://github.com/erizmr/SPH_Taichi) — extended by a student team for the "工程实践与科技创新" (Engineering Practice & Tech Innovation) course. Written in Python using the [Taichi](https://github.com/taichi-dev/taichi) GPU computing library.

## Team Contributions (vs upstream master)

The following features were added on top of the upstream WCSPH rigid-body simulator:

1. **DFSPH Solver** (`DFSPH.py`, simulationMethod=4) — Divergence-Free SPH with Jacobi-style iterative divergence solve + pressure solve, surface tension, and rigid-body two-way coupling via Akinci2012 boundary handling.
2. **Gaussian Splat Rendering** (`run_simulation_gaussian.py`) — Offline screen-space Gaussian splatting pipeline with physically-based Snell refraction (chromatic dispersion), Fresnel specular, Beer-Lambert absorption, and procedural environment maps. Replaces the real-time GGUI viewer.
3. **Surface Normal Estimation** (`particle_system.py:compute_surface_normals`) — GPU kernel computing per-particle normals via Müller's colour-field gradient (cubic spline kernel derivative weighting). Used by the Gaussian renderer for refraction/specular.
4. **Custom Prefix Sum** (`scan_single_buffer.py`) — Warp-level CUDA inclusive scan for neighbor-search counting sort (replaces Taichi built-in for performance on CUDA backend).
5. **New Scene Files** — DFSPH variants: `dragon_bath_dfsph.json`, `armadillo_bath_dynamic_dfsph.json`, `dragon_bath_dynamic_dfsph.json`, `high_fluid_dfsph.json`.

Branch layout:
- `dev_zhm` — Main integration branch (HEAD): Gaussian rendering, DFSPH, physically-based refraction, surface normals
- `dev_nby` — Energy adjustments branch
- `origin/cxy_dev` — Fluid reflection updates (merged via PR #1)
- `master` — Upstream snapshot

## Project Architecture

1. **Configuration (`config_builder.py`)**  
   `SimConfig` parses JSON scenes from `data/scenes/`. Keys: `Configuration` (domain, solver, timestep, etc.), `FluidBlocks`, `RigidBlocks`, `RigidBodies`. Solver method IDs: `0` = WCSPH, `4` = DFSPH.

2. **Particle System (`particle_system.py`)**  
   `ParticleSystem` owns all GPU fields (`x`, `v`, `density`, `pressure`, `material`, `color`, `is_dynamic`, `normal`, plus DFSPH-specific `dfsph_factor`/`density_adv`). Handles memory allocation, particle initialization from JSON blocks/meshes, and neighbor search.
   - **Neighbor Search:** Grid-based spatial hash → GPU prefix sum (`PrefixSumExecutor` or custom `scan_single_buffer`) → counting sort. Particles are physically reordered in memory for cache coherence.
   - **Buffer Convention:** Every sortable field has a corresponding `_buffer` mirror. When adding new fields, add both the field and its buffer, and wire them into `counting_sort()`.
   - **Material IDs:** `0` = solid (boundary/rigid), `1` = fluid. Checked via `is_static_rigid_body(p)` / `is_dynamic_rigid_body(p)`.

3. **Physics Solvers** → All inherit from `SPHBase` (`sph_base.py`):
   - `WCSPH.py` (Method 0): Tait equation of state, explicit pressure forces.
   - `DFSPH.py` (Method 4): Iterative divergence solve (velocity correction) + pressure solve (density invariance). Key kernel: `compute_DFSPH_factor` precomputes the Jacobi stiffness denominator.
   - `IISPH.py`: Implicit IISPH variant (uses older direct-neighbor-list approach; **may not be fully integrated** with the current grid-based neighbor search).

4. **Rendering (`run_simulation_gaussian.py`)** → See detailed explanation: [`run_simulation_gaussian_explanation.md`](./run_simulation_gaussian_explanation.md) (Chinese). Key tunables: `--sigma_scale`, `--alpha_scale`, `--ior`, `--ior_dispersion`, `--normal_source` (particle/screen/hybrid), `--background_mode` (plain/studio/checker).

5. **Standalone Scripts:**
   - `demo_high_fluid.py` — Standalone WCSPH/IISPH demo (does NOT use JSON config system; hardcodes domain and particles).
   - `run_simulation.py` — Original real-time GGUI viewer (uses JSON configs).

## Taichi Conventions

- `@ti.data_oriented` on any class containing kernels.
- `@ti.kernel` for GPU entry points; `@ti.func` for device-only helpers.
- **No dynamic field resizing** during the simulation loop — Taichi compiles AOT for the data layout.
- `for_all_neighbors(p_i, task, ret)` iterates over the 27-cell neighborhood. `ret` is a mutable scalar/vector/struct passed by template — this is the core hot path; **minimize branching inside `task`**.
- Use `ti.static()` for compile-time conditionals (e.g., `ti.static(self.simulation_method == 4)` to guard DFSPH-specific buffer swaps).

## Running

```bash
pip install -r requirements.txt   # taichi>=1.2.0, trimesh, tqdm

# Real-time GGUI:
python run_simulation.py --scene_file ./data/scenes/dragon_bath.json

# Offline Gaussian splat rendering:
python run_simulation_gaussian.py --scene_file ./data/scenes/dragon_bath_dfsph.json \
    --frames 240 --fps 30 --width 1280 --height 720 \
    --sigma_scale 1.6 --alpha_scale 0.75 --ior 1.333 --background_mode studio
```

## Common Pitfalls

- **CUDA backend only** for prefix sum (`scan_single_buffer.py` is CUDA-only; `PrefixSumExecutor` works on CUDA/Vulkan). Metal/CPU backends will fail or degrade.
- **Counting sort buffer sync:** When adding a new particle property that needs correct neighbor values after sorting, you MUST add it to ALL of: field declaration, buffer declaration, swap in `counting_sort()`. Otherwise neighbors will read stale data.
- IISPH uses a different neighbor storage (`fluid_neighbors_num` / `solid_neighbors_num` arrays) and may not work with the current grid-based `for_all_neighbors` iterator.
- Rigid-body coupling uses shape-matching (polar decomposition). `is_dynamic=0` = static boundary, `is_dynamic=1` = moving rigid body.
- The `results/` directory contains pre-rendered MP4 outputs — these are binary artifacts, not source.