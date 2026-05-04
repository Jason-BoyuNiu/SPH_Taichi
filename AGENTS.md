# Agent Instructions for SPH Taichi

This project is a high-performance Smooth Particle Hydrodynamics (SPH) simulator written in Python using the [Taichi](https://github.com/taichi-dev/taichi) library. It implements Weakly Compressible SPH (WCSPH), Divergence-free SPH (DFSPH), Implicit Incompressible SPH (IISPH), and solid-fluid coupling.

## Project Architecture

1. **Configuration Layer (`config_builder.py`)**  
   Parses JSON scene files (located in `data/scenes/`) to build `SimConfig`. Converts scene properties (domain bounds, solvers, bodies, fluids) into Python configurations used at initialize.

2. **Data & Memory (`particle_system.py`)**  
   A centralized `ParticleSystem` handles memory allocation for all fields (positions `x`, velocities `v`, `density`, `pressure`, `material`, etc.) across fluids and rigid bodies.
   - **Neighbor Search**: Implements a grid-based spatial partitioning mechanism accelerated by GPU parallel prefix sum and counting sort to update neighbors efficiently at each substep.

3. **Physics Solvers**  
   All solvers inherit from `SPHBase` (`sph_base.py`), providing neighbor iterators (`for_all_neighbors`), shape-matching for dynamic rigid bodies, viscosity, and density calculations.
   - `WCSPH.py` (Method 0): Uses an equation of state (Tait equation) to compute pressure.
   - `DFSPH.py` (Method 4): Solves incompressibility iteratively with a divergence solve and a pressure solve.
   - `IISPH.py`: Another implicit solver variant.

4. **Gaussian Splat Rendering (`run_simulation_gaussian.py`)**  
   An offline rendering pipeline that replaces the real-time GUI display with a 2D screen-space Gaussian splatting approach.
   - **Method:** Projects SPH particles as isotropic 2D Gaussians. Uses volumetric density accumulation (`opacity = 1 - exp(-alpha)`) rather than depth-sorted alpha compositing to render fluids continuously.
   - **Outputs:** Renders a sequence of PNG frames and compiles them into an MP4 video via `ffmpeg`.
   - **Key Parameters:** Tune `sigma_scale` (splat connectivity/blur), `alpha_scale` (splat intensity), and particle filtering (`max_render_particles`) directly in the script.

## Taichi Conventions and Guidelines

- **Taichi Decorators:**
  - Decorate any class that contains kernels with `@ti.data_oriented`.
  - Use `@ti.kernel` for GPU entry points called from the Python host code.
  - Use `@ti.func` for helper functions executed entirely on the device (GPU).
- **GPU Memory Management:**
  - Taichi compiles kernels AOT for the data layout. Avoid dynamically resizing particle counts or re-allocating fields during the main simulation loop.
  - When adding new particle properties, remember that the neighbor search sorts particles physically in memory (via counting sort buffers) to maximize cache coherence. Add mirrors for new fields if they need synchronous sorting during advection.
- **Material IDs:** `0` identifies solid particles (boundaries and rigid bodies), `1` identifies fluid particles.
- **Performance:** `for_all_neighbors` is the core computational bottleneck. Avoid branching inside the neighbor loop.

## Running and Testing

- **Install:** `python -m pip install -r requirements.txt` (requires `taichi>=1.2.0`, `trimesh`, `tqdm`).
- **Run simulation:** `python run_simulation.py --scene_file ./data/scenes/dragon_bath.json`
- Use the GG GUI parameters or JSON variables to modify visuals. JSON fields dictate variables like `numberOfStepsPerRenderUpdate` and `timeStepSize`.

## Common Pitfalls
- **Backend Limitations:** The spatial hashing depends on Taichi's scan (prefix sum), which historically works best on the `cuda` and `vulkan` backends. macOS (Metal) or CPU backends may run into limitations or degraded performance.
- Rigid-body coupling relies on a shape-matching technique (polar decomposition) on particles assigned to rigid bodies. Be careful mixing fixed boundary particles (`is_dynamic = 0`) with floating bodies (`is_dynamic = 1`).