# run_simulation_gaussian.py 详细解释

`run_simulation_gaussian.py` 的核心作用是：先跑 SPH 物理仿真，再把每一帧粒子当作“3D 点 + 屏幕空间 2D 高斯核”去渲染成 PNG，最后尝试用 `ffmpeg` 合成 MP4。  
它实现的是“粒子高斯泼溅渲染”，不是训练型/神经渲染那套完整 3DGS 管线。

关键代码都在这里：`run_simulation_gaussian.py`

## 文件结构

1. 初始化 Taichi GPU: `run_simulation_gaussian.py:14`
2. 相机与坐标变换工具函数: `run_simulation_gaussian.py:17` 到 `run_simulation_gaussian.py:39`
3. 高斯泼溅渲染核心 `gaussian_splat_render`: `run_simulation_gaussian.py:42`
4. 可选 `ffmpeg` 编码: `run_simulation_gaussian.py:129`
5. 主循环 `main`: `run_simulation_gaussian.py:155`

## 每帧是怎么跑的

1. 读取场景配置并创建粒子系统/求解器: `run_simulation_gaussian.py:198` 到 `run_simulation_gaussian.py:218`
2. 每输出一帧之前先做 `substeps` 次 SPH 更新: `run_simulation_gaussian.py:222` 到 `run_simulation_gaussian.py:224`
3. 从 Taichi 字段拷出粒子位置/颜色/物体 ID: `run_simulation_gaussian.py:226` 到 `run_simulation_gaussian.py:229`
4. 可选隐藏物体、可选随机下采样粒子数: `run_simulation_gaussian.py:231` 到 `run_simulation_gaussian.py:239`
5. 调 `gaussian_splat_render` 生成 RGB 帧: `run_simulation_gaussian.py:241`
6. 写 PNG 序列: `run_simulation_gaussian.py:260`
7. 全部结束后尝试编码 MP4: `run_simulation_gaussian.py:264`

## 高斯泼溅到底怎么完成

核心在 `run_simulation_gaussian.py:42` 到 `run_simulation_gaussian.py:126`。

### 1. 世界坐标转相机坐标

代码先构建 look-at 基向量 `right/up/forward`，再做线性变换：`run_simulation_gaussian.py:31` 到 `run_simulation_gaussian.py:39`

### 2. 近裁剪 + 透视投影

只保留 `z > near` 的点：`run_simulation_gaussian.py:63` 到 `run_simulation_gaussian.py:70`  
投影到像素：

```text
f = 0.5 * H / tan(fov/2)
u = f * x/z + W/2
v = H/2 - f * y/z
```

对应 `run_simulation_gaussian.py:72` 到 `run_simulation_gaussian.py:74`

### 3. 把“粒子半径”变成“屏幕高斯 sigma”

```text
sigma_px = clip(f * particle_radius / z * sigma_scale, min_sigma, max_sigma)
radius_px = min(ceil(3*sigma), max_kernel_radius)
```

对应 `run_simulation_gaussian.py:76` 到 `run_simulation_gaussian.py:77`  
这里的 `particle_radius` 来自粒子系统配置：`particle_system.py:33` 到 `particle_system.py:34`

### 4. 对每个粒子在局部像素块“泼”2D 高斯

每个粒子只在 `cx±r, cy±r` 的小窗口内计算：

```text
g = exp(-(dx^2 + dy^2)/(2*sigma^2))
a = alpha_scale * g
accum_alpha += a
accum_rgb += a * color_i
```

对应 `run_simulation_gaussian.py:97` 到 `run_simulation_gaussian.py:118`

### 5. 颜色与透明度合成

```text
rgb = accum_rgb / max(accum_alpha, 1e-6)
opacity = 1 - exp(-accum_alpha)
frame = rgb*opacity + bg*(1-opacity)
```

对应 `run_simulation_gaussian.py:120` 到 `run_simulation_gaussian.py:123`  
最后做 `gamma=2.2` 矫正：`run_simulation_gaussian.py:125`

## 这版“高斯泼溅”的特点

1. 是各向同性高斯（`sigma` 单值），不是每点完整 3x3 协方差的 anisotropic splat。
2. 没有按深度排序的 alpha compositing；更像体积密度累积（`opacity = 1-exp(-A)`）。
3. 颜色直接取粒子颜色字段（`ps.color`）并归一化：`run_simulation_gaussian.py:228`
4. 重点是把 SPH 粒子渲染得更连续柔和，减少“点状噪声”。

## 你最该调的参数

1. `sigma_scale`：控制“糊化/连通”程度，越大越粘连。
2. `alpha_scale`：控制每个 splat 的贡献强度，越大越厚重。
3. `min_sigma/max_sigma/max_kernel_radius`：控制近远处核大小上下限和性能。
4. `max_render_particles`：控制速度与细节的平衡。
