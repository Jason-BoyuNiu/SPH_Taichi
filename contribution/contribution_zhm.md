## 基于物理的折射与真实表面法线

### 1）SPH-Taichi 转换到 3DGS 存在的不足

我们观察到 SPH-Taichi 转换到 3DGS 的表面十分粗糙，并且透明度较差，具体分析原因如下：

| 缺陷 | baseline的问题 |
|------|---------------|
| **折射是 UV 偏移 hack** | 仅根据法线的 x/y 分量线性偏移背景图采样坐标：`sample_u = xs + normal.x * offset`。这完全没有 Snell 定律的物理依据，折射角不依赖入射角和 IOR，只是一种视觉上的近似扭曲 |
| **法线估计采用自创的二次衰减** | `compute_surface_normals` 中使用了 `q = 1 - r/h`、权重为 `q²`，没有 SPH 核函数理论的支撑，导致法线噪声大、缺乏物理正确性 |
| **z-buffer 索引遗漏** | `z = z[valid]` 之后遗漏了 `z = z[in_view]`，导致 `mean_depth` 使用错误的深度值，进而破坏屏幕空间法线计算 |
| **粒子噪声直接暴露在法线上** | 累积法线未经平滑就直接用于折射和镜面反射，颗粒感严重 |

### 2）改进方法（基于的数学原理）

#### 2.1 折射：从 UV 偏移 -> Snell 定律角偏转

旧版只是将法线 $x/y$ 分量乘以一个全局系数偏移 UV：

$$\text{sample\_u} = x_s + n_x \cdot \text{refraction\_strength} \cdot \min(W, H)$$

新版实现了完整的 Snell 折射公式（`snell_refract`，`run_simulation_gaussian.py:49-55`）：

$$\sin^2\theta_t = \eta^2 (1 - \cos^2\theta_i), \quad \eta = \frac{n_{\text{入射}}}{n_{\text{透射}}}$$

折射方向：

$$\mathbf{r} = \eta \cdot \mathbf{v}_{\text{view}} + (\eta\cos\theta_i - \cos\theta_t) \cdot \mathbf{n}$$

然后基于"假想的背景深度平面"将角度偏转转换回像素偏移：

$$\Delta u = f \cdot \left(\frac{r_x}{r_z} - \frac{v_x}{v_z}\right) \cdot d_{\text{bg}} \cdot \alpha_{\text{fluid}}$$

$$\Delta v = f \cdot \left(\frac{r_y}{r_z} - \frac{v_y}{v_z}\right) \cdot d_{\text{bg}} \cdot \alpha_{\text{fluid}}$$

其中 $f = \frac{H}{2\tan(\text{fov}/2)}$ 是焦距，$d_{\text{bg}}$ 是估计的背景深度，$\alpha_{\text{fluid}} = 1 - e^{-\tau}$ 用累积厚度调制折射强度。

#### 2.2 色散（Chromatic Dispersion）

对 R、G、B 三个通道分别使用不同的 IOR 进行 Snell 折射计算：

$$n_R = n_{\text{ior}} - \Delta n_{\text{disp}} \quad n_G = n_{\text{ior}} \quad n_B = n_{\text{ior}} + \Delta n_{\text{disp}}$$

然后重组每个通道从其对应 IOR 采样到的颜色。这在物理上对应于光在水中不同波长的折射率差异（$n_{\text{红}} < n_{\text{绿}} < n_{\text{蓝}}$），产生边缘色散效果。

#### 2.3 法线估计：从自创二次衰减 -> Müller 色彩场梯度

baseline 使用的权重函数：

$$w(q) = q^2, \quad q = 1 - r/h$$

新版使用 cubic spline 核函数的导数（Müller 色彩场梯度法）：

$$q = \frac{r}{h}, \quad \nabla W(q) = q(3q - 4)$$

法线方向由色彩场梯度的负方向给出：

$$\mathbf{n} \propto -\sum_j \frac{\mathbf{r}}{\|\mathbf{r}\|} \cdot \nabla W\left(\frac{\|\mathbf{r}\|}{h}\right)$$

cubic spline 核导数在 $q = 2/3$ 处达到峰值，在 $q=0$（中心）和 $q=1$（边界）处均为零，因此能正确强调自由表面附近的粒子并抑制内部粒子的贡献——这正是颜色场梯度方法区分表面与内部的物理直觉。

#### 2.4 法线平滑

在累积法线图上做 mask-weighted box blur ：

$$\mathbf{n}_{\text{smooth}} = \text{Blur}_{k}(\mathbf{n}_{\text{accum}} \cdot \text{mask})$$

mask 为 `tau > 1e-6`（有流体覆盖的像素），避免背景区域的零值污染前景法线。

#### 2.5 z-buffer 修复

修复了 `z = z[in_view]` 的遗漏——此前 `z` 在 `valid` 过滤后未随 `in_view` 二次过滤，导致 $z$ 数组长度与 $u, v, \sigma$ 不一致，后续依赖正确 `z` 的 `mean_depth` 计算（进而屏幕空间法线梯度）完全错误。

### 3）改进结果

| 改进项 | 视觉效果提升 |
|--------|-------------|
| **Snell 折射** | 折射角随视角变化呈现真实的入射角依赖性（浅视角时更强），不再是对所有像素统一的平移量 |
| **色散** | 水面的高光边缘出现 R/G/B 分离的彩色条纹（chromatic fringing），物理上正确 |
| **Müller 法线** | 表面法线在自由表面附近更平滑，颗粒噪声大幅减少，法线方向与物理表面一致 |
| **法线平滑** | 粒子边界处的离散脉冲噪声被 box blur 消除，液态表面呈现连续的视觉外观 |
| **z-buffer 修复** | 屏幕空间法线不再因错误的深度值产生伪影，`normal_source=screen` 和 `hybrid` 模式可以正常工作 |

预渲染的结果视频 fixed_checker.mp4 (baseline)、loosen_checker.mp4 (improved)、loosen_studio.mp4 (improved + lighting) 展示了改进后的效果，可通过以下命令复现：

```bash
python run_simulation_gaussian.py --scene_file ./data/scenes/dragon_bath_dfsph.json \
    --frames 240 --fps 30 --width 1280 --height 720 \
    --sigma_scale 1.6 --alpha_scale 0.75 --ior 1.333 --ior_dispersion 0.004 \
    --normal_source hybrid --background_mode checker
```

结果如下:
- fixed_checker.mp4  
<video width="640" height="480" controls>
  <source src="../results/rotated_fixed_checker.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

- loosen_checker.mp4  
<video width="640" height="480" controls>
  <source src="../results/rotated_loosen_checker.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

- loosen_studio.mp4  
<video width="640" height="480" controls>
  <source src="../results/rotated_loosen_studio.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### 4）超高透明度与平滑表面高光优化 (进阶渲染修正)

在后续调试与超高透明度测试中，我们进一步修复了以下两个影响渲染真实感的问题：

#### 1. 动能染色覆盖问题（导致物理透明度失效）
**问题**：即使调低了厚度累积系数（`alpha_scale`），水流依然呈现不透明的白色和红色。深入代码发现 `kinetic_to_white_red_colors` 函数会强制根据动能（velocity）将粒子染色，直接覆盖了水的本色属性，使得背景折射和透明度大打折扣。
**修正**：在渲染流程中引入了屏蔽机制，当 `--kinetic_percentile < 0.0`（如设定为 `-1.0`）时，直接跳过动能染色步骤，保留流体的原始基础物理色（baseline buffer color），从而彻底恢复了纯粹基于 Beer-Lambert 吸收与 Snell 折射定律的真实水体透明质感。

#### 2. 高光反射的"方块化/锯齿"边界伪影
**问题**：在实现高透明水体（降低 `opacity_gain`）并增强表面高光反射时，水滴和流体边缘呈现出严重的锯齿状“方块”伪影（Cubic artifacts）。此时高光项（specular）被直接乘以一个强硬的二进制掩码 `mask.astype(np.float32)`，这破坏了高斯溅射本该平滑的边界衰减。
**修正**：我们从物理层面解耦了“体积不透明度（opacity）”与“表面高光层”。纯净的水即使内部完全透明，其表面也依然应当产生明亮的镜面反射。为了消除硬阴影带来的锯齿伪影，我们引入了随厚度快速但连续变化的平滑高光掩码 `surface_alpha = 1.0 - np.exp(-tau * 20.0)`。使用这个平滑的 Alpha 层来调制高光，使得水体反光能够在极细微的飞沫和边界平滑过渡，完美消除了二值 Mask 切割造成的方块化现象。

修正前/后的结果对比如下:
- Before fix: 
<video width="640" height="480" controls>
  <source src="../results/basin/tune_ultra/L1p8_tw0p12_v3p2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

- After fix: 
<video width="640" height="480" controls>
  <source src="../results/basin/tune_fix/L1p8_tw0p12_v3p2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>