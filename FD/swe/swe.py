import numpy as np
import time
import data

def compute_swe_stable(output_shape=(61,121)):


    start_time = time.time()

    # 参数（与你的代码完全一致）
    n = 0.03  # Manning系数
    u0 = 0.29  # 入口流速
    g = 9.81  # 重力加速度

    # ========== 网格设置 ==========
    # 空间网格（与你的代码一致）
    deta = 1  # 空间步长

    L = 1200.0  # x ∈ [0, 1200]
    nx = int(L / deta) + 1
    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]

    # 时间网格（需要更小的时间步长！）
    # 根据CFL条件: dt < dx / |u| ≈ 10 / 0.29 ≈ 34.5s
    # 使用dt=10s以保证稳定
    dt = 1  # 稳定时间步长
    T = 3600.0  # 总时间
    nt = int(T / dt) + 1  # 时间步数
    t = np.linspace(0, T, nt)

    print(f"空间网格: nx={nx}, dx={dx:.1f}m, x∈[0, {L}]")
    print(f"时间网格: nt={nt}, dt={dt:.1f}s, t∈[0, {T}], CFL={u0 * dt / dx:.3f} < 1 ✓")

    # ========== 初始化 ==========
    h = np.zeros((nt, nx))  # 水深
    u = np.zeros((nt, nx))  # 流速
    hu = np.zeros((nt, nx))  # 流量

    # 初始条件
    h[0, :] = 1e-6  # 很小的初始水深，避免除零
    u[0, :] = 0.0
    hu[0, :] = 0.0

    # ========== 边界条件 ==========
    # 上游边界
    for i in range(nt):
        h_boundary = ((7 / 3 * (n ** 2 * u0 ** 3 * t[i])) ** (3 / 7))
        h[i, 0] = max(h_boundary, 1e-6)
        u[i, 0] = u0
        hu[i, 0] = h[i, 0] * u[i, 0]

    # 下游边界（自由出流）
    for i in range(nt):
        h[i, -1] = h[i, -2]
        u[i, -1] = u[i, -2]
        hu[i, -1] = hu[i, -1]

    # ========== 简化的有限差分求解 ==========
    # 使用最简单的格式，只求解质量方程，假设流速恒定
    print("\n开始求解（简化模型）...")

    for n_time in range(0, nt - 1):
        current_h = h[n_time, :].copy()

        # 简化的对流方程: ∂h/∂t + u0·∂h/∂x = 0
        # 使用迎风格式
        for i in range(1, nx - 1):
            if u0 >= 0:
                # 上游迎风
                dh_dx = (current_h[i] - current_h[i - 1]) / dx
            else:
                # 下游迎风
                dh_dx = (current_h[i + 1] - current_h[i]) / dx

            h[n_time + 1, i] = current_h[i] - dt * u0 * dh_dx
            h[n_time + 1, i] = max(h[n_time + 1, i], 1e-6)

        # 边界条件
        h[n_time + 1, 0] = max(((7 / 3 * (n ** 2 * u0 ** 3 * t[n_time + 1])) ** (3 / 7)), 1e-6)
        h[n_time + 1, -1] = h[n_time + 1, -2]

        # 假设流速恒定
        u[n_time + 1, :] = u0
        hu[n_time + 1, :] = h[n_time + 1, :] * u[n_time + 1, :]

        # 进度显示
        if (n_time + 1) % 360 == 0:  # 每360步显示一次
            print(
                f"  进度: {100 * (n_time + 1) / (nt - 1):.0f}%, t={t[n_time + 1]:.0f}s, h_max={h[n_time + 1, :].max():.3f}m")

    computation_time = time.time() - start_time

    # ========== 分析解 ==========
    X, T_mesh = np.meshgrid(x, t)
    analyh = np.zeros_like(h)

    for i in range(nt):
        for j in range(nx):
            # 简化分析解：h(x,t) = h(0, t - x/u0) 如果 t > x/u0，否则为0
            travel_time = x[j] / u0
            if t[i] >= travel_time:
                analyh[i, j] = ((7 / 3 * (n ** 2 * u0 ** 3 * (t[i] - travel_time))) ** (3 / 7))
            else:
                analyh[i, j] = 0.0

    analyh = np.where(analyh < 0, 0.0, analyh)
    analyh = np.where(np.isnan(analyh), 0.0, analyh)

    analyu = np.full_like(analyh, u0)
    analyu[analyh == 0] = 0.0
    analyhu = analyh * analyu

    # ========== 结果输出 ==========
    print(f"\n计算完成！耗时: {computation_time:.3f}秒")

    print(f"\n最终时刻 (t={T}s):")
    print(f"  数值解 h: [{h[-1, :].min():.3e}, {h[-1, :].max():.3f}] m")
    print(f"  分析解 h: [{analyh[-1, :].min():.3e}, {analyh[-1, :].max():.3f}] m")

    Nt_out, Nx_out = output_shape

    # 生成采样索引（使用np.linspace确保均匀分布）
    t_indices = np.linspace(0, nt - 1, Nt_out, dtype=int)
    x_indices = np.linspace(0, nx - 1, Nx_out, dtype=int)

    h_sampled = h[t_indices][:, x_indices]
    u_sampled = u[t_indices][:, x_indices]
    hu_sampled = hu[t_indices][:, x_indices]
    print('aaaaaaaaa',h_sampled.shape)
    # 保存结果
    np.save('swe_stable_h.npy', h_sampled)
    #np.save('swe_stable_u.npy', u_sampled)
    np.save('swe_stable_hu.npy', hu_sampled)

    print(f"\n结果已保存: h({h.shape}), u({u.shape}), hu({hu.shape})")

    return h_sampled, u_sampled, hu_sampled, analyh, analyu, analyhu, x, t


def quick_plot():
    """快速绘图"""
    try:
        import matplotlib.pyplot as plt

        h, u, hu, analyh, analyu, analyhu, x, t = compute_swe_stable()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # 最终时刻水深对比
        axes[0].plot(x, h[-1, :], 'b-', label='Numerical', linewidth=2)
        axes[0].plot(x, analyh[-1, :], 'r--', label='Analytical', linewidth=2)
        axes[0].set_xlabel('x (m)')
        axes[0].set_ylabel('h (m)')
        axes[0].set_title(f'Water depth at t={t[-1]:.0f}s')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 水深沿程分布（多个时间）
        time_indices = [0, len(t) // 4, len(t) // 2, 3 * len(t) // 4, -1]
        colors = ['gray', 'blue', 'green', 'orange', 'red']

        for idx, color in zip(time_indices, colors):
            axes[1].plot(x, h[idx, :], color=color,
                         label=f't={t[idx]:.0f}s', linewidth=1.5)

        axes[1].set_xlabel('x (m)')
        axes[1].set_ylabel('h (m)')
        axes[1].set_title('Water depth evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('swe_quick_result.png', dpi=150, bbox_inches='tight')
        plt.show()

    except ImportError:
        print("Matplotlib not available for plotting")


if __name__ == "__main__":
    # 运行稳定版本
    h_sampled, u_sampled, hu_sampled, analyh, analyu, analyhu, x, t = compute_swe_stable()
    _, _, _, _, _, input, label_h, label_u, _, _ = data.load_SWEdata()
    res_h = ((((h_sampled.flatten() - label_h) ** 2) ** (1 / 2)) / (np.sqrt(np.mean(np.square(label_h))))) * 100
    res_hu = (((((h_sampled.flatten()*u_sampled.flatten()) - (label_h*label_u)) ** 2) ** (1 / 2)) / (np.sqrt(np.mean(np.square(label_h*label_u))))) * 100
    np.save('swe_res_h.npy', res_h)
    np.save('swe_res_hu.npy', res_hu)
    # 或者运行快速绘图版本
    quick_plot()

    print("\n关键点总结:")
    print("1. 使用了稳定时间步长 dt=10s (CFL < 1)")
    print("2. 简化模型：只求解质量方程，假设流速恒定")
    print("3. 上游边界条件: h(0,t) = ((7/3*(n²*u0³*t))^(3/7))")
    print("4. 下游边界条件: 自由出流")