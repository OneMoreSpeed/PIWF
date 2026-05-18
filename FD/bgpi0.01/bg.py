import numpy as np
import time
from scipy.io import loadmat


def gen_testdata():
    data = np.load("./Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    # print(xx.shape)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    # print(X.shape)
    y = exact.flatten()[:, None]
    # print(y.shape)
    return X, y


data_x_t, data_u = gen_testdata()
data_u = data_u.astype(np.float32)
true_u_js = data_u.reshape(100, 256)


def compute_burgers(output_shape=None):
    start_time = time.time()

    # Burgers方程参数
    nu = 0.01 / np.pi  # 粘度系数

    Nx = 1000
    Nt = 2000
    L = 2.0  # x从-1到1
    T = 1.0

    x = np.linspace(-1.0, 1.0, Nx)
    t = np.linspace(0.0, T, Nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # 稳定性检查 (CFL条件)
    cfl = np.abs(1.0) * dt / dx  # 最大速度估计为1.0
    diff_cfl = nu * dt / dx ** 2

    if cfl > 1.0:
        print(f"警告: 对流CFL数 = {cfl:.3f} > 1.0")
    if diff_cfl > 0.5:
        print(f"警告: 扩散CFL数 = {diff_cfl:.3f} > 0.5")

    u = np.zeros((Nt, Nx))

    # 初始条件: u(x, 0) = -sin(πx)
    u[0, :] = -np.sin(np.pi * x)

    # 边界条件: u(-1, t) = u(1, t) = 0
    u[:, 0] = 0.0  # x = -1
    u[:, -1] = 0.0  # x = 1

    # Burgers方程求解: u_t + u*u_x = nu * u_xx
    for n in range(0, Nt - 1):
        for i in range(1, Nx - 1):
            # 中心差分计算空间导数
            u_xx = (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1]) / dx ** 2
            u_x = (u[n, i + 1] - u[n, i - 1]) / (2 * dx)

            # Burgers方程的时间推进
            u[n + 1, i] = u[n, i] + dt * (nu * u_xx - u[n, i] * u_x)

        # 确保边界条件
        u[n + 1, 0] = 0.0  # x = -1
        u[n + 1, -1] = 0.0  # x = 1

    # 如果指定了输出形状，进行均匀采样
    if output_shape is not None:
        Nt_out, Nx_out = output_shape

        # 生成采样索引（使用np.linspace确保均匀分布）
        t_indices = np.linspace(0, Nt - 1, Nt_out, dtype=int)
        x_indices = np.linspace(0, Nx - 1, Nx_out, dtype=int)

        u_sampled = u[t_indices][:, x_indices]

    computation_time = time.time() - start_time
    print(f"计算完成，耗时: {computation_time:.3f} 秒")
    print(f"解的形状: {u.shape} (时间步数={u.shape[0]}, 空间点数={u.shape[1]})")

    np.save('burgers_solution.npy', u_sampled)
    print("结果已保存到 'burgers_solution.npy'")

    return u_sampled


if __name__ == "__main__":
    # 示例：输出指定大小 (101, 201)
    u2 = compute_burgers(output_shape=(100, 256))
    print(u2.shape)

    # 注意：这里使用原来的Allen-Cahn数据做对比可能不合适
    # 因为Burgers方程的解不同
    res_u = ((((u2 - true_u_js) ** 2) ** (1 / 2)) / (np.sqrt(np.mean(np.square(true_u_js))))) * 100
    np.save('burgers_solution_res.npy', res_u)