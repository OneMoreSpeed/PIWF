import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
from scipy.interpolate import griddata
import string

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fontSize = 16
config = {
    "text.usetex": False,
    "font.family": "Times New Roman",
    "font.size": fontSize,
    "mathtext.fontset": 'cm',
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.2,
}
mpl.rcParams.update(config)

###################################################################################
##### The property of drawing graphs #####
# 颜色和线型定义 - 按新顺序: DNS, PINN, PIKAN, PIWT
colors = ['k', 'b', 'C1', 'r']  # 黑色(DNS)，蓝色(PINN)，橙色(PIKAN)，红色(PIWT)
linestyles = ['-', ':', (0, (5, 5)), '--']  # 实线，点线，短划线，虚线
markers = ['', 's', 'd', 'o']  # 无标记，方框，菱形，圆圈
line_labels = ['DNS', 'PINN', 'PIKAN', 'PIWT']
line_widths = [2.0, 2.5, 2.5, 2.5]


###################################################################################

def compute_velocity_gradients(x, y, u, v):
    """
    计算速度梯度: du/dx, du/dy, dv/dx, dv/dy
    使用中心差分法计算导数
    """
    # 创建结构化网格
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    # 确保网格是单调的
    x_unique.sort()
    y_unique.sort()

    # 重塑为网格
    X_grid, Y_grid = np.meshgrid(x_unique, y_unique, indexing='ij')
    U_grid = np.zeros_like(X_grid)
    V_grid = np.zeros_like(X_grid)

    # 插值到网格
    U_grid = griddata(np.c_[x, y], u, (X_grid, Y_grid), method='linear', fill_value=np.nan)
    V_grid = griddata(np.c_[x, y], v, (X_grid, Y_grid), method='linear', fill_value=np.nan)

    # 计算导数
    dx = x_unique[1] - x_unique[0]
    dy = y_unique[1] - y_unique[0]

    # 速度梯度
    dudx = np.gradient(U_grid, dx, axis=0)
    dudy = np.gradient(U_grid, dy, axis=1)
    dvdx = np.gradient(V_grid, dx, axis=0)
    dvdy = np.gradient(V_grid, dy, axis=1)

    # 将网格数据转换回散点形式
    mask = ~(np.isnan(U_grid) | np.isnan(V_grid))
    mask_flat = mask.flatten()

    return (X_grid.flatten()[mask_flat], Y_grid.flatten()[mask_flat],
            dudx.flatten()[mask_flat], dudy.flatten()[mask_flat],
            dvdx.flatten()[mask_flat], dvdy.flatten()[mask_flat])


def compute_vorticity(x, y, u, v):
    """
    计算涡量: omega = dv/dx - du/dy
    """
    x_out, y_out, dudx, dudy, dvdx, dvdy = compute_velocity_gradients(x, y, u, v)
    omega = dvdx - dudy
    return x_out, y_out, omega


def compute_q_criterion(x, y, u, v):
    """
    计算Q判据: Q = 0.5 * (||Omega||^2 - ||S||^2)
    对于二维流动，Q = -0.5 * (dudx*dvdy - dudy*dvdx)
    """
    x_out, y_out, dudx, dudy, dvdx, dvdy = compute_velocity_gradients(x, y, u, v)

    # 完整的Q判据公式
    # 应变率张量 S 的范数平方
    S_norm_sq = dudx ** 2 + dvdy ** 2 + 0.5 * (dudy + dvdx) ** 2
    # 涡量张量 Omega 的范数平方
    Omega_norm_sq = 0.5 * (dvdx - dudy) ** 2

    Q = 0.5 * (Omega_norm_sq - S_norm_sq)

    return x_out, y_out, Q


def plot_combined_profiles():
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    fig_Path = os.path.join(cwd, "figure_profiles")
    g_suffix = ["pdf", "png"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)

    # 加载原始数据用于插值
    print("📂 加载原始数据...")
    data = loadmat("./cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # 重新整理数据
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T
    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x_all = XX.flatten()[:, None]  # NT x 1
    y_all = YY.flatten()[:, None]  # NT x 1
    t_all = TT.flatten()[:, None]  # NT x 1
    u_all = UU.flatten()[:, None]  # NT x 1
    v_all = VV.flatten()[:, None]  # NT x 1
    p_all = PP.flatten()[:, None]  # NT x 1

    full_data = np.concatenate([t_all, x_all, y_all, u_all, v_all, p_all], axis=1)

    # 指定时间点
    target_t = 3
    print(f"🎯 提取 t={target_t} 时刻的数据...")

    # 找到目标时间点的数据
    tolerance = 0.05  # 时间容差
    data_at_t = full_data[np.abs(full_data[:, 0] - target_t) < tolerance]

    if len(data_at_t) == 0:
        print(f"❌ 找不到 t={target_t} 时刻的数据")
        return

    X_points = data_at_t[:, 1]
    Y_points = data_at_t[:, 2]
    xy_points = np.c_[X_points, Y_points]

    # 定义要绘制的变量（只保留涡量和Q判据）
    variables = ['omega', 'q_criterion']
    var_names = [r'$\omega_z$', r'$Q$']

    # 定义x位置
    x_positions = [2.5, 3, 3.5]

    # y轴范围
    y_range = [-2, 2]
    y_query = np.linspace(y_range[0], y_range[1], 300)

    # 创建图形和子图 - 2行3列
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=False)
    plt.subplots_adjust(wspace=0.15, hspace=0.2, top=0.92, bottom=0.12, left=0.08, right=0.98)

    print("📊 开始绘制 2x3 组合图（涡量和Q判据）...")

    # 计算DNS数据的涡量和Q判据
    print("🔄 计算DNS涡量和Q判据...")
    u_true = data_at_t[:, 3]
    v_true = data_at_t[:, 4]

    # DNS涡量
    print("  计算DNS涡量...")
    x_dns_omega, y_dns_omega, omega_dns = compute_vorticity(X_points, Y_points, u_true, v_true)
    omega_dns_points = np.c_[x_dns_omega, y_dns_omega]
    print(f"    DNS涡量计算完成，有效点数: {len(omega_dns)}")

    # DNS Q判据
    print("  计算DNS Q判据...")
    x_dns_q, y_dns_q, q_dns = compute_q_criterion(X_points, Y_points, u_true, v_true)
    q_dns_points = np.c_[x_dns_q, y_dns_q]
    print(f"    DNS Q判据计算完成，有效点数: {len(q_dns)}")

    # 计算预测数据的涡量和Q判据
    print("🔄 计算预测数据的涡量和Q判据...")
    try:
        # PINN
        print("  计算PINN...")
        u_pinn = np.load(f"u_predNS_T{target_t}_pinn.npy").flatten()
        v_pinn = np.load(f"v_predNS_T{target_t}_pinn.npy").flatten()
        print(f"    PINN数据形状: u={u_pinn.shape}, v={v_pinn.shape}")

        x_pred_omega, y_pred_omega, omega_pinn = compute_vorticity(X_points, Y_points, u_pinn, v_pinn)
        x_pred_q, y_pred_q, q_pinn = compute_q_criterion(X_points, Y_points, u_pinn, v_pinn)
        print(f"    PINN涡量有效点数: {len(omega_pinn)}")

        # PIWT
        print("  计算PIWT...")
        u_piwt = np.load(f"u_predNS_T{target_t}_piwt.npy").flatten()
        v_piwt = np.load(f"v_predNS_T{target_t}_piwt.npy").flatten()
        x_pred_omega, y_pred_omega, omega_piwt = compute_vorticity(X_points, Y_points, u_piwt, v_piwt)
        x_pred_q, y_pred_q, q_piwt = compute_q_criterion(X_points, Y_points, u_piwt, v_piwt)
        print(f"    PIWT涡量有效点数: {len(omega_piwt)}")

        # PIKAN
        print("  计算PIKAN...")
        u_pikan = np.load(f"u_predNS_T{target_t}_pikan.npy").flatten()
        v_pikan = np.load(f"v_predNS_T{target_t}_pikan.npy").flatten()
        x_pred_omega, y_pred_omega, omega_pikan = compute_vorticity(X_points, Y_points, u_pikan, v_pikan)
        x_pred_q, y_pred_q, q_pikan = compute_q_criterion(X_points, Y_points, u_pikan, v_pikan)
        print(f"    PIKAN涡量有效点数: {len(omega_pikan)}")

        # 存储预测数据 - 注意这里使用正确的变量名
        pred_data = {
            'omega': {
                'pinn': (omega_pinn, x_pred_omega, y_pred_omega),
                'piwt': (omega_piwt, x_pred_omega, y_pred_omega),
                'pikan': (omega_pikan, x_pred_omega, y_pred_omega)
            },
            'q_criterion': {
                'pinn': (q_pinn, x_pred_q, y_pred_q),
                'piwt': (q_piwt, x_pred_q, y_pred_q),
                'pikan': (q_pikan, x_pred_q, y_pred_q)
            }
        }

    except Exception as e:
        print(f"❌ 加载预测数据失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 遍历每个变量 (行)
    for row_idx, (var, var_label) in enumerate(zip(variables, var_names)):
        print(f"  绘制变量: {var}")

        # 选择DNS数据
        if var == 'omega':
            dns_points = omega_dns_points
            dns_values = omega_dns
        else:  # q_criterion
            dns_points = q_dns_points
            dns_values = q_dns

        # 遍历每个x位置 (列)
        for col_idx, x_pos in enumerate(x_positions):
            ax = axes[row_idx, col_idx]

            # 创建x位置为常数的查询点
            x_query = np.full_like(y_query, x_pos)

            # 插值DNS数据
            dns_interp = griddata(dns_points, dns_values, (x_query, y_query), method='linear')

            # 插值预测数据
            pinn_points = np.c_[pred_data[var]['pinn'][1], pred_data[var]['pinn'][2]]
            piwt_points = np.c_[pred_data[var]['piwt'][1], pred_data[var]['piwt'][2]]
            pikan_points = np.c_[pred_data[var]['pikan'][1], pred_data[var]['pikan'][2]]

            pinn_interp = griddata(pinn_points, pred_data[var]['pinn'][0],
                                   (x_query, y_query), method='linear')
            piwt_interp = griddata(piwt_points, pred_data[var]['piwt'][0],
                                   (x_query, y_query), method='linear')
            pikan_interp = griddata(pikan_points, pred_data[var]['pikan'][0],
                                    (x_query, y_query), method='linear')

            # 绘制曲线
            # 1. DNS (原Ground Truth)
            ax.plot(y_query, dns_interp, color=colors[0], linestyle=linestyles[0],
                    linewidth=line_widths[0], zorder=2)

            # 2. PINN
            ax.plot(y_query, pinn_interp, color=colors[1], linestyle=linestyles[1],
                    linewidth=line_widths[1], zorder=3)
            ax.plot(y_query[::20], pinn_interp[::20], marker=markers[1], color=colors[1], linestyle='',
                    markersize=5, markerfacecolor='none', markeredgewidth=1.2, zorder=4)

            # 3. PIKAN
            ax.plot(y_query, pikan_interp, color=colors[2], linestyle=linestyles[2],
                    linewidth=line_widths[2], zorder=5)
            ax.plot(y_query[::20], pikan_interp[::20], marker=markers[2], color=colors[2], linestyle='',
                    markersize=5, markerfacecolor='none', markeredgewidth=1.2, zorder=6)

            # 4. PIWT
            ax.plot(y_query, piwt_interp, color=colors[3], linestyle=linestyles[3],
                    linewidth=line_widths[3], zorder=7)
            ax.plot(y_query[::20], piwt_interp[::20], marker=markers[3], color=colors[3], linestyle='',
                    markersize=5, markerfacecolor='none', markeredgewidth=1.2, zorder=8)

            # 设置x轴范围
            ax.set_xlim([-2, 2])

            # 设置x轴主刻度
            x_major_ticks = np.array([-2, -1, 0, 1, 2])
            ax.set_xticks(x_major_ticks)

            # 设置x轴次要刻度
            x_minor_ticks = np.arange(-2, 2.05, 0.2)
            ax.set_xticks(x_minor_ticks, minor=True)

            # 设置x轴标签 - 只保留最下面一行
            if row_idx == 1:  # 最后一行（Q判据行）
                ax.set_xlabel(r"$y$", fontsize=18)
                ax.tick_params(axis='x', which='both', labelbottom=True)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis='x', which='both', labelbottom=False)

            # 设置y轴标签 - 只保留最左侧一列
            if col_idx == 0:
                ax.set_ylabel(var_label, fontsize=18)
                ax.tick_params(axis='y', which='both', labelleft=True)
            else:
                ax.set_ylabel("")
                ax.tick_params(axis='y', which='both', labelleft=False)

            # 顶部标题 (x位置) - 只保留第一行
            if row_idx == 0:
                ax.set_title(r"$x = " + f"{x_pos:.1f}" + r"$", fontsize=18, pad=10, fontfamily='Times New Roman')

            # 设置y轴范围和刻度
            all_data = []
            for interp in [dns_interp, pinn_interp, piwt_interp, pikan_interp]:
                if interp is not None:
                    valid_data = interp[~np.isnan(interp)]
                    if len(valid_data) > 0:
                        all_data.extend(valid_data)

            if len(all_data) > 0:
                # 根据变量类型设置y轴范围
                data_min = np.min(all_data)
                data_max = np.max(all_data)
                data_range = data_max - data_min
                margin = data_range * 0.1
                ax.set_ylim([data_min - margin, data_max + margin])

                # 自动设置主刻度（5个刻度）
                y_major_ticks = np.linspace(data_min - margin, data_max + margin, 5)
                y_minor_ticks = np.linspace(data_min - margin, data_max + margin, 11)

                # 应用y轴主刻度和标签
                ax.set_yticks(y_major_ticks)

                # 格式化y轴标签
                y_tick_labels = []
                for tick in y_major_ticks:
                    if abs(tick) < 1e-6:
                        y_tick_labels.append("0")
                    else:
                        # 根据数值大小决定小数位数
                        if abs(tick) < 0.1:
                            label = f"{tick:.3f}"
                        elif abs(tick) < 1:
                            label = f"{tick:.2f}"
                        else:
                            label = f"{tick:.1f}"
                        # 去除末尾多余的0
                        if '.' in label:
                            label = label.rstrip('0').rstrip('.')
                        y_tick_labels.append(label)

                # 只在最左侧一列显示标签
                if col_idx == 0:
                    ax.set_yticklabels(y_tick_labels)
                else:
                    ax.set_yticklabels([])

                # 设置次要刻度
                ax.set_yticks(y_minor_ticks, minor=True)

            # 设置刻度参数
            ax.tick_params(axis='both', which='major', size=6, width=1.2, direction='in')
            ax.tick_params(axis='both', which='minor', size=3, width=1.2, direction='in')

            # 设置边框线宽
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)

            # 添加网格线（可选）
            ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)

    # 添加图例
    from matplotlib.lines import Line2D
    custom_handles = []

    for color, linestyle, marker, label, lw in zip(colors, linestyles, markers, line_labels, line_widths):
        if marker:
            custom_handles.append(
                Line2D([0], [0], color=color, lw=lw, linestyle=linestyle, marker=marker,
                       markersize=8, markerfacecolor='none', markeredgewidth=1.5, label=label)
            )
        else:
            custom_handles.append(
                Line2D([0], [0], color=color, lw=lw, linestyle=linestyle, label=label)
            )

    # 将图例放在第一个子图的合适位置
    axes[0, 0].legend(handles=custom_handles,
                      loc='upper right',
                      frameon=False,
                      fontsize=12,
                      handlelength=2.5,
                      ncol=1)

    # 对齐y轴标签
    fig.align_ylabels(axes[:, 0])

    # 添加总标题
    fig.suptitle(f"Vorticity and Q-criterion Profiles at t = {target_t}",
                 fontsize=14, y=0.98)

    # 保存图形
    for suffix in g_suffix:
        figName = f"vorticity_q_criterion_profiles_t{target_t}.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=300, bbox_inches='tight')
        print(f"✅ 图表已保存: {figFile}")

    plt.close()


if __name__ == "__main__":
    plot_combined_profiles()