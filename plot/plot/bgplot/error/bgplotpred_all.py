import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fontSize = 20
config = {
    "text.usetex": False,
    "font.family": "Times New Roman",
    "font.size": fontSize,
    "mathtext.fontset": 'cm',
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.2,
    "axes.grid": False,
}
mpl.rcParams.update(config)

plt.rcParams['xtick.labelsize'] = fontSize
plt.rcParams['ytick.labelsize'] = fontSize
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['ytick.major.pad'] = 8

###################################################################################
##### The property of drawing graphs #####
# 定义统一的颜色和线型
colors = ['darkgreen', 'b', 'C1', 'r']  # FD, PINN, PIKAN, PIWT
linestyles = ['-', ':', (0, (5, 5)), '--']  # 实线，点线，长虚线，虚线
line_labels = ['FD', 'PINN', 'PIKAN', 'PIWF']
line_widths = [2.5, 2.5, 2.5, 2.5]
###################################################################################


def scientific_format(x, pos):
    """自定义科学计数法格式化函数"""
    if x == 0:
        return '0'
    power = int(np.floor(np.log10(abs(x))))
    if power == 0:
        return '$1$'
    else:
        return f'$10^{{{power}}}$'


if __name__ == "__main__":
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    fig_Path = os.path.join(cwd, "figure")
    g_suffix = ["pdf", "png"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
    ###################################################################################

    print("📂 加载误差数据...")

    # ==================== 加载第一个图的数据 ====================
    try:
        data_piwt_1 = np.load("u_resBG_PIFAN.npy")
        data_pinn_1 = np.load("u_resBG_PINN.npy")
        data_pikan_1 = np.load("u_resBG_PIKAN.npy")
        data_fd_1 = np.load("u_resBG_FD.npy")

        error_piwt_1 = np.sqrt(np.mean(data_piwt_1 ** 2, axis=1)) / 100
        error_pinn_1 = np.sqrt(np.mean(data_pinn_1 ** 2, axis=1)) / 100
        error_pikan_1 = np.sqrt(np.mean(data_pikan_1 ** 2, axis=1)) / 100
        error_fd_1 = np.sqrt(np.mean(data_fd_1 ** 2, axis=1)) / 100

        t = np.linspace(0, 1, 100)

        print(f"图1 - FD误差范围: {error_fd_1.min():.2e} - {error_fd_1.max():.2e}")
        print(f"图1 - PINN误差范围: {error_pinn_1.min():.2e} - {error_pinn_1.max():.2e}")
        print(f"图1 - PIKAN误差范围: {error_pikan_1.min():.2e} - {error_pikan_1.max():.2e}")
        print(f"图1 - PIWT误差范围: {error_piwt_1.min():.2e} - {error_piwt_1.max():.2e}")

    except Exception as e:
        print(f"❌ 加载第一个图数据失败: {e}")
        print("请确保以下文件存在:")
        print("  - u_resBG_PIFAN.npy")
        print("  - u_resBG_PINN.npy")
        print("  - u_resBG_PIKAN.npy")
        print("  - u_resBG_FD.npy")
        exit(1)

    # ==================== 加载第二个图的数据 ====================
    try:
        data_piwt_2 = np.load("u_resBG_PIFANtanh.npy")
        data_pinn_2 = np.load("u_resBG_PINNtanh.npy")
        data_pikan_2 = np.load("u_resBG_PIKAN1000.npy")
        data_fd_2 = np.load("u_resBG_FD1000.npy")

        error_piwt_2 = np.sqrt(np.mean(data_piwt_2 ** 2, axis=1)) / 100
        error_pinn_2 = np.sqrt(np.mean(data_pinn_2 ** 2, axis=1)) / 100
        error_pikan_2 = np.sqrt(np.mean(data_pikan_2 ** 2, axis=1)) / 100
        error_fd_2 = np.sqrt(np.mean(data_fd_2 ** 2, axis=1)) / 100

        print(f"图2 - FD误差范围: {error_fd_2.min():.2e} - {error_fd_2.max():.2e}")
        print(f"图2 - PINN误差范围: {error_pinn_2.min():.2e} - {error_pinn_2.max():.2e}")
        print(f"图2 - PIKAN误差范围: {error_pikan_2.min():.2e} - {error_pikan_2.max():.2e}")
        print(f"图2 - PIWT误差范围: {error_piwt_2.min():.2e} - {error_piwt_2.max():.2e}")

    except Exception as e:
        print(f"❌ 加载第二个图数据失败: {e}")
        print("请确保以下文件存在:")
        print("  - u_resBG_PIFANtanh.npy")
        print("  - u_resBG_PINNtanh.npy")
        print("  - u_resBG_PIKAN.npy")
        print("  - u_resBG_FD1000.npy")
        exit(1)

    # ==================== 创建左右排列的两个子图 ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    # 设置两个子图的公共参数
    for ax in [ax1, ax2]:
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)

        ax.tick_params(axis='both', which='major', size=6, width=1.2, direction='in', labelsize=fontSize)
        ax.tick_params(axis='both', which='minor', size=3, width=1.2, direction='in')
        ax.tick_params(axis='x', pad=8)
        ax.tick_params(axis='y', pad=8)

        ax.set_xlim([0, 1])
        ax.set_xlabel(r'$t$', fontsize=20, fontfamily='Times New Roman')
        ax.set_yscale('log')

        # 设置x轴刻度和标签
        x_major_ticks = np.arange(0, 1.1, 0.2)
        ax.set_xticks(x_major_ticks)
        x_tick_labels = []
        for tick in x_major_ticks:
            if abs(tick - round(tick)) < 0.001:
                x_tick_labels.append(f"{int(tick)}")
            else:
                label = f"{tick:.1f}"
                if '.' in label:
                    label = label.rstrip('0').rstrip('.')
                x_tick_labels.append(label)
        ax.set_xticklabels(x_tick_labels, fontfamily='Times New Roman')

        # 设置x轴次要刻度
        x_minor_ticks = np.arange(0, 1.01, 0.04)
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.tick_params(axis='x', which='minor', length=3, width=1.0, direction='in', labelbottom=False)

    # ==================== 处理第一个子图 (图1) ====================
    # 计算y轴范围
    all_errors_1 = np.concatenate([error_fd_1, error_pinn_1, error_pikan_1, error_piwt_1])
    all_errors_1 = all_errors_1[all_errors_1 > 0]

    if len(all_errors_1) > 0:
        y_min_1 = max(1e-5, np.min(all_errors_1) * 0.5)
        y_max_1 = np.max(all_errors_1) * 2.0
    else:
        y_min_1 = 1e-5
        y_max_1 = 1e-1

    y_min_power_1 = np.floor(np.log10(y_min_1))
    y_max_power_1 = np.ceil(np.log10(y_max_1))
    y_min_1 = 10 ** y_min_power_1
    y_max_1 = 10 ** y_max_power_1

    # 调整上限
    max_error_1 = np.max(all_errors_1)
    if max_error_1 > y_max_1:
        y_max_power_1 = np.ceil(np.log10(max_error_1 * 1.5))
        y_max_1 = 10 ** y_max_power_1

    ax1.set_ylim([y_min_1, y_max_1])

    # 设置y轴刻度
    powers_1 = np.arange(y_min_power_1, y_max_power_1 + 1)
    y_major_ticks_1 = 10 ** powers_1
    y_major_ticks_1 = y_major_ticks_1[(y_major_ticks_1 >= y_min_1 * 0.9) & (y_major_ticks_1 <= y_max_1 * 1.1)]
    if len(y_major_ticks_1) == 0 or y_major_ticks_1[0] > y_min_1 * 1.1:
        y_major_ticks_1 = np.concatenate([[y_min_1], y_major_ticks_1])
    if len(y_major_ticks_1) > 0 and y_major_ticks_1[-1] < y_max_1 * 0.9:
        y_major_ticks_1 = np.concatenate([y_major_ticks_1, [y_max_1]])
    ax1.set_yticks(y_major_ticks_1)

    # 添加y轴次要刻度
    y_minor_ticks_1 = []
    for i in range(len(powers_1) - 1):
        start_power = powers_1[i]
        for factor in [2, 3, 4, 5, 6, 7, 8, 9]:
            minor_tick = factor * (10 ** start_power)
            if minor_tick >= y_min_1 * 0.9 and minor_tick <= y_max_1 * 1.1:
                y_minor_ticks_1.append(minor_tick)
    ax1.set_yticks(y_minor_ticks_1, minor=True)

    # 应用科学计数法格式化器
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(scientific_format)
    ax1.yaxis.set_major_formatter(formatter)

    # 设置y轴标签
    ax1.set_ylabel(r'Er($u$)', fontsize=20, fontfamily='Times New Roman')

    # 绘制第一个图的曲线 - 按统一标准修改线型（只有线条，没有标记）
    ax1.plot(t, error_fd_1, color='#2ca02c', linestyle='-', linewidth=2.5, zorder=3, label='FD')
    ax1.plot(t, error_pinn_1, color='#8c564b', linestyle='-.', linewidth=2.5, zorder=5, label='PINN')
    ax1.plot(t, error_pikan_1, color='#1f77b4', linestyle='--', linewidth=2.5, zorder=7, label='PIKAN')
    ax1.plot(t, error_piwt_1, color='#ff0000', linestyle=':', linewidth=2.5, zorder=9, label='PIWF')

    # 添加注释(a)在左下角，使用LaTeX格式
    ax1.text(0.05, 0.05, r'(a) $\nu=0.01/\pi$', transform=ax1.transAxes,
             fontsize=fontSize, ha='left', va='bottom',
             fontweight='normal', fontfamily='Times New Roman')

    # 添加图例在第一个子图的上方内部，横着一排显示
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.98), frameon=False,
               fontsize=14, handlelength=2.5, ncol=4)

    # ==================== 处理第二个子图 (图2) ====================
    # 计算y轴范围
    all_errors_2 = np.concatenate([error_fd_2, error_pinn_2, error_pikan_2, error_piwt_2])
    all_errors_2 = all_errors_2[all_errors_2 > 0]

    if len(all_errors_2) > 0:
        y_min_2 = max(1e-5, np.min(all_errors_2) * 0.5)
        y_max_2 = np.max(all_errors_2) * 2.0
    else:
        y_min_2 = 1e-5
        y_max_2 = 1e-1

    y_min_power_2 = np.floor(np.log10(y_min_2))
    y_max_power_2 = np.ceil(np.log10(y_max_2))
    y_min_2 = 10 ** y_min_power_2
    y_max_2 = 10 ** y_max_power_2

    # 调整上限
    max_error_2 = np.max(all_errors_2)
    if max_error_2 > y_max_2:
        y_max_power_2 = np.ceil(np.log10(max_error_2 * 1.5))
        y_max_2 = 10 ** y_max_power_2

    ax2.set_ylim([y_min_2, y_max_2])

    # 设置y轴刻度
    powers_2 = np.arange(y_min_power_2, y_max_power_2 + 1)
    y_major_ticks_2 = 10 ** powers_2
    y_major_ticks_2 = y_major_ticks_2[(y_major_ticks_2 >= y_min_2 * 0.9) & (y_major_ticks_2 <= y_max_2 * 1.1)]
    if len(y_major_ticks_2) == 0 or y_major_ticks_2[0] > y_min_2 * 1.1:
        y_major_ticks_2 = np.concatenate([[y_min_2], y_major_ticks_2])
    if len(y_major_ticks_2) > 0 and y_major_ticks_2[-1] < y_max_2 * 0.9:
        y_major_ticks_2 = np.concatenate([y_major_ticks_2, [y_max_2]])
    ax2.set_yticks(y_major_ticks_2)

    # 添加y轴次要刻度
    y_minor_ticks_2 = []
    for i in range(len(powers_2) - 1):
        start_power = powers_2[i]
        for factor in [2, 3, 4, 5, 6, 7, 8, 9]:
            minor_tick = factor * (10 ** start_power)
            if minor_tick >= y_min_2 * 0.9 and minor_tick <= y_max_2 * 1.1:
                y_minor_ticks_2.append(minor_tick)
    ax2.set_yticks(y_minor_ticks_2, minor=True)

    # 应用科学计数法格式化器
    ax2.yaxis.set_major_formatter(formatter)

    # 设置y轴标签（隐藏，因为与第一个子图共享）
    ax2.set_ylabel('')
    ax2.tick_params(axis='y', which='both', labelleft=False)

    # 绘制第二个图的曲线 - 按统一标准修改线型（只有线条，没有标记）
    ax2.plot(t, error_fd_2, color='#2ca02c', linestyle='-', linewidth=2.5, zorder=3, label='FD')
    ax2.plot(t, error_pinn_2, color='#8c564b', linestyle='-.', linewidth=2.5, zorder=5, label='PINN')
    ax2.plot(t, error_pikan_2, color='#1f77b4', linestyle='--', linewidth=2.5, zorder=7, label='PIKAN')
    ax2.plot(t, error_piwt_2, color='#ff0000', linestyle=':', linewidth=2.5, zorder=9, label='PIWF')

    # 添加注释(b)在左下角，使用LaTeX格式
    ax2.text(0.05, 0.05, r'(b) $\nu=0.001$', transform=ax2.transAxes,
             fontsize=fontSize, ha='left', va='bottom',
             fontweight='normal', fontfamily='Times New Roman')

    # 调整布局
    plt.subplots_adjust(wspace=0.08, top=0.92, bottom=0.12, left=0.08, right=0.98)

    # 保存图形
    for suffix in g_suffix:
        figName = f"combined_error_plots_u.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=300, bbox_inches='tight')
        print(f"✅ 组合图已保存: {figFile}")

    plt.close()