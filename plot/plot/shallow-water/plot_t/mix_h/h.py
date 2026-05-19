import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fontSize = 22  # 改为22，与第二个代码一致
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

def plot_combined_three_times():
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    fig_Path = os.path.join(cwd, "figure")
    g_suffix = ["png", "pdf"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
        ###################################################################################

    # 加载数据 - 添加FD和PIKAN数据
    data = np.load("../h_true.npy").flatten().reshape(-1, 1)
    data3 = np.load("../h_pred_PINN.npy").flatten().reshape(-1, 1)
    data4 = np.load("../h_pred_PIFAN.npy").flatten().reshape(-1, 1)

    # 添加新的数据
    data_FD = np.load("../h_pred_FD.npy").flatten().reshape(-1, 1)
    data_PIKAN = np.load("../h_pred_PIKAN.npy").flatten().reshape(-1, 1)

    # 创建 x 和 t 的值
    x = np.linspace(0, 1200, 121)
    t = np.linspace(0, 3600, 61)
    m = np.vstack(np.meshgrid(x, t)).reshape(2, -1).T

    # 将 x 和 t 的信息 concatenate 到数据中
    data = np.concatenate([m, data], axis=1)
    data3 = np.concatenate([m, data3], axis=1)
    data4 = np.concatenate([m, data4], axis=1)
    data_FD = np.concatenate([m, data_FD], axis=1)
    data_PIKAN = np.concatenate([m, data_PIKAN], axis=1)

    # 定义要绘制的时间点
    time_points = [900, 1800, 2700]
    time_labels = [r'$t=900$ s', r'$t=1800$ s', r'$t=2700$ s']

    # 定义子图标签
    subplot_labels = ['(a)', '(b)', '(c)']

    # 完全按照参考代码的样式设置
    plt.rcParams.update({
        'font.size': fontSize,  # 使用统一的fontSize
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.linewidth': 1.2,
        'legend.frameon': False
    })

    # 创建图形和子图
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    plt.subplots_adjust(hspace=0.15, wspace=0.15, top=0.95, bottom=0.1, left=0.15, right=0.95)

    for idx, (ax, t_point, t_label, sub_label) in enumerate(zip(axes, time_points, time_labels, subplot_labels)):
        # 提取当前时间点的数据
        curve_true = data[:, 2][data[:, 1] == t_point]
        curve_pinn = data3[:, 2][data3[:, 1] == t_point]
        curve_pifan = data4[:, 2][data4[:, 1] == t_point]
        curve_FD = data_FD[:, 2][data_FD[:, 1] == t_point]
        curve_PIKAN = data_PIKAN[:, 2][data_PIKAN[:, 1] == t_point]

        # 绘制曲线 - 严格按照参考代码的顺序和线型，通过zorder控制图层
        # 1. DNS: 黑色空心圆圈 (zorder=1, 最底层)
        ax.plot(x[::10], curve_true[::10], 'o', color='#000000', linestyle='', markersize=5,
                markerfacecolor='none', markeredgewidth=1.2, zorder=1, label='DNS' if idx == 0 else "")

        # 2. FD: 绿色实线 (zorder=2)
        ax.plot(x, curve_FD, color='#2ca02c', linestyle='-', lw=2.5, zorder=2, label='FD' if idx == 0 else "")

        # 3. PINN: 棕色点划线 (zorder=3)
        ax.plot(x, curve_pinn, color='#8c564b', linestyle='-.', lw=2.5, zorder=3, label='PINN' if idx == 0 else "")

        # 4. PIKAN: 蓝色虚线 (zorder=4)
        ax.plot(x, curve_PIKAN, color='#1f77b4', linestyle='--', lw=2.5, zorder=4, label='PIKAN' if idx == 0 else "")

        # 5. PIWT: 红色点线 (zorder=5, 最上层)
        ax.plot(x, curve_pifan, color='#ff0000', linestyle=':', lw=2.5, zorder=5, label='PIWF' if idx == 0 else "")

        # 设置y轴标签
        ax.set_ylabel(r'$h$', fontsize=fontSize)

        # 设置y轴范围
        ax.set_ylim([-0.1, 0.5])

        # 设置y轴主刻度
        y_major_ticks = np.arange(-0.1, 0.6, 0.1)
        ax.set_yticks(y_major_ticks)

        # 设置y轴次要刻度
        y_minor_ticks = np.arange(-0.1, 0.5, 0.02)
        ax.set_yticks(y_minor_ticks, minor=True)

        # 格式化y轴标签
        y_tick_labels = []
        for tick in y_major_ticks:
            if tick == 0:
                y_tick_labels.append("0")
            elif tick == -0.1:
                y_tick_labels.append("-0.1")
            else:
                label = f"{tick:.1f}"
                if label.endswith('.0'):
                    label = label[:-2]
                y_tick_labels.append(label)
        ax.set_yticklabels(y_tick_labels, fontsize=fontSize - 2)

        # 添加时间标签 - 放在右上角
        ax.text(0.97, 0.9, t_label, transform=ax.transAxes,
                fontsize=fontSize, ha='right', fontweight='bold')

        # 在每个子图的左下角添加对应的注释
        ax.text(0.03, 0.03, sub_label, transform=ax.transAxes,
                fontsize=fontSize, ha='left', va='bottom',
                fontweight='normal', fontfamily='Times New Roman')

        # 设置刻度参数
        ax.tick_params(axis='both', which='major', size=7, width=1.5, direction='in', labelsize=fontSize - 2)
        ax.tick_params(axis='both', which='minor', size=4, width=1.2, direction='in')

        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # 设置x轴
    axes[-1].set_xlabel(r'$x$', fontsize=fontSize)
    axes[-1].set_xlim([0, 1200])

    # 设置x轴主刻度
    x_major_ticks = np.arange(0, 1201, 200)
    axes[-1].set_xticks(x_major_ticks)
    axes[-1].set_xticklabels([f"{int(tick)}" for tick in x_major_ticks], fontsize=fontSize - 2)

    # 设置x轴次要刻度
    x_minor_ticks = np.arange(0, 1201, 40)
    for ax in axes:
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.tick_params(axis='x', which='minor', length=4, width=1.2, direction='in')

    # 图例处理 - 放在第一个子图的左上角内部
    from matplotlib.lines import Line2D

    # 定义图例 - 使用修改后的颜色和线型
    custom_handles = [
        # DNS: 只有空心圆圈
        Line2D([0], [0], color='#000000', lw=0, linestyle='none', marker='o',
               markersize=5, markerfacecolor='none', markeredgewidth=1.2, label='DNS'),
        # FD: 绿色实线
        Line2D([0], [0], color='#2ca02c', lw=2.5, linestyle='-', label='FD'),
        # PINN: 棕色点划线
        Line2D([0], [0], color='#8c564b', lw=2.5, linestyle='-.', label='PINN'),
        # PIKAN: 蓝色虚线
        Line2D([0], [0], color='#1f77b4', lw=2.5, linestyle='--', label='PIKAN'),
        # PIWT: 红色点线
        Line2D([0], [0], color='#ff0000', lw=2.5, linestyle=':', label='PIWF')
    ]

    # 将图例放在第一个子图的左上角，调整为两列显示
    axes[0].legend(handles=custom_handles,
                   loc='upper left',
                   frameon=False,
                   fontsize=fontSize - 8,
                   handlelength=2.5,
                   ncol=2,
                   columnspacing=1.0,
                   borderaxespad=0.5)

    # 对齐y轴标签
    fig.align_ylabels(axes)

    # 保存图形
    for suffix in g_suffix:
        figName = f"combined_h_profiles_times.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=600, bbox_inches='tight')
        print(f"✅ 图表已保存: {figFile}")

    plt.close()


if __name__ == "__main__":
    plot_combined_three_times()