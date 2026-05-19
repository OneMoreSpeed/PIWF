import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fontSize = 26  # 改为32，与第一个图一致
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
colors = ['#ff0000', '#8c564b', '#1f77b4']  # 红色，棕色，蓝色
linestyles = [':', '-.', '--']
line_labels = ['PIWF', 'PINN', 'PIKAN']
line_widths = [2.5, 2.5, 2.5]


###################################################################################


def plot_combined_errors():
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    fig_Path = os.path.join(cwd, "figure_errors")
    g_suffix = ["png", "pdf"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
        ###################################################################################

    # 定义要绘制的变量
    variables = ['u', 'v', 'p']
    var_names = [r'$u$', r'$v$', r'$p$']
    subplot_labels = [r'(a)', r'(b)', r'(c)']  # 子图注释

    # x轴数据
    x = np.linspace(0, 7, 15)

    # 创建图形和子图 - 横向排列（1行3列），宽高比9:6
    fig, axes = plt.subplots(1, 3, figsize=(27, 6), sharey=False)
    plt.subplots_adjust(wspace=0.1, top=0.92, bottom=0.12, left=0.08, right=0.98)

    # 固定y轴范围为10^-4 到 10^-1
    y_min = 1e-4
    y_max = 1e-1

    # 设置y轴主要刻度
    y_major_ticks = [1e-4, 1e-3, 1e-2, 1e-1]

    # 添加y轴次要刻度（子刻度线）
    y_minor_ticks = []
    powers = [-4, -3, -2, -1]
    for i in range(len(powers) - 1):
        start_power = powers[i]
        for factor in [2, 3, 4, 5, 6, 7, 8, 9]:
            minor_tick = factor * (10 ** start_power)
            if minor_tick >= y_min and minor_tick <= y_max:
                y_minor_ticks.append(minor_tick)

    # 自定义科学计数法格式化函数
    def scientific_format(x, pos):
        if x == 0:
            return '0'
        power = int(np.floor(np.log10(abs(x))))
        if power == 0:
            return '$1$'
        else:
            return f'$10^{{{power}}}$'

    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(scientific_format)

    # 柱状图的宽度
    bar_width = 0.25

    # 遍历每个变量
    for idx, (ax, var, var_label, sub_label) in enumerate(zip(axes, variables, var_names, subplot_labels)):
        print(f"处理变量: {var}")

        try:
            # 加载误差数据
            data_pinn = np.load(f"res_{var}_mean_PINN.npy")
            data_pifan = np.load(f"res_{var}_mean_PIFAN.npy")
            data_pikan = np.load(f"res_{var}_mean_PIKAN.npy")

            # 处理数据
            error_pinn = data_pinn / 100
            error_pifan = data_pifan / 100
            error_pikan = data_pikan / 100

            # 设置每组柱子的x位置
            x_positions = np.arange(len(x))

            # 绘制柱状图 - 顺序：PINN（第一个柱），PIKAN（第二个柱），PIWT（第三个柱）
            # 1. PINN: 棕色（第一个柱，位置 x_positions - bar_width）
            ax.bar(x_positions - bar_width, error_pinn, width=bar_width,
                   color='#8c564b', edgecolor='black', linewidth=0.8,
                   zorder=3, label='PINN', alpha=0.8)

            # 2. PIKAN: 蓝色（第二个柱，位置 x_positions）
            ax.bar(x_positions, error_pikan, width=bar_width,
                   color='#1f77b4', edgecolor='black', linewidth=0.8,
                   zorder=2, label='PIKAN', alpha=0.8)

            # 3. PIWT: 红色（第三个柱，位置 x_positions + bar_width）
            ax.bar(x_positions + bar_width, error_pifan, width=bar_width,
                   color='#ff0000', edgecolor='black', linewidth=0.8,
                   zorder=1, label='PIWF', alpha=0.8)

            # 设置x轴刻度的位置和标签
            ax.set_xticks(x_positions)
            # 格式化x轴标签
            x_tick_labels = []
            for tick in x:
                if abs(tick - round(tick)) < 0.001:
                    x_tick_labels.append(f"{int(tick)}")
                else:
                    label = f"{tick:.1f}"
                    if '.' in label:
                        label = label.rstrip('0').rstrip('.')
                    x_tick_labels.append(label)
            ax.set_xticklabels(x_tick_labels, fontfamily='Times New Roman', fontsize=fontSize - 2)

            # 设置y轴标签 - 只有第一个子图显示
            if idx == 0:
                ax.set_ylabel('Er', fontsize=fontSize, fontfamily='Times New Roman')
            else:
                # 第二、第三个子图隐藏y轴标签和数字
                ax.set_ylabel('')
                ax.tick_params(axis='y', which='both', labelleft=False, left=False)

            # 设置y轴范围为对数坐标
            ax.set_yscale('log')
            ax.set_ylim([y_min, y_max])

            # 设置y轴刻度 - 只有第一个子图显示
            if idx == 0:
                ax.set_yticks(y_major_ticks)
                ax.set_yticks(y_minor_ticks, minor=True)
                # 应用格式化器
                ax.yaxis.set_major_formatter(formatter)
            else:
                # 第二、三个子图不显示y轴刻度
                ax.set_yticks([])
                ax.set_yticks([], minor=True)

            # 设置x轴标签
            ax.set_xlabel(r'$t$', fontsize=fontSize, fontfamily='Times New Roman')
            ax.set_xlim([-0.5, len(x) - 0.5])

            # 设置x轴主刻度
            ax.set_xticks(x_positions)

            # 设置刻度参数
            ax.tick_params(axis='both', which='major', size=7, width=1.5, direction='in',
                           labelsize=fontSize - 2)
            ax.tick_params(axis='both', which='minor', size=4, width=1.2, direction='in')
            ax.tick_params(axis='x', which='minor', labelbottom=False)

            # 设置边框线宽
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            # 在子图右上角添加注释(a)、(b)、(c)
            ax.text(0.97, 0.97, sub_label, transform=ax.transAxes,
                    fontsize=fontSize, ha='right', va='top',
                    fontweight='normal', fontfamily='Times New Roman')

        except Exception as e:
            print(f"❌ 加载{var}数据失败: {e}")
            continue

    # 图例处理 - 放在第一个子图内部的上方横向排列
    from matplotlib.patches import Patch

    custom_handles = []
    # 顺序：PINN（棕色），PIKAN（蓝色），PIWT（红色）
    legend_colors = ['#8c564b', '#1f77b4', '#ff0000']
    legend_labels = ['PINN', 'PIKAN', 'PIWF']

    for color, label in zip(legend_colors, legend_labels):
        custom_handles.append(
            Patch(facecolor=color, edgecolor='black', alpha=0.8, label=label)
        )

    # 将图例放在第一个子图内部的上方，横向排列
    legend = axes[0].legend(handles=custom_handles,
                            loc='upper center',
                            bbox_to_anchor=(0.5, 1.02),  # 放在子图上方，横向居中
                            frameon=False,
                            fontsize=fontSize - 8,
                            handlelength=2.5,
                            ncol=3)  # ncol=3 表示横向排列3列

    # 设置图例字体
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')

    # 保存图形
    for suffix in g_suffix:
        figName = f"combined_errors_uvp_horizontal_bar.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=600, bbox_inches='tight')
        print(f"✅ 误差图表已保存: {figFile}")

    plt.close()


if __name__ == "__main__":
    plot_combined_errors()