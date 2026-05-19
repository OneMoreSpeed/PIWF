import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, FuncFormatter

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fontSize = 32  # 与第一个代码一致
########## font: Time New Roman ##########
config = {
    "text.usetex": False,
    "font.family": "Times New Roman",
    "font.size": fontSize,
    "mathtext.fontset": 'cm',
    "xtick.direction": "in",
    "ytick.direction": "in",
}
mpl.rcParams.update(config)
###################################################################################
##### The property of drawing graphs #####
# 定义颜色和样式 - 按照PINN, PIKAN, PIWT的顺序
pde_colors = ['#00D200', '#0000FF', '#FF0000']  # PINN绿, PIKAN蓝, PIWT红
train_colors = ['#32CD32', '#6495ED', '#FF6B6B']  # PINN浅绿, PIKAN浅蓝, PIWT浅红

# PDE损失使用实线，训练损失使用虚线
pde_linestyles = ["-", "-", "-"]  # 所有PDE损失都用实线
train_linestyles = ["--", "--", "--"]  # 所有训练损失都用虚线

linewidths = [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]


###################################################################################

def moving_average_convolve(arr, window_size):
    """移动平均平滑函数"""
    window = np.ones(window_size) / window_size
    return np.convolve(arr, window, 'valid')


def apply_edge_padding(arr, window_size):
    """处理边缘效应，保持数组长度不变"""
    if len(arr) <= window_size:
        return arr

    half_window = window_size // 2
    # 使用对称填充处理边缘
    padded_arr = np.pad(arr, (half_window, half_window), mode='reflect')
    smoothed = moving_average_convolve(padded_arr, window_size)
    return smoothed


if __name__ == "__main__":
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    fig_Path = os.path.join(cwd, "figure_pdeloss")  # 保持原路径
    g_suffix = ["svg", "pdf", "png"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
    ###################################################################################

    # x & y labels
    xlabel = r"Epoch ($\times 10^4$)"
    ylabel = r"Loss"

    ###################################################################################
    # 加载所有损失数据
    print("📂 加载损失数据...")

    # PINN 数据
    try:
        loss_pinn_train = np.load("./[3, 128, 128, 128, 128, 3]layer_LBFGS_train_loss.npy")
        loss_pinn_pde = np.load("./[3, 128, 128, 128, 128, 3]layer_LBFGS_pde_loss.npy")
        print(
            f"  PINN训练损失: {len(loss_pinn_train)}个点, 范围: {loss_pinn_train.min():.2e} - {loss_pinn_train.max():.2e}")
        print(f"  PINN PDE损失: {len(loss_pinn_pde)}个点, 范围: {loss_pinn_pde.min():.2e} - {loss_pinn_pde.max():.2e}")
    except Exception as e:
        print(f"❌ 加载PINN数据失败: {e}")
        loss_pinn_train = np.array([])
        loss_pinn_pde = np.array([])

    # PIWT 数据
    try:
        loss_piwt_train = np.load("./4layer_LBFGS_train_loss.npy")
        loss_piwt_pde = np.load("./4layer_LBFGS_pde_loss.npy")
        print(
            f"  PIWT训练损失: {len(loss_piwt_train)}个点, 范围: {loss_piwt_train.min():.2e} - {loss_piwt_train.max():.2e}")
        print(f"  PIWT PDE损失: {len(loss_piwt_pde)}个点, 范围: {loss_piwt_pde.min():.2e} - {loss_piwt_pde.max():.2e}")
    except Exception as e:
        print(f"❌ 加载PIWT数据失败: {e}")
        loss_piwt_train = np.array([])
        loss_piwt_pde = np.array([])

    # PIKAN 数据 - 使用与第二个代码相同的路径
    try:
        loss_pikan_pde = np.load("./pikanlayer_LBFGS_pde_loss.npy")
        print(
            f"  PIKAN PDE损失: {len(loss_pikan_pde)}个点, 范围: {loss_pikan_pde.min():.2e} - {loss_pikan_pde.max():.2e}")

        # 尝试加载PIKAN训练损失
        try:
            loss_pikan_train = np.load("./pikanlayer_LBFGS_train_loss.npy")
            print(
                f"  PIKAN训练损失: {len(loss_pikan_train)}个点, 范围: {loss_pikan_train.min():.2e} - {loss_pikan_train.max():.2e}")
        except:
            print("  ⚠️ 未找到PIKAN训练损失，使用PDE损失代替")
            loss_pikan_train = loss_pikan_pde.copy()
    except Exception as e:
        print(f"❌ 加载PIKAN数据失败: {e}")
        loss_pikan_train = np.array([])
        loss_pikan_pde = np.array([])

    # 检查数据长度是否一致，如果不同则截断到最小长度
    valid_lengths = []
    for data in [loss_pinn_train, loss_pinn_pde, loss_piwt_train, loss_piwt_pde, loss_pikan_train, loss_pikan_pde]:
        if len(data) > 0:
            valid_lengths.append(len(data))

    if valid_lengths:
        min_length = min(valid_lengths)
        print(f"⚙️ 统一数据长度为: {min_length}")

        if len(loss_pinn_train) > 0 and len(loss_pinn_train) > min_length:
            loss_pinn_train = loss_pinn_train[:min_length]
        if len(loss_pinn_pde) > 0 and len(loss_pinn_pde) > min_length:
            loss_pinn_pde = loss_pinn_pde[:min_length]
        if len(loss_piwt_train) > 0 and len(loss_piwt_train) > min_length:
            loss_piwt_train = loss_piwt_train[:min_length]
        if len(loss_piwt_pde) > 0 and len(loss_piwt_pde) > min_length:
            loss_piwt_pde = loss_piwt_pde[:min_length]
        if len(loss_pikan_train) > 0 and len(loss_pikan_train) > min_length:
            loss_pikan_train = loss_pikan_train[:min_length]
        if len(loss_pikan_pde) > 0 and len(loss_pikan_pde) > min_length:
            loss_pikan_pde = loss_pikan_pde[:min_length]
    else:
        print("⚠️ 所有数据文件为空或加载失败")
        exit()

    # 打印最小值信息
    if len(loss_pinn_pde) > 0:
        min_pinn_pde = np.min(loss_pinn_pde)
        min_idx_pinn_pde = np.argmin(loss_pinn_pde)
        print(f"📊 PINN PDE损失最小值: {min_pinn_pde:.2e} (位置: {min_idx_pinn_pde})")

    if len(loss_pikan_pde) > 0:
        min_pikan_pde = np.min(loss_pikan_pde)
        min_idx_pikan_pde = np.argmin(loss_pikan_pde)
        print(f"📊 PIKAN PDE损失最小值: {min_pikan_pde:.2e} (位置: {min_idx_pikan_pde})")

    if len(loss_piwt_pde) > 0:
        min_piwt_pde = np.min(loss_piwt_pde)
        min_idx_piwt_pde = np.argmin(loss_piwt_pde)
        print(f"📊 PIWT PDE损失最小值: {min_piwt_pde:.2e} (位置: {min_idx_piwt_pde})")

    ###################################################################################
    # 应用平滑处理
    print("🔄 应用平滑处理...")
    window_size = 49

    # 对每条曲线应用平滑 - 按照PINN, PIKAN, PIWT的顺序
    curves_to_smooth = [
        (loss_pinn_pde, "PINN PDE"),
        (loss_pikan_pde, "PIKAN PDE"),
        (loss_piwt_pde, "PIWT PDE"),
        (loss_pinn_train, "PINN训练"),
        (loss_pikan_train, "PIKAN训练"),
        (loss_piwt_train, "PIWT训练")
    ]

    smoothed_curves = []
    for curve, name in curves_to_smooth:
        if len(curve) > 0:
            try:
                smoothed = apply_edge_padding(curve, window_size)
                if len(smoothed) != len(curve):
                    print(f"  ⚠️ {name}平滑后长度不匹配: {len(curve)} -> {len(smoothed)}")
                    smoothed = curve
                smoothed_curves.append(smoothed)
                print(f"  ✓ {name}平滑完成")
            except Exception as e:
                print(f"  ❌ {name}平滑失败: {e}")
                smoothed_curves.append(curve)
        else:
            print(f"  ⚠️ {name}数据为空，跳过平滑")
            smoothed_curves.append(np.array([]))

    # 解包平滑后的曲线
    if len(smoothed_curves) == 6:
        curve_pinn_pde_s, curve_pikan_pde_s, curve_piwt_pde_s, curve_pinn_train_s, curve_pikan_train_s, curve_piwt_train_s = smoothed_curves
    else:
        curve_pinn_pde_s = loss_pinn_pde
        curve_pikan_pde_s = loss_pikan_pde
        curve_piwt_pde_s = loss_piwt_pde
        curve_pinn_train_s = loss_pinn_train
        curve_pikan_train_s = loss_pikan_train
        curve_piwt_train_s = loss_piwt_train

    ###################################################################################
    # 绘制图形
    fig = plt.figure(figsize=(10, 7), dpi=600)
    ax = fig.add_axes([0.12, 0.12, 0.85, 0.85])

    # 设置边框线宽和可见性
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_visible(True)

    # 设置刻度参数 - 与第一个代码一致
    ax.tick_params(axis='both', which='major', size=7, width=2.5, direction='in',
                   top=False, right=False, labelsize=fontSize - 2)  # 刻度标签大小30
    ax.tick_params(axis='both', which='minor', size=4, width=2.0, direction='in',
                   top=False, right=False)
    ax.tick_params(axis='x', pad=10, which='both')
    ax.tick_params(axis='y', pad=10, which='both')

    # 设置坐标轴范围
    x_limit = [0, 300000]
    y_limit = [1e-10, 1]
    ax.set_xlim(x_limit[0], x_limit[1])
    ax.set_ylim(y_limit[0], y_limit[1])

    # 设置x轴刻度
    x_major_ticks = np.arange(x_limit[0], 200001, 50000)
    ax.set_xticks(x_major_ticks)

    # 格式化x轴标签，转换为万单位
    x_tick_labels = []
    for tick in x_major_ticks:
        value = tick / 10000
        if value == int(value):
            x_tick_labels.append(f"{int(value)}")
        else:
            label = f"{value:.1f}".rstrip('0').rstrip('.')
            x_tick_labels.append(label)

    ax.set_xticklabels(x_tick_labels, fontfamily='Times New Roman', fontsize=fontSize - 2)  # x轴刻度标签大小30

    # x轴次要刻度
    x_minor_ticks = np.arange(x_limit[0], 200001, 10000)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=4, width=1.5, labelbottom=False)

    # 设置y轴为对数坐标
    ax.set_yscale('log')

    # 设置y轴主要刻度
    y_major_ticks = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0]
    ax.set_yticks(y_major_ticks)

    # 设置y轴次要刻度
    y_minor_ticks = []
    for i in range(len(y_major_ticks) - 1):
        start = np.log10(y_major_ticks[i])
        for factor in [2, 3, 4, 5, 6, 7, 8, 9]:
            minor_tick = factor * 10 ** start
            if minor_tick < y_major_ticks[i + 1] * 0.99:
                y_minor_ticks.append(minor_tick)

    ax.set_yticks(y_minor_ticks, minor=True)
    ax.tick_params(axis='y', which='minor', length=3, width=1.2, labelleft=False)

    # 自定义科学计数法格式化函数 - 与第一个代码一致
    def log_format(x, pos):
        if x == 0:
            return '0'
        power = int(np.floor(np.log10(abs(x))))
        if power == 0:
            return '$1$'
        else:
            return f'$10^{{{power}}}$'

    # 应用格式化器
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(log_format)
    ax.yaxis.set_major_formatter(formatter)

    # 确保刻度只在底部和左侧
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # 添加坐标轴标签 - 与第一个代码一致
    ax.set_xlabel(xlabel, fontsize=fontSize, fontweight='normal', fontfamily='Times New Roman', labelpad=12)  # x轴标签大小32
    ax.set_ylabel(ylabel, fontsize=fontSize, fontweight='normal', fontfamily='Times New Roman', labelpad=12)  # y轴标签大小32

    ###################################################################################
    # 绘制曲线 - 按照PINN, PIKAN, PIWT的顺序
    print("🎨 绘制曲线...")

    # 定义图例标签
    legend_labels = [
        "PDE loss (PINN)",
        "PDE loss (PIKAN)",
        "PDE loss (PIWT)",
    ]

    curves_to_plot = [
        (curve_pinn_pde_s, pde_colors[0], pde_linestyles[0], linewidths[0]),  # PINN PDE
        (curve_pikan_pde_s, pde_colors[1], pde_linestyles[1], linewidths[1]),  # PIKAN PDE
        (curve_piwt_pde_s, pde_colors[2], pde_linestyles[2], linewidths[2])   # PIWT PDE
    ]

    plotted_handles = []
    plotted_labels = []

    for i, (curve_data, color, linestyle, linewidth) in enumerate(curves_to_plot):
        if len(curve_data) > 0:
            x_data = np.linspace(0, 200000, len(curve_data))

            line = ax.plot(x_data, curve_data,
                           color=color,
                           linestyle=linestyle,
                           linewidth=linewidth,
                           alpha=0.9)

            plotted_handles.append(line[0])
            plotted_labels.append(legend_labels[i])
            print(f"  ✓ 绘制 {legend_labels[i]}: {len(curve_data)}个点")
        else:
            print(f"  ✗ 跳过 {legend_labels[i]}: 数据为空")

    ###################################################################################
    # 设置图例 - 与第一个代码一致
    if plotted_handles:
        print("🏷️ 设置图例...")

        # 图例字体大小与第一个代码一致 (fontSize - 12 = 20)
        legend_font = {'family': 'Times New Roman', 'size': fontSize - 12, 'weight': 'normal'}

        legend = ax.legend(plotted_handles, plotted_labels,
                           loc='upper right',
                           frameon=False,
                           prop=legend_font,
                           handlelength=1.8,
                           labelspacing=0.2,
                           borderpad=0.3,
                           ncol=1,
                           columnspacing=0.6)

        for text in legend.get_texts():
            text.set_fontfamily('Times New Roman')
    else:
        print("⚠️ 没有可显示的曲线，跳过图例设置")

    # 在左下角添加注释(b) - 与第一个代码一致
    ax.text(0.02, 0.02, r'(b)', transform=ax.transAxes,
            fontsize=fontSize, ha='left', va='bottom',
            fontweight='normal', fontfamily='Times New Roman')

    ###################################################################################
    # 保存图形
    print("💾 保存图形...")

    for suffix in g_suffix:
        figName = f"trainloss.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=600, bbox_inches='tight', pad_inches=0.1)
        print(f"  ✅ 保存: {figFile}")

    plt.close()
    print("🎉 完成!")