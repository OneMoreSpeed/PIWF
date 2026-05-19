import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib - JCP style #####
fontSize = 32
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


# 自定义刻度格式化函数，去掉多余的0
def format_ticks(x, pos):
    """格式化刻度，去掉多余的0"""
    if x == int(x):
        return f'{int(x)}'
    else:
        s = f'{x:.2f}'
        if s.endswith('.00'):
            return s[:-3]
        elif s.endswith('0'):
            return s.rstrip('0').rstrip('.') if '.' in s else s
        else:
            return s


###################################################################################
##### The property of drawing graphs - 按照标准修改颜色和线型 #####
# 模型颜色和线型配置
# 按照标准：PIWT红色点线，PINN棕色点划线，PIKAN蓝色虚线
model_properties = {
    'PIWF': {'base_color': '#ff0000', 'linestyle': ':', 'name': 'PIWF'},  # 红色点线
    'PINN': {'base_color': '#8c564b', 'linestyle': '-.', 'name': 'PINN'},  # 棕色点划线
    'PIKAN': {'base_color': '#1f77b4', 'linestyle': '--', 'name': 'PIKAN'}  # 蓝色虚线
}

# 线宽设置
model_linewidth = {
    'PIWF': 2.5,
    'PINN': 2.5,
    'PIKAN': 2.5
}


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
    padded_arr = np.pad(arr, (half_window, half_window), mode='reflect')
    smoothed = moving_average_convolve(padded_arr, window_size)
    return smoothed


if __name__ == "__main__":
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    fig_Path = os.path.join(cwd, "figure_loss")
    g_suffix = ["svg", "pdf", "png"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
    ###################################################################################

    # 加载所有损失数据
    print("📂 加载损失数据...")

    loss_pinn_train = np.load("./[3, 128, 128, 128, 128, 3]layer_LBFGS_train_loss.npy")
    loss_piwt_train = np.load("./4layer_LBFGS_train_loss.npy")
    loss_pikan_train = np.load("./pikanlayer_LBFGS_train_loss.npy")

    loss_pinn_test = np.load("./[3, 128, 128, 128, 128, 3]layer_LBFGS_test_loss.npy")
    loss_piwt_test = np.load("./4layer_LBFGS_test_loss.npy")
    loss_pikan_test = np.load("./pikanlayer_LBFGS_test_loss.npy")

    # 检查数据长度是否一致，如果不同则截断到最小长度
    min_length = min(len(loss_pinn_train), len(loss_pinn_test),
                     len(loss_piwt_train), len(loss_piwt_test),
                     len(loss_pikan_train), len(loss_pikan_test))

    if min_length > 0:
        print(f"⚙️ 统一数据长度为: {min_length}")
        loss_pinn_train = loss_pinn_train[:min_length]
        loss_pinn_test = loss_pinn_test[:min_length]
        loss_piwt_train = loss_piwt_train[:min_length]
        loss_piwt_test = loss_piwt_test[:min_length]
        loss_pikan_train = loss_pikan_train[:min_length]
        loss_pikan_test = loss_pikan_test[:min_length]

    ###################################################################################
    # 应用平滑处理
    print("🔄 应用平滑处理...")
    window_size = 49

    curves_to_smooth = [
        (loss_pinn_train, "PINN训练"),
        (loss_pinn_test, "PINN测试"),
        (loss_piwt_train, "PIWT训练"),
        (loss_piwt_test, "PIWT测试"),
        (loss_pikan_train, "PIKAN训练"),
        (loss_pikan_test, "PIKAN测试")
    ]

    smoothed_curves = []
    for curve, name in curves_to_smooth:
        if len(curve) > 0:
            try:
                smoothed = apply_edge_padding(curve, window_size)
                if len(smoothed) != len(curve):
                    smoothed = curve
                smoothed_curves.append(smoothed)
            except Exception as e:
                smoothed_curves.append(curve)
        else:
            smoothed_curves.append(np.array([]))

    curve_pinn_train_s, curve_pinn_test_s, curve_piwt_train_s, curve_piwt_test_s, curve_pikan_train_s, curve_pikan_test_s = smoothed_curves

    ###################################################################################
    # 创建两个子图 (1行2列，左右排列) - 不共享y轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=600)

    # 调整子图间距
    plt.subplots_adjust(wspace=0.08, left=0.08, right=0.95, top=0.92, bottom=0.12)

    # 共同的x轴范围
    x_limit = [0, 100000]

    # 共同的y轴范围
    y_limit = [1e-9, 1]

    # ==================== 左边子图: Training Loss ====================
    ax = ax1

    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_visible(True)

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', size=7, width=2.5, direction='in',
                   top=False, right=False, labelsize=fontSize - 2)
    ax.tick_params(axis='both', which='minor', size=4, width=2.0, direction='in',
                   top=False, right=False)
    ax.tick_params(axis='x', pad=10, which='both')
    ax.tick_params(axis='y', pad=10, which='both')

    # 设置坐标轴范围
    ax.set_xlim(x_limit[0], x_limit[1])
    ax.set_ylim(y_limit[0], y_limit[1])

    # 设置x轴刻度
    x_major_ticks = np.arange(x_limit[0], 100001, 20000)
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

    ax.set_xticklabels(x_tick_labels, fontfamily='Times New Roman', fontsize=fontSize - 2)

    # x轴次要刻度
    x_minor_ticks = np.arange(x_limit[0], 100001, 4000)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=4, width=1.5, labelbottom=False)

    # 设置y轴为对数坐标
    ax.set_yscale('log')

    # 设置y轴主要刻度
    y_major_ticks = [1e-9, 1e-7, 1e-5, 1e-3, 1]
    ax.set_yticks(y_major_ticks)

    # 设置y轴次要刻度
    y_minor_ticks = []
    full_major_ticks = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    for i in range(len(full_major_ticks) - 1):
        start = np.log10(full_major_ticks[i])
        end = np.log10(full_major_ticks[i + 1])
        for factor in [2, 3, 4, 5, 6, 7, 8, 9]:
            minor_tick = factor * 10 ** start
            if minor_tick < full_major_ticks[i + 1] * 0.99:
                y_minor_ticks.append(minor_tick)

    ax.set_yticks(y_minor_ticks, minor=True)
    ax.tick_params(axis='y', which='minor', length=3, width=1.2)


    # 格式化y轴标签
    def log_format(x, pos):
        power = int(np.floor(np.log10(x)))
        if power == 0:
            return '$10^{0}$'
        else:
            return f'$10^{{{power}}}$'


    ax.yaxis.set_major_formatter(FuncFormatter(log_format))

    # 确保刻度只在底部和左侧
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # 添加坐标轴标签
    ax.set_xlabel(r"Epoch ($\times 10^4$)", fontsize=fontSize, fontweight='normal',
                  fontfamily='Times New Roman', labelpad=12)
    ax.set_ylabel(r"Loss", fontsize=fontSize, fontweight='normal',
                  fontfamily='Times New Roman', labelpad=12)

    # 绘制Training Loss曲线 - 按照标准顺序：PIWT, PINN, PIKAN
    train_curves = [(curve_pinn_train_s, model_properties['PINN']),  # PINN: 棕色点划线
        (curve_pikan_train_s, model_properties['PIKAN']),  # PIKAN: 蓝色虚线
        (curve_piwt_train_s, model_properties['PIWF'])  # PIWT: 红色点线

    ]

    for curve_data, props in train_curves:
        if len(curve_data) > 0:
            x_data = np.linspace(0, x_limit[1], len(curve_data))
            ax.plot(x_data, curve_data,
                    color=props['base_color'],
                    linestyle=props['linestyle'],
                    linewidth=model_linewidth[props['name']],
                    alpha=0.85,
                    label=props['name'])

    # 添加图例（只保留模型名称）- 按照标准顺序
    legend_elements = []
    for model_name in ['PINN', 'PIKAN', 'PIWF']:
        props = model_properties[model_name]
        legend_elements.append(Line2D([0], [0],
                                      color=props['base_color'],
                                      linewidth=model_linewidth[model_name],
                                      linestyle=props['linestyle'],
                                      label=props['name']))

    legend_font = {'family': 'Times New Roman', 'size': fontSize - 8, 'weight': 'normal'}
    legend = ax.legend(handles=legend_elements,
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

    # 在左边子图左下角添加注释(a)
    ax.text(0.02, 0.02, r'(a)', transform=ax.transAxes,
            fontsize=fontSize, ha='left', va='bottom',
            fontweight='normal', fontfamily='Times New Roman')

    # ==================== 右边子图: Testing Loss ====================
    ax = ax2

    # 设置边框线宽
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_visible(True)

    # 设置刻度参数
    ax.tick_params(axis='both', which='major', size=7, width=2.5, direction='in',
                   top=False, right=False, labelsize=fontSize - 2)
    ax.tick_params(axis='both', which='minor', size=4, width=2.0, direction='in',
                   top=False, right=False)
    ax.tick_params(axis='x', pad=10, which='both')
    ax.tick_params(axis='y', pad=10, which='both')

    # 设置坐标轴范围
    ax.set_xlim(x_limit[0], x_limit[1])
    ax.set_ylim(y_limit[0], y_limit[1])

    # 设置x轴刻度
    ax.set_xticks(x_major_ticks)
    ax.set_xticklabels(x_tick_labels, fontfamily='Times New Roman', fontsize=fontSize - 2)

    # x轴次要刻度
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=4, width=1.5, labelbottom=False)

    # 设置y轴为对数坐标
    ax.set_yscale('log')

    # 设置y轴主要刻度（但不显示标签）
    ax.set_yticks(y_major_ticks)

    # 设置y轴次要刻度
    ax.set_yticks(y_minor_ticks, minor=True)
    ax.tick_params(axis='y', which='minor', length=3, width=1.2)

    # 隐藏y轴标签（右边子图不显示y轴数字标签）
    ax.set_yticklabels([])

    # 移除y轴标签文字
    ax.set_ylabel('')

    # 确保刻度只在底部和左侧
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # 添加x轴标签
    ax.set_xlabel(r"Epoch ($\times 10^4$)", fontsize=fontSize, fontweight='normal',
                  fontfamily='Times New Roman', labelpad=12)

    # 绘制Testing Loss曲线 - 按照标准顺序：PIWT, PINN, PIKAN
    test_curves = [
        (curve_piwt_test_s, model_properties['PIWF']),  # PIWT: 红色点线
        (curve_pinn_test_s, model_properties['PINN']),  # PINN: 棕色点划线
        (curve_pikan_test_s, model_properties['PIKAN'])  # PIKAN: 蓝色虚线
    ]

    for curve_data, props in test_curves:
        if len(curve_data) > 0:
            x_data = np.linspace(0, x_limit[1], len(curve_data))
            ax.plot(x_data, curve_data,
                    color=props['base_color'],
                    linestyle=props['linestyle'],
                    linewidth=model_linewidth[props['name']],
                    alpha=0.95)

    # 在右边子图左下角添加注释(b)
    ax.text(0.02, 0.02, r'(b)', transform=ax.transAxes,
            fontsize=fontSize, ha='left', va='bottom',
            fontweight='normal', fontfamily='Times New Roman')

    ###################################################################################
    # 保存图形
    print("💾 保存图形...")
    for suffix in g_suffix:
        figName = f"trainloss_two_subplots_horizontal.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=600, bbox_inches='tight', pad_inches=0.1)
        print(f"  ✅ 保存: {figFile}")

    plt.close()
    print("🎉 完成!")