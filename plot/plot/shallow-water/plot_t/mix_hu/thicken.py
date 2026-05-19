import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fontSize = 29

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

def plot_zoomed_subplots():
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    fig_Path = os.path.join(cwd, "figure_zoomed")
    g_suffix = ["png", "pdf"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
        ###################################################################################

    # 加载数据 - 添加FD和PIKAN数据
    data = np.load("../hu_true.npy").flatten().reshape(-1, 1)
    data3 = np.load("../hu_pred_PINN.npy").flatten().reshape(-1, 1)
    data4 = np.load("../hu_pred_PIFAN.npy").flatten().reshape(-1, 1)
    data_FD = np.load("../hu_pred_FD.npy").flatten().reshape(-1, 1)
    data_PIKAN = np.load("../hu_pred_PIKAN.npy").flatten().reshape(-1, 1)

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

    # 定义要绘制的时间点和对应的放大范围
    # 格式: (时间点, x范围, y范围, 输出文件名前缀)
    zoom_configs = [
        (900, [200, 300], [-0.03, 0.05], "zoom_t900"),
        (1800, [500, 550], [-0.03, 0.05], "zoom_t1800"),
        (2700, [750, 820], [-0.03, 0.05], "zoom_t2700")
    ]

    # 按照参考代码的样式设置
    plt.rcParams.update({
        'font.size': fontSize,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'axes.linewidth': 1.2,
    })

    for t_point, x_range, y_range, filename_prefix in zoom_configs:
        # 为每个时间点创建单独的图形 - 使用窄长的比例 (8, 4) 或类似
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        # 提取当前时间点的数据
        curve_true = data[:, 2][data[:, 1] == t_point]
        curve_pinn = data3[:, 2][data3[:, 1] == t_point]
        curve_pifan = data4[:, 2][data4[:, 1] == t_point]
        curve_FD = data_FD[:, 2][data_FD[:, 1] == t_point]
        curve_PIKAN = data_PIKAN[:, 2][data_PIKAN[:, 1] == t_point]

        # 找到在x范围内的数据点索引
        x_mask = (x >= x_range[0]) & (x <= x_range[1])
        x_zoomed = x[x_mask]

        # 计算步长用于DNS空心圆圈
        step = max(1, len(x_zoomed) // 8)

        # 绘制放大区域的曲线 - 严格按照参考代码的顺序和线型，通过zorder控制图层
        # 1. DNS: 黑色空心圆圈 (zorder=1, 最底层)
        ax.plot(x_zoomed[::step], curve_true[x_mask][::step], 'o', color='#000000', linestyle='', markersize=5,
                markerfacecolor='none', markeredgewidth=1.2, zorder=1)

        # 2. FD: 绿色实线 (zorder=2)
        ax.plot(x_zoomed, curve_FD[x_mask], color='#2ca02c', linestyle='-', lw=2.5, zorder=2)

        # 3. PINN: 棕色点划线 (zorder=3)
        ax.plot(x_zoomed, curve_pinn[x_mask], color='#8c564b', linestyle='-.', lw=2.5, zorder=3)

        # 4. PIKAN: 蓝色虚线 (zorder=4)
        ax.plot(x_zoomed, curve_PIKAN[x_mask], color='#1f77b4', linestyle='--', lw=2.5, zorder=4)

        # 5. PIWT: 红色点线 (zorder=5, 最上层)
        ax.plot(x_zoomed, curve_pifan[x_mask], color='#ff0000', linestyle=':', lw=2.5, zorder=5)

        # 设置坐标轴范围
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        # 移除坐标轴标签
        ax.set_xlabel('')
        ax.set_ylabel('')

        # 设置刻度参数
        ax.tick_params(axis='both', which='major', size=6, width=1.2, direction='in', labelsize=fontSize)
        ax.tick_params(axis='both', which='minor', size=3, width=1.2, direction='in')

        # 设置边框线宽
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        # ========== x轴刻度设置 ==========
        x_span = x_range[1] - x_range[0]

        # 根据不同的x范围设置主刻度间隔
        if t_point == 900:  # t=900: 范围200-300，间隔20
            x_major_step = 20
        elif t_point == 1800:  # t=1800: 范围500-550，间隔10
            x_major_step = 10
        else:  # t=2700: 范围750-820，间隔10
            x_major_step = 10

        # 确保刻度包含两端点
        x_major_ticks = [x_range[0]]
        current_tick = x_range[0] + x_major_step
        while current_tick < x_range[1] - 0.001:
            x_major_ticks.append(current_tick)
            current_tick += x_major_step
        x_major_ticks.append(x_range[1])

        # 去重并排序
        x_major_ticks = sorted(list(set(x_major_ticks)))

        ax.set_xticks(x_major_ticks)

        # x轴次要刻度 - 间隔为主刻度的1/5
        x_minor_step = x_major_step / 5
        x_minor_ticks = np.arange(x_range[0], x_range[1] + 0.001, x_minor_step)
        ax.set_xticks(x_minor_ticks, minor=True)

        # ========== y轴刻度设置 - 减少一半标签 ==========
        y_span = y_range[1] - y_range[0]

        # y轴主刻度间隔设置为0.02（原来是0.01，现在加倍，减少标签数量）
        y_major_step = 0.02

        # 确保刻度包含两端点
        y_major_ticks = [y_range[0]]
        current_tick = y_range[0] + y_major_step
        while current_tick < y_range[1] - 0.0001:
            y_major_ticks.append(current_tick)
            current_tick += y_major_step
        y_major_ticks.append(y_range[1])

        # 去重并排序
        y_major_ticks = sorted(list(set(y_major_ticks)))

        ax.set_yticks(y_major_ticks)

        # y轴次要刻度 - 间隔为主刻度的1/5 (0.004)
        y_minor_step = y_major_step / 5
        y_minor_ticks = np.arange(y_range[0], y_range[1] + 0.0001, y_minor_step)
        ax.set_yticks(y_minor_ticks, minor=True)

        # ========== 格式化刻度标签 ==========
        # x轴标签格式化
        x_tick_labels = []
        for tick in x_major_ticks:
            if abs(tick - round(tick)) < 0.001:
                x_tick_labels.append(f"{int(round(tick))}")
            else:
                label = f"{tick:.1f}"
                if '.' in label:
                    label = label.rstrip('0').rstrip('.')
                x_tick_labels.append(label)
        ax.set_xticklabels(x_tick_labels)

        # y轴标签格式化 - 显示两位小数
        y_tick_labels = []
        for tick in y_major_ticks:
            if abs(tick) < 0.0001:
                y_tick_labels.append("0")
            else:
                label = f"{tick:.2f}"
                if '.' in label:
                    label = label.rstrip('0').rstrip('.')
                y_tick_labels.append(label)
        ax.set_yticklabels(y_tick_labels)

        # 确保所有刻度标签都可见
        ax.tick_params(axis='x', which='both', labelbottom=True, pad=8)
        ax.tick_params(axis='y', which='both', labelleft=True, pad=8)

        # 设置边距确保标签完全显示
        plt.subplots_adjust(left=0.12, right=0.98, bottom=0.15, top=0.95)

        # 保存图形
        for suffix in g_suffix:
            figName = f"{filename_prefix}.{suffix}"
            figFile = os.path.join(fig_Path, figName)
            plt.savefig(figFile, dpi=300, bbox_inches='tight')
            print(f"✅ 放大图表已保存: {figFile}")
            print(f"   x范围: {x_range}, y范围: {y_range}")

        plt.close(fig)


if __name__ == "__main__":
    plot_zoomed_subplots()