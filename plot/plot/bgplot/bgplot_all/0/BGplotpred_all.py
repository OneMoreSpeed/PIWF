import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fontSize = 24
# 计算放大20%后的字体大小（用于除了t注释外的所有字体）
enlarged_fontSize = int(fontSize * 1.2)  # 24 * 1.2 = 28.8，取整为29

########## font: Time New Roman ##########
config = {
    "text.usetex": False,
    "font.family": "Times New Roman",
    "font.size": enlarged_fontSize,  # 默认字体大小改为放大后的
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
        # 对于小数，去掉末尾的0
        s = f'{x:.2f}'
        if s.endswith('.00'):
            return s[:-3]
        elif s.endswith('0'):
            return s.rstrip('0').rstrip('.') if '.' in s else s
        else:
            return s


###################################################################################

##### The property of drawing graphs #####
# 为不同模型设置基础颜色和线型 - 修改颜色
model_properties = {
    'PIWF': {'base_color': '#ff0000', 'linestyle': ':', 'name': 'PIWF'},  # 紫色实线
    'PINN': {'base_color': '#8c564b', 'linestyle': '-.', 'name': 'PINN'},  # 棕色虚线
    'PIKAN': {'base_color': '#1f77b4', 'linestyle': '--', 'name': 'PIKAN'},  # 蓝色点划线
    'FD': {'base_color': '#2ca02c', 'linestyle': '-', 'name': 'FD'},  # 绿色点线
    'DNS': {'base_color': '#000000', 'linestyle': '-', 'name': 'DNS'}  # 黑色实线
}

# 为不同时间点设置 - 所有模型都使用这些时间点
time_points = [0, 0.2, 0.5, 0.8, 1]

# 为不同模型设置线宽
model_linewidth = {
    'PIWF': 2.5,
    'PINN': 2.5,
    'PIKAN': 2.5,
    'FD': 2.5,
    'DNS': 2.2  # DNS线宽稍增大一点
}

# DNS统一使用小圆圈标记，更密集
dns_marker = {'marker': 'o', 'markevery': 6, 'size': 5}  # 从10改为6，更密集；大小从6改为5

# 标记边缘宽度
marker_edge_width = 1.2

# x=0处垂直线的圆圈标记设置
vertical_line_marker = {'marker': 'o', 'size': 5, 'count': 20}  # 在x=0处画20个圆圈


# 为不同时间点创建颜色映射函数
def get_time_color(base_color, time):
    """根据时间调整颜色深浅 - 时间越大颜色越浅，时间越小颜色越深"""
    # 将十六进制转换为RGB
    if isinstance(base_color, str) and base_color.startswith('#'):
        base_color = base_color.lstrip('#')
        rgb = tuple(int(base_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    else:
        rgb = mcolors.to_rgb(base_color)

    # 时间范围0-1，时间越小颜色越深，时间越大颜色越浅
    # 浅色因子：时间0时1.0（原始颜色最深），时间1时0.3（最浅）
    light_factor = 1.0 - 0.7 * time  # time=0时light_factor=1.0, time=1时light_factor=0.3

    # 混合白色来调浅颜色
    mixed_rgb = [1.0 - light_factor + light_factor * c for c in rgb]

    return mixed_rgb


###################################################################################


if __name__ == "__main__":
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    pdir1 = os.path.dirname(cwd)
    pdir2 = os.path.dirname(pdir1)
    filePath = os.path.join(cwd)
    workPath = os.path.join(cwd)
    fig_Path = os.path.join(cwd, "figure")
    g_suffix = ["png", "pdf"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
    ###################################################################################

    # ==================== 第一张图的数据 (ν=0.01/π) ====================
    models1 = [
        ("DNS", "u_trueBG.npy"),
        ("FD", "u_predBG_FD.npy"),
        ("PINN", "u_predBG_PINN.npy"),
        ("PIKAN", "u_predBG_PIKAN.npy"),
        ("PIWF", "u_predBG_PIWT.npy"),

    ]

    # 存储第一张图的数据
    model_data1 = {}

    # 加载第一张图的数据
    print("=" * 50)
    print("加载第一张图数据 (ν=0.01/π)...")
    for model_name, filename in models1:
        try:
            data = np.load(f"../{filename}")
            print(f"成功加载 {model_name} 数据，形状: {data.shape}")

            # 创建 x 和 t 的值
            x = np.linspace(-1, 1, 256)
            t = np.linspace(0, 1, 100)

            # 生成网格
            X, T = np.meshgrid(x, t)

            # 将 x 和 t 的信息 concatenate 到数据中
            data_with_coords = np.concatenate([X[..., np.newaxis], T[..., np.newaxis], data[..., np.newaxis]], axis=2)
            model_data1[model_name] = data_with_coords
            print(f"  {model_name} 最终数据形状: {data_with_coords.shape}")

        except FileNotFoundError:
            print(f"警告: 文件 {filename} 未找到，跳过 {model_name}")
            continue
        except Exception as e:
            print(f"处理 {model_name} 时出错: {e}")
            continue

    # ==================== 第二张图的数据 (ν=0.001) ====================
    models2 = [
        ("DNS", "u_trueBG.npy"),
        ("FD", "u_predBG_FD1000.npy "),
        ("PINN", "u_predBG_PINN1000.npy"),
        ("PIKAN", "u_predBG_PIKAN1000.npy"),
        ("PIWF", "u_predBG_PIWT1000.npy"),

    ]

    # 存储第二张图的数据
    model_data2 = {}

    # 加载第二张图的数据
    print("\n" + "=" * 50)
    print("加载第二张图数据 (ν=0.001)...")
    for model_name, filename in models2:
        try:
            data = np.load(f"../{filename}")
            print(f"成功加载 {model_name} 数据，形状: {data.shape}")

            # 创建 x 和 t 的值
            x = np.linspace(-1, 1, 256)
            t = np.linspace(0, 1, 100)

            # 生成网格
            X, T = np.meshgrid(x, t)

            # 将 x 和 t 的信息 concatenate 到数据中
            data_with_coords = np.concatenate([X[..., np.newaxis], T[..., np.newaxis], data[..., np.newaxis]], axis=2)
            model_data2[model_name] = data_with_coords
            print(f"  {model_name} 最终数据形状: {data_with_coords.shape}")

        except FileNotFoundError:
            print(f"警告: 文件 {filename} 未找到，跳过 {model_name}")
            continue
        except Exception as e:
            print(f"处理 {model_name} 时出错: {e}")
            continue

    # 检查第二张图是否成功加载了PIKAN
    print("\n" + "=" * 50)
    print("第二张图加载的模型:")
    for model_name in model_data2.keys():
        print(f"  - {model_name}")
    print("=" * 50)

    ###################################################################################
    # 创建上下拼合的图形 (2行1列)，保持原有图的大小比例
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15), dpi=600)  # 高度加倍，保持宽度不变

    ###################################################################################
    # 第一张图 - 完全保持原有样式（字体放大20%）
    ###################################################################################
    ax = ax1

    # 完全复制第一张图的代码
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['top'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)

    ax.xaxis.set_tick_params(which='major', size=10, width=2.5, direction='in', top=False)
    ax.xaxis.set_tick_params(which='minor', size=6, width=2.0, direction='in', top=False)
    ax.yaxis.set_tick_params(which='major', size=10, width=2.5, direction='in', right=False)
    ax.yaxis.set_tick_params(which='minor', size=6, width=2.0, direction='in', right=False)
    ax.xaxis.set_tick_params(pad=8)
    ax.yaxis.set_tick_params(pad=8)

    xlimit = [-1, 1]
    ylimit = [-1, 1]
    ax.set_xlim(xlimit[0], xlimit[1])
    ax.set_ylim(ylimit[0], ylimit[1])

    x_major_ticks = np.arange(xlimit[0], xlimit[1] + 0.1, 0.5)
    ax.set_xticks(x_major_ticks)
    y_major_ticks = np.arange(ylimit[0], ylimit[1] + 0.1, 0.5)
    ax.set_yticks(y_major_ticks)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

    x_minor_ticks = np.arange(xlimit[0], xlimit[1], 0.1)
    ax.set_xticks(x_minor_ticks, minor=True)
    y_minor_ticks = np.arange(ylimit[0], ylimit[1], 0.1)
    ax.set_yticks(y_minor_ticks, minor=True)

    ax.tick_params(axis='both', which='minor', length=6, color='black', width=2.0, labelbottom=False, labelleft=False)

    # ax.set_xlabel(r"$x$", fontsize=fontSize)  # 注释掉x轴标签
    ax.set_ylabel(r"$u$", fontsize=enlarged_fontSize)  # 放大20%
    ax.set_xticklabels([])

    # 为每个时间点绘制所有模型的曲线
    for time in time_points:
        for model_name in model_data1.keys():
            if model_name not in model_data1:
                continue

            data = model_data1[model_name]

            diff = np.abs(data[:, :, 1] - time)
            closest_idx = np.unravel_index(np.argmin(diff), diff.shape)

            x_vals = data[closest_idx[0], :, 0]
            u_vals = data[closest_idx[0], :, 2]

            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            u_sorted = u_vals[sort_idx]

            base_color = model_properties[model_name]['base_color']
            time_color = get_time_color(base_color, time)

            if model_name == "DNS":
                marker_config = dns_marker

                n_points = len(x_sorted)
                mask = np.zeros(n_points, dtype=bool)

                for i in range(0, n_points, marker_config['markevery']):
                    mask[i] = True

                mask[0] = True
                mask[-1] = True

                marker_x = x_sorted[mask]
                marker_u = u_sorted[mask]

                ax.scatter(marker_x, marker_u,
                           color=time_color,
                           marker=marker_config['marker'],
                           s=marker_config['size'] ** 2,
                           facecolors='none',
                           edgecolors=time_color,
                           linewidths=marker_edge_width,
                           zorder=4)
            else:
                ax.plot(x_sorted, u_sorted,
                        color=time_color,
                        linewidth=model_linewidth[model_name],
                        linestyle=model_properties[model_name]['linestyle'],
                        zorder=3)

    # 在曲线上标注时间 - 去掉白色背景
    if "DNS" in model_data1:
        for i, time in enumerate(time_points):
            data = model_data1["DNS"]
            diff = np.abs(data[:, :, 1] - time)
            closest_idx = np.unravel_index(np.argmin(diff), diff.shape)

            x_vals = data[closest_idx[0], :, 0]
            u_vals = data[closest_idx[0], :, 2]

            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            u_sorted = u_vals[sort_idx]

            if time == 0:
                label_x = -0.8
            elif time == 0.2:
                label_x = -0.75
            elif time == 0.5:
                label_x = -0.7
            elif time == 0.8:
                label_x = -0.67
            elif time == 1:
                label_x = -0.62

            label_idx = np.argmin(np.abs(x_sorted - label_x))
            label_y = u_sorted[label_idx]

            if time == 0:
                y_offset = 0.17
            elif time == 0.2:
                y_offset = 0.17
            elif time == 0.5:
                y_offset = 0.13
            elif time == 0.8:
                y_offset = 0.1
            elif time == 1:
                y_offset = -0.02

            if time == 0:
                rotation_angle = 45
            elif time == 0.2:
                rotation_angle = 45
            elif time == 0.5:
                rotation_angle = 40
            elif time == 0.8:
                rotation_angle = 25
            elif time == 1:
                rotation_angle = 25

            # t注释去掉白色背景
            ax.text(label_x, label_y + y_offset, f'$t={time}$',
                    fontsize=fontSize,  # 保持原来的大小，不放大
                    color='black',
                    ha='left', va='center',
                    rotation=rotation_angle,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor='none'))  # 去掉白色背景

    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.8)

    # 在第一张图左下角添加标签 - 放大20%
    ax.text(-0.98, -0.9, '(a) $\\nu=0.01/\\pi$ ',
            fontsize=enlarged_fontSize - 2,  # 基于放大后的字体调整
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'))

    ###################################################################################
    # 创建合并的图例 - 放大20%
    ###################################################################################
    legend_elements = []

    for model_name in ['DNS', 'FD', 'PINN', 'PIKAN', 'PIWF']:
        props = model_properties[model_name]
        rep_color = get_time_color(props['base_color'], 0.5)

        if model_name == "DNS":
            legend_elements.append(Line2D([0], [0],
                                          color='none',
                                          marker=dns_marker['marker'],
                                          markersize=dns_marker['size'],
                                          markerfacecolor='none',
                                          markeredgecolor=rep_color,
                                          markeredgewidth=marker_edge_width,
                                          linestyle='None',
                                          label=props['name']))
        else:
            legend_elements.append(Line2D([0], [0],
                                          color=rep_color,
                                          linewidth=model_linewidth[model_name],
                                          linestyle=props['linestyle'],
                                          label=props['name']))

    legend = ax1.legend(handles=legend_elements,
                        loc='center right',
                        bbox_to_anchor=(0.98, 0.76),
                        frameon=False,
                        fancybox=False,
                        edgecolor='black',
                        facecolor='white',
                        framealpha=0.95,
                        fontsize=enlarged_fontSize - 4,  # 放大20%
                        handlelength=2.0,
                        labelspacing=0.4,
                        ncol=1)

    ###################################################################################
    # 第二张图 - 完全保持原有样式（包括x=0处的加密圆圈和PIKAN结果，字体放大20%）
    ###################################################################################
    ax = ax2

    # 完全复制第二张图的代码
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['top'].set_linewidth(2.5)
    ax.spines['right'].set_linewidth(2.5)

    ax.xaxis.set_tick_params(which='major', size=10, width=2.5, direction='in', top=False)
    ax.xaxis.set_tick_params(which='minor', size=6, width=2.0, direction='in', top=False)
    ax.yaxis.set_tick_params(which='major', size=10, width=2.5, direction='in', right=False)
    ax.yaxis.set_tick_params(which='minor', size=6, width=2.0, direction='in', right=False)
    ax.xaxis.set_tick_params(pad=8)
    ax.yaxis.set_tick_params(pad=8)

    xlimit = [-1, 1]
    ylimit = [-1, 1]
    ax.set_xlim(xlimit[0], xlimit[1])
    ax.set_ylim(ylimit[0], ylimit[1])

    x_major_ticks = np.arange(xlimit[0], xlimit[1] + 0.1, 0.5)
    ax.set_xticks(x_major_ticks)
    y_major_ticks = np.arange(ylimit[0], ylimit[1] + 0.1, 0.5)
    ax.set_yticks(y_major_ticks)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

    x_minor_ticks = np.arange(xlimit[0], xlimit[1], 0.1)
    ax.set_xticks(x_minor_ticks, minor=True)
    y_minor_ticks = np.arange(ylimit[0], ylimit[1], 0.1)
    ax.set_yticks(y_minor_ticks, minor=True)

    ax.tick_params(axis='both', which='minor', length=6, color='black', width=2.0, labelbottom=False, labelleft=False)

    ax.set_xlabel(r"$x$", fontsize=enlarged_fontSize)  # 放大20%
    ax.set_ylabel(r"$u$", fontsize=enlarged_fontSize)  # 放大20%

    # 为每个时间点绘制所有模型的曲线
    for time in time_points:
        for model_name in model_data2.keys():
            if model_name not in model_data2:
                continue

            data = model_data2[model_name]

            diff = np.abs(data[:, :, 1] - time)
            closest_idx = np.unravel_index(np.argmin(diff), diff.shape)

            x_vals = data[closest_idx[0], :, 0]
            u_vals = data[closest_idx[0], :, 2]

            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            u_sorted = u_vals[sort_idx]

            base_color = model_properties[model_name]['base_color']
            time_color = get_time_color(base_color, time)

            if model_name == "DNS":
                marker_config = dns_marker

                n_points = len(x_sorted)
                mask = np.zeros(n_points, dtype=bool)

                for i in range(0, n_points, marker_config['markevery']):
                    mask[i] = True

                # 排除x=0附近的点（这些点将由专门的垂直线上的圆圈表示）
                exclude_region = [-0.05, 0.05]
                exclude_indices = np.where((x_sorted >= exclude_region[0]) & (x_sorted <= exclude_region[1]))[0]
                for idx in exclude_indices:
                    mask[idx] = False

                if x_sorted[0] < exclude_region[0]:
                    mask[0] = True
                if x_sorted[-1] > exclude_region[1]:
                    mask[-1] = True

                marker_x = x_sorted[mask]
                marker_u = u_sorted[mask]

                ax.scatter(marker_x, marker_u,
                           color=time_color,
                           marker=marker_config['marker'],
                           s=marker_config['size'] ** 2,
                           facecolors='none',
                           edgecolors=time_color,
                           linewidths=marker_edge_width,
                           zorder=4)

                # 在x=0处绘制垂直的圆圈线
                if len(exclude_indices) > 0:
                    u_at_x0 = u_sorted[exclude_indices]
                    u_min, u_max = np.min(u_at_x0), np.max(u_at_x0)

                    vertical_y = np.linspace(u_min, u_max, vertical_line_marker['count'])
                    vertical_x = np.zeros_like(vertical_y)

                    ax.scatter(vertical_x, vertical_y,
                               color=time_color,
                               marker=vertical_line_marker['marker'],
                               s=vertical_line_marker['size'] ** 2,
                               facecolors='none',
                               edgecolors=time_color,
                               linewidths=marker_edge_width,
                               zorder=5)
            else:
                ax.plot(x_sorted, u_sorted,
                        color=time_color,
                        linewidth=model_linewidth[model_name],
                        linestyle=model_properties[model_name]['linestyle'],
                        zorder=3)

    # 在曲线上标注时间 - 去掉白色背景
    if "DNS" in model_data2:
        for i, time in enumerate(time_points):
            data = model_data2["DNS"]
            diff = np.abs(data[:, :, 1] - time)
            closest_idx = np.unravel_index(np.argmin(diff), diff.shape)

            x_vals = data[closest_idx[0], :, 0]
            u_vals = data[closest_idx[0], :, 2]

            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            u_sorted = u_vals[sort_idx]

            if time == 0:
                label_x = -0.8
            elif time == 0.2:
                label_x = -0.75
            elif time == 0.5:
                label_x = -0.7
            elif time == 0.8:
                label_x = -0.67
            elif time == 1:
                label_x = -0.62

            label_idx = np.argmin(np.abs(x_sorted - label_x))
            label_y = u_sorted[label_idx]

            if time == 0:
                y_offset = 0.17
            elif time == 0.2:
                y_offset = 0.17
            elif time == 0.5:
                y_offset = 0.13
            elif time == 0.8:
                y_offset = 0.1
            elif time == 1:
                y_offset = -0.02

            if time == 0:
                rotation_angle = 45
            elif time == 0.2:
                rotation_angle = 45
            elif time == 0.5:
                rotation_angle = 40
            elif time == 0.8:
                rotation_angle = 25
            elif time == 1:
                rotation_angle = 25

            # t注释去掉白色背景
            ax.text(label_x, label_y + y_offset, f'$t={time}$',
                    fontsize=fontSize,  # 保持原来的大小，不放大
                    color='black',
                    ha='left', va='center',
                    rotation=rotation_angle,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="none", edgecolor='none'))  # 去掉白色背景

    ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.8)

    # 在第二张图左下角添加标签 - 放大20%
    ax.text(-0.98, -0.9, '(b) $\\nu=0.001$ ',
            fontsize=enlarged_fontSize - 2,  # 基于放大后的字体调整
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'))

    # ==================== 为第二张图添加放大子图（修改范围） ====================
    # 创建子图，位置在右上角
    inset_ax2 = ax.inset_axes([0.65, 0.65, 0.32, 0.32])

    # 设置子图的范围 - 修改为x:0.8-1, y:-0.5-0
    x_zoom = [0.9, 1]
    y_zoom = [-0.3, 0]
    inset_ax2.set_xlim(x_zoom)
    inset_ax2.set_ylim(y_zoom)
    # 添加这两行来防止自动扩展
    inset_ax2.set_xlim(x_zoom)  # 再次设置确保
    inset_ax2.set_ylim(y_zoom)  # 再次设置确保
    inset_ax2.autoscale(enable=False)  # 关闭自动缩放
    # 设置主刻度和子刻度 - 修改这里，不要超出范围
    x_major_zoom = np.arange(x_zoom[0], x_zoom[1] + 0.001, 0.05)  # 改为+0.001而不是0.05
    inset_ax2.set_xticks(x_major_zoom)
    y_major_zoom = np.arange(y_zoom[0], y_zoom[1] + 0.001, 0.1)  # 改为+0.001而不是0.1
    inset_ax2.set_yticks(y_major_zoom)

    # 设置子刻度 - 同样修改
    x_minor_zoom = np.arange(x_zoom[0], x_zoom[1] + 0.001, 0.01)  # 改为+0.001
    inset_ax2.set_xticks(x_minor_zoom, minor=True)
    y_minor_zoom = np.arange(y_zoom[0], y_zoom[1] + 0.001, 0.02)  # 改为+0.001
    inset_ax2.set_yticks(y_minor_zoom, minor=True)

    # 强制设置坐标轴范围（再次确认）
    inset_ax2.set_xlim(x_zoom)
    inset_ax2.set_ylim(y_zoom)

    # 设置刻度样式
    inset_ax2.tick_params(axis='both', which='major', direction='in', length=6, width=1.5,
                          labelsize=enlarged_fontSize - 6)
    inset_ax2.tick_params(axis='both', which='minor', direction='in', length=3, width=1.0)
    inset_ax2.xaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))
    inset_ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

    # 设置边框线宽
    for spine in inset_ax2.spines.values():
        spine.set_linewidth(1.5)

    # 在子图中绘制所有模型的曲线
    for time in time_points:
        for model_name in model_data2.keys():
            if model_name not in model_data2:
                continue

            data = model_data2[model_name]
            diff = np.abs(data[:, :, 1] - time)
            closest_idx = np.unravel_index(np.argmin(diff), diff.shape)

            x_vals = data[closest_idx[0], :, 0]
            u_vals = data[closest_idx[0], :, 2]

            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            u_sorted = u_vals[sort_idx]

            # 只绘制x在0.8-1范围内的数据
            mask = (x_sorted >= x_zoom[0]) & (x_sorted <= x_zoom[1])
            x_zoom_data = x_sorted[mask]
            u_zoom_data = u_sorted[mask]

            if len(x_zoom_data) == 0:
                continue

            base_color = model_properties[model_name]['base_color']
            time_color = get_time_color(base_color, time)

            if model_name == "DNS":
                # DNS用圆圈标记
                inset_ax2.scatter(x_zoom_data, u_zoom_data,
                                  color=time_color,
                                  marker=dns_marker['marker'],
                                  s=dns_marker['size'] ** 2,
                                  facecolors='none',
                                  edgecolors=time_color,
                                  linewidths=marker_edge_width,
                                  zorder=4)
            else:
                inset_ax2.plot(x_zoom_data, u_zoom_data,
                               color=time_color,
                               linewidth=model_linewidth[model_name] * 0.8,  # 子图中线宽稍微减小
                               linestyle=model_properties[model_name]['linestyle'],
                               zorder=3)

    # 添加网格
    inset_ax2.grid(True, alpha=0.2, linestyle=':', linewidth=0.6)

    # 添加坐标轴标签
    inset_ax2.set_xlabel(r'$x$', fontsize=enlarged_fontSize - 4)
    inset_ax2.set_ylabel(r'$u$', fontsize=enlarged_fontSize - 4)

    # 调整子图间距
    plt.subplots_adjust(hspace=0.15)

    # 保存合并后的图片
    for suffix in g_suffix:
        figName = f"Burgers_Combined_Vertical_with_inset.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, bbox_inches='tight', dpi=600)
        print(f"已保存合并图片: {figFile}")

    plt.close()
    print("\n上下拼合图片生成完成！")