import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter  # 添加Savitzky-Golay滤波器

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fontSize = 22  # 改为22，与其他图一致
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
##### 平滑处理函数 #####
def apply_smoothing(data, method='moving_average', window_size=5, polyorder=2):
    """
    对数据进行平滑处理

    参数:
    data: 输入数据数组
    method: 平滑方法，可选 'moving_average' 或 'savgol'
    window_size: 窗口大小（奇数）
    polyorder: Savitzky-Golay的多项式阶数
    """
    if len(data) < window_size:
        return data

    if method == 'moving_average':
        # 简单的移动平均
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(data, window, mode='same')

        # 处理边界效应
        half_window = window_size // 2
        # 起始部分使用前向平均
        for i in range(half_window):
            smoothed[i] = np.mean(data[:i + half_window + 1])
        # 结束部分使用后向平均
        for i in range(len(data) - half_window, len(data)):
            smoothed[i] = np.mean(data[i - half_window:])

        return smoothed

    elif method == 'savgol':
        # Savitzky-Golay滤波器，适用于保持峰值特征
        try:
            return savgol_filter(data, window_size, polyorder)
        except:
            # 如果失败，退回移动平均
            return apply_smoothing(data, 'moving_average', window_size)

    else:
        return data


###################################################################################
##### 数据检查和清理函数 #####
def check_and_clean_data(data, var_name, method_name):
    """
    检查并清理数据中的异常值

    参数:
    data: 输入数据
    var_name: 变量名称
    method_name: 方法名称

    返回:
    清理后的数据
    """
    print(f"   检查{method_name}的{var_name}数据...")

    # 记录原始形状
    original_shape = data.shape

    # 检查NaN和inf值
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    zero_count = np.sum(data == 0)
    negative_count = np.sum(data < 0)

    if nan_count > 0:
        print(f"     ⚠️ 发现{nan_count}个NaN值")
        data = np.nan_to_num(data, nan=1e-10)

    if inf_count > 0:
        print(f"     ⚠️ 发现{inf_count}个inf值")
        # 将inf替换为最大值或大数
        max_finite = np.max(data[np.isfinite(data)]) if np.any(np.isfinite(data)) else 1.0
        data = np.where(np.isinf(data), max_finite * 100, data)

    if negative_count > 0:
        print(f"     ⚠️ 发现{negative_count}个负值")
        # 对于误差数据，负值可能不合理，取绝对值
        data = np.abs(data)

    # 检查数值范围
    if np.any(data > 0):
        min_val = np.min(data[data > 0])
        max_val = np.max(data)
        mean_val = np.mean(data)
        print(f"     数值范围: {min_val:.2e} 到 {max_val:.2e}, 均值: {mean_val:.2e}")

        # 检查异常小值
        very_small_threshold = 1e-15
        very_small_count = np.sum(data < very_small_threshold)
        if very_small_count > 0:
            print(f"     ⚠️ 有{very_small_count}个值小于{very_small_threshold:.0e}")

        # 检查异常大值
        very_large_threshold = 1e10
        very_large_count = np.sum(data > very_large_threshold)
        if very_large_count > 0:
            print(f"     ⚠️ 有{very_large_count}个值大于{very_large_threshold:.0e}")

    return data


###################################################################################
##### The property of drawing graphs #####
# 按照第二个代码的线型设置修改
colors = ['#2ca02c', '#8c564b', '#1f77b4', '#ff0000']  # FD, PINN, PIKAN, PIWF
linestyles = ['-', '-.', '--', ':']  # 实线，点划线，虚线，点线
markers = ['', '', '', '']  # 全部去掉标记
line_labels = ['FD', 'PINN', 'PIKAN', 'PIWF']
line_widths = [2.5, 2.5, 2.5, 2.5]
###################################################################################


if __name__ == "__main__":
    ###################################################################################
    ############################# 平滑参数设置 #############################
    SMOOTH_METHOD = 'savgol'  # 可选: 'moving_average' 或 'savgol'
    SMOOTH_WINDOW = 7  # 窗口大小（建议奇数，如5, 7, 9）
    ###################################################################################

    ############################# file Path of input data #############################
    cwd = os.getcwd()
    filePath = os.path.join(cwd)
    fig_Path = os.path.join(cwd, "figure_errors_combined")
    g_suffix = ["pdf", "png"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
        ###################################################################################

    # 定义要绘制的变量
    variables = ['h', 'hu']  # h和hu两个变量

    # x轴数据 - 根据您的原代码是x坐标 (0-1200)
    x = np.linspace(0, 1200, 121)

    ###################################################################################
    # 修改为横向排列：1行2列，减小子图间距
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)  # 不共享y轴，因为两个变量的误差范围可能不同
    plt.subplots_adjust(wspace=0.1, hspace=0.15, top=0.92, bottom=0.18, left=0.08, right=0.95)  # wspace从0.25减小到0.1
    ###################################################################################

    # 遍历每个变量
    for idx, (ax, var) in enumerate(zip(axes, variables)):
        print(f"📂 处理变量: {var}")
        print(f"  平滑方法: {SMOOTH_METHOD}, 窗口大小: {SMOOTH_WINDOW}")

        try:
            # 加载误差数据 - 按照您的文件名格式
            print(f"  加载数据文件...")
            data_piwt = np.load(f"{var}_res_PIFAN.npy")
            data_pinn = np.load(f"{var}_res_PINN.npy")
            data_pikan = np.load(f"{var}_res_PIKAN.npy")
            data_fd = np.load(f"{var}_res_FD.npy")

            # 检查数据维度
            print(f"  数据形状: PIWT={data_piwt.shape}, PINN={data_pinn.shape}, "
                  f"PIKAN={data_pikan.shape}, FD={data_fd.shape}")

            ###################################################################################
            ##### 检查并清理原始数据 #####
            print(f"  --- 检查原始数据 ---")
            data_piwt = check_and_clean_data(data_piwt, var, "PIWT原始数据")
            data_pinn = check_and_clean_data(data_pinn, var, "PINN原始数据")
            data_pikan = check_and_clean_data(data_pikan, var, "PIKAN原始数据")
            data_fd = check_and_clean_data(data_fd, var, "FD原始数据")
            ###################################################################################

            # 处理二维数据 (121, 61) - 按照您的处理方式：计算沿x轴的RMS并除以100
            if data_piwt.ndim == 2:
                print(f"  计算RMS误差...")
                error_piwt = np.sqrt(np.mean(data_piwt ** 2, axis=1)) / 100
                error_pinn = np.sqrt(np.mean(data_pinn ** 2, axis=1)) / 100
                error_pikan = np.sqrt(np.mean(data_pikan ** 2, axis=1)) / 100

                # FD数据是一维的 (7381,)，需要特殊处理
                if data_fd.ndim == 1:
                    print(f"  FD数据是一维的，形状为{data_fd.shape}")
                    # 如果FD数据是展平的一维数组，尝试重塑为二维
                    if data_fd.size == 7381:  # 121*61 = 7381
                        try:
                            data_fd_2d = data_fd.reshape(121, 61)
                            error_fd = np.sqrt(np.mean(data_fd_2d ** 2, axis=1)) / 100
                            print(f"  FD数据成功重塑为(121, 61)")
                        except:
                            # 如果重塑失败，直接使用原始数据（假设已经是误差值）
                            print(f"  FD数据重塑失败，直接使用原始数据")
                            error_fd = data_fd / 100
                            # 如果FD数据长度不同，只取前121个点
                            if len(error_fd) > len(x):
                                error_fd = error_fd[:len(x)]
                    else:
                        # 其他情况，假设FD已经是误差序列
                        error_fd = data_fd / 100
                elif data_fd.ndim == 2:
                    error_fd = np.sqrt(np.mean(data_fd ** 2, axis=1)) / 100
                else:
                    print(f"  ❌ FD数据维度不支持: {data_fd.ndim}维")
                    continue
            else:
                print(f"  ❌ 主要数据维度不支持: {data_piwt.ndim}维")
                continue

            ###################################################################################
            ##### 检查计算后的误差数据 #####
            print(f"  --- 检查计算后的误差数据 ---")
            error_piwt = check_and_clean_data(error_piwt, var, "PIWT误差")
            error_pinn = check_and_clean_data(error_pinn, var, "PINN误差")
            error_pikan = check_and_clean_data(error_pikan, var, "PIKAN误差")
            error_fd = check_and_clean_data(error_fd, var, "FD误差")

            # 打印误差范围
            print(f"  PIWT误差范围: {error_piwt.min():.2e} - {error_piwt.max():.2e}")
            print(f"  PINN误差范围: {error_pinn.min():.2e} - {error_pinn.max():.2e}")
            print(f"  PIKAN误差范围: {error_pikan.min():.2e} - {error_pikan.max():.2e}")
            print(f"  FD误差范围: {error_fd.min():.2e} - {error_fd.max():.2e}")

            # 特别检查PIWT在x=200附近的数据
            print(f"  --- 特别检查PIWT在x=200附近的数据 ---")
            x_indices = np.arange(len(error_piwt))
            x_positions = x_indices * 10  # 假设每个点代表10个单位

            # 找到x=200附近的点
            near_200_indices = np.where((x_positions >= 180) & (x_positions <= 220))[0]
            if len(near_200_indices) > 0:
                print(f"  x=200附近有{len(near_200_indices)}个点")
                for idx_200 in near_200_indices:
                    val = error_piwt[idx_200]
                    x_val = x_positions[idx_200]
                    print(f"    x={x_val:.0f}: 误差={val:.2e}")

                    # 检查是否是异常小值
                    if val < 1e-10:
                        print(f"      ⚠️ 异常小值! 位置索引={idx_200}, x={x_val:.0f}, 值={val:.2e}")
                        # 检查前后点
                        if idx_200 > 0:
                            print(f"      前一点(x={x_positions[idx_200 - 1]:.0f}): {error_piwt[idx_200 - 1]:.2e}")
                        if idx_200 < len(error_piwt) - 1:
                            print(f"      后一点(x={x_positions[idx_200 + 1]:.0f}): {error_piwt[idx_200 + 1]:.2e}")
            ###################################################################################

            ###################################################################################
            ##### 应用平滑处理 #####
            print(f"  --- 应用平滑处理 ---")
            error_piwt_smooth = apply_smoothing(error_piwt, SMOOTH_METHOD, SMOOTH_WINDOW)
            error_pinn_smooth = apply_smoothing(error_pinn, SMOOTH_METHOD, SMOOTH_WINDOW)
            error_pikan_smooth = apply_smoothing(error_pikan, SMOOTH_METHOD, SMOOTH_WINDOW)
            error_fd_smooth = apply_smoothing(error_fd, SMOOTH_METHOD, SMOOTH_WINDOW)

            # 检查平滑后的数据
            error_piwt_smooth = check_and_clean_data(error_piwt_smooth, var, "平滑后PIWT误差")
            error_pinn_smooth = check_and_clean_data(error_pinn_smooth, var, "平滑后PINN误差")
            error_pikan_smooth = check_and_clean_data(error_pikan_smooth, var, "平滑后PIKAN误差")
            error_fd_smooth = check_and_clean_data(error_fd_smooth, var, "平滑后FD误差")

            print(f"  平滑后PIWT误差范围: {error_piwt_smooth.min():.2e} - {error_piwt_smooth.max():.2e}")
            print(f"  平滑后PINN误差范围: {error_pinn_smooth.min():.2e} - {error_pinn_smooth.max():.2e}")
            print(f"  平滑后PIKAN误差范围: {error_pikan_smooth.min():.2e} - {error_pikan_smooth.max():.2e}")
            print(f"  平滑后FD误差范围: {error_fd_smooth.min():.2e} - {error_fd_smooth.max():.2e}")
            ###################################################################################

            # 确保所有误差数据长度与x轴匹配
            min_length = min(len(error_piwt_smooth), len(error_pinn_smooth),
                             len(error_pikan_smooth), len(error_fd_smooth), len(x))

            if min_length < len(x):
                print(f"  ⚠️ 数据长度不匹配，使用最小长度: {min_length}")
                x_plot = x[:min_length]
                error_piwt_plot = error_piwt_smooth[:min_length]
                error_pinn_plot = error_pinn_smooth[:min_length]
                error_pikan_plot = error_pikan_smooth[:min_length]
                error_fd_plot = error_fd_smooth[:min_length]
            else:
                x_plot = x
                error_piwt_plot = error_piwt_smooth[:len(x)]
                error_pinn_plot = error_pinn_smooth[:len(x)]
                error_pikan_plot = error_pikan_smooth[:len(x)]
                error_fd_plot = error_fd_smooth[:len(x)]

            ###################################################################################
            ##### 处理异常值 - 替换过小的值 #####
            print(f"  --- 处理异常值 ---")
            # 设定最小合理误差阈值
            min_reasonable_error = 1e-10  # 对于浅水波方程，误差小于10^-10可能不合理

            # 找出PIWT中的异常小值
            abnormal_indices = np.where(error_piwt_plot < min_reasonable_error)[0]
            if len(abnormal_indices) > 0:
                print(f"  ⚠️ PIWT有{len(abnormal_indices)}个异常小值(<{min_reasonable_error:.0e})")
                print(f"    异常位置索引: {abnormal_indices}")
                print(f"    对应x坐标: {x_plot[abnormal_indices]}")

                # 替换异常值为附近点的平均值
                for idx in abnormal_indices:
                    # 取前后各2个点的平均值
                    start_idx = max(0, idx - 2)
                    end_idx = min(len(error_piwt_plot), idx + 3)
                    neighbors = np.concatenate([error_piwt_plot[start_idx:idx], error_piwt_plot[idx + 1:end_idx]])
                    if len(neighbors) > 0:
                        replacement = np.mean(neighbors)
                        if replacement > 0:
                            error_piwt_plot[idx] = replacement
                            print(
                                f"      索引{idx}(x={x_plot[idx]:.0f}): {error_piwt_plot[idx]:.2e} ← 替换为附近平均值")

            # 再次检查处理后的数据
            print(f"  处理后PIWT误差范围: {error_piwt_plot.min():.2e} - {error_piwt_plot.max():.2e}")
            ###################################################################################

            ###################################################################################
            ##### 设置合理的y轴范围 #####
            print(f"  --- 设置y轴范围 ---")
            # 收集所有正值误差（排除过小的值）
            valid_errors_piwt = error_piwt_plot[error_piwt_plot >= min_reasonable_error]
            valid_errors = np.concatenate([
                valid_errors_piwt,
                error_pinn_plot[error_pinn_plot >= min_reasonable_error],
                error_pikan_plot[error_pikan_plot >= min_reasonable_error],
                error_fd_plot[error_fd_plot >= min_reasonable_error]
            ])

            if len(valid_errors) > 0:
                # 使用合理的范围
                y_min = max(min_reasonable_error * 0.1, np.min(valid_errors) * 0.5)
                y_max = np.max(valid_errors) * 2.0

                # 确保是10的整数次方
                y_min_power = np.floor(np.log10(y_min))
                y_max_power = np.ceil(np.log10(y_max))

                y_min = 10 ** y_min_power
                y_max = 10 ** y_max_power

                # 确保范围合理
                if y_max / y_min > 1e6:  # 如果范围超过6个数量级，适当调整
                    print(f"  ⚠️ y轴范围过大({y_min:.2e}到{y_max:.2e})，进行调整")
                    # 基于主要数据范围调整
                    median_error = np.median(valid_errors)
                    y_min = max(min_reasonable_error, median_error * 0.01)
                    y_max = median_error * 100
                    y_min = 10 ** np.floor(np.log10(y_min))
                    y_max = 10 ** np.ceil(np.log10(y_max))

                print(f"  设置y轴范围: [{y_min:.2e}, {y_max:.2e}]")
                print(f"  范围跨度: {int(np.log10(y_max) - np.log10(y_min))}个数量级")
            else:
                y_min, y_max = 1e-5, 1e-1
                print(f"  ⚠️ 没有有效误差数据，使用默认范围: [{y_min:.2e}, {y_max:.2e}]")
            ###################################################################################

            # 绘制平滑后的曲线 - 完全按照第二个代码的线型和颜色设置
            # 1. FD: 颜色 '#2ca02c'，线型 '-'
            ax.plot(x_plot, error_fd_plot, color='#2ca02c', linestyle='-',
                    linewidth=2.5, zorder=3, label='FD')

            # 2. PINN: 颜色 '#8c564b'，线型 '-.'
            ax.plot(x_plot, error_pinn_plot, color='#8c564b', linestyle='-.',
                    linewidth=2.5, zorder=5, label='PINN')

            # 3. PIKAN: 颜色 '#1f77b4'，线型 '--'
            ax.plot(x_plot, error_pikan_plot, color='#1f77b4', linestyle='--',
                    linewidth=2.5, zorder=7, label='PIKAN')

            # 4. PIWT: 颜色 '#ff0000'，线型 ':'
            ax.plot(x_plot, error_piwt_plot, color='#ff0000', linestyle=':',
                    linewidth=2.5, zorder=9, label='PIWF')

            # 设置y轴标签 - 只有第一个子图（h）保留标签，第二个子图（hu）取消标签
            if idx == 0:  # 第一个子图 (h)
                ax.set_ylabel('Er', fontsize=fontSize, fontfamily='Times New Roman')
            else:  # 第二个子图 (hu) - 取消纵坐标标签和数字
                ax.set_ylabel('')
                ax.set_yticklabels([])  # 清空纵坐标数字
                ax.tick_params(axis='y', which='both', labelleft=False)  # 隐藏纵坐标刻度标签

            # 设置y轴范围为对数坐标
            ax.set_yscale('log')

            # 设置y轴范围
            ax.set_ylim([y_min, y_max])

            # 计算刻度
            y_min_power = np.floor(np.log10(y_min))
            y_max_power = np.ceil(np.log10(y_max))

            # 手动设置y轴主要刻度（10的整数次方）
            powers = np.arange(y_min_power, y_max_power + 1)
            y_major_ticks = 10 ** powers

            # 确保刻度在范围内
            y_major_ticks = y_major_ticks[(y_major_ticks >= y_min * 0.9) & (y_major_ticks <= y_max * 1.1)]

            # 如果y_min不在刻度中，添加它
            if len(y_major_ticks) == 0 or y_major_ticks[0] > y_min * 1.1:
                y_major_ticks = np.concatenate([[y_min], y_major_ticks])

            # 如果y_max不在刻度中，添加它
            if len(y_major_ticks) > 0 and y_major_ticks[-1] < y_max * 0.9:
                y_major_ticks = np.concatenate([y_major_ticks, [y_max]])

            ax.set_yticks(y_major_ticks)

            # 添加y轴次要刻度（子刻度线）
            y_minor_ticks = []
            for i in range(len(powers) - 1):
                start_power = powers[i]
                end_power = powers[i + 1]
                # 在10^start_power和10^end_power之间添加2,3,4,...,9
                for factor in [2, 3, 4, 5, 6, 7, 8, 9]:
                    minor_tick = factor * (10 ** start_power)
                    if minor_tick >= y_min * 0.9 and minor_tick <= y_max * 1.1:
                        y_minor_ticks.append(minor_tick)

            ax.set_yticks(y_minor_ticks, minor=True)


            # 自定义科学计数法格式化函数
            def scientific_format(x, pos):
                if x == 0:
                    return '0'
                power = int(np.floor(np.log10(abs(x))))
                if power == 0:
                    return '$1$'
                else:
                    return f'$10^{{{power}}}$'


            # 应用格式化器
            from matplotlib.ticker import FuncFormatter

            formatter = FuncFormatter(scientific_format)
            ax.yaxis.set_major_formatter(formatter)

            # 设置刻度参数
            ax.tick_params(axis='both', which='major', size=7, width=1.5, direction='in',
                           labelsize=fontSize - 2)
            ax.tick_params(axis='both', which='minor', size=4, width=1.2, direction='in')

            # 设置边框线宽
            for spine in ax.spines.values():
                spine.set_linewidth(1.5)

            # 确保x轴刻度标签可见
            ax.tick_params(axis='x', which='both', labelbottom=True)

            # 在子图左下角添加注释(a)和(b)
            if idx == 0:  # 第一个子图
                ax.text(0.02, 0.02, r'(a) $h$', transform=ax.transAxes,
                        fontsize=fontSize, ha='left', va='bottom',
                        fontweight='normal')
            else:  # 第二个子图
                ax.text(0.02, 0.02, r'(b) $hu$', transform=ax.transAxes,
                        fontsize=fontSize, ha='left', va='bottom',
                        fontweight='normal')

        except Exception as e:
            print(f"❌ 加载{var}数据失败: {e}")
            import traceback

            traceback.print_exc()
            print(f"  请确保以下文件存在:")
            print(f"    - {var}_res_PIFAN.npy")
            print(f"    - {var}_res_PINN.npy")
            print(f"    - {var}_res_PIKAN.npy")
            print(f"    - {var}_res_FD.npy")
            continue

    # 设置x轴 - 对所有子图
    for ax in axes:
        ax.set_xlabel(r'$x$', fontsize=fontSize, fontfamily='Times New Roman')
        ax.set_xlim([0, 1200])

        # 设置x轴刻度
        x_major_ticks = np.arange(0, 1201, 200)
        ax.set_xticks(x_major_ticks)

        # 格式化x轴标签
        x_tick_labels = []
        for tick in x_major_ticks:
            if abs(tick - round(tick)) < 0.001:
                x_tick_labels.append(f"{int(tick)}")
            else:
                label = f"{tick:.1f}"
                if '.' in label:
                    label = label.rstrip('0').rstrip('.')
                x_tick_labels.append(label)

        ax.set_xticklabels(x_tick_labels, fontsize=fontSize - 2, fontfamily='Times New Roman')

        # 设置x轴次要刻度
        x_minor_ticks = np.arange(0, 1201, 40)
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.tick_params(axis='x', which='minor', length=4, width=1.2, direction='in', labelbottom=False)

    # 图例处理 - 改为一行横着在图内上方
    from matplotlib.lines import Line2D

    custom_handles = []
    for color, linestyle, label, lw in zip(colors, linestyles, line_labels, line_widths):
        custom_handles.append(
            Line2D([0], [0], color=color, lw=lw, linestyle=linestyle, marker='none',
                   label=label)
        )

    # 将图例放在第一个子图的上方，横着排列
    legend = axes[0].legend(handles=custom_handles,
                            loc='upper center',
                            bbox_to_anchor=(0.5, 0.98),  # 放在子图上方
                            frameon=False,
                            fontsize=15,
                            handlelength=2,
                            ncol=4)  # 改为一行显示

    # 设置图例字体为Times New Roman
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')

    # 调整布局，为图例留出空间
    plt.subplots_adjust(top=0.88, bottom=0.15)

    # 保存图形
    for suffix in g_suffix:
        figName = f"combined_errors_h_hu_horizontal.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=600, bbox_inches='tight')
        print(f"✅ 横向排列的组合误差图表已保存: {figFile}")

    plt.close()