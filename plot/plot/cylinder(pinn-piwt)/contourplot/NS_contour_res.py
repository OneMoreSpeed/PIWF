import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
import scipy.io as sio
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm  # 添加LogNorm用于对数尺度

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fig_DPI = 600
fontSize = 32  # 从26改为32，放大文字
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
lcolor = ['k', 'r', 'k', 'r', '#FF0000', '#00D200', '#0000FF', '#FF00FF', '#000000', '#000000', '#000000', 'salmon',
          'violet', 'yellowgreen']
lstyle = ["--", "--", "-", "--", ":", "-", "--", "-.", ":", "--", "-."]
lwidth = [2.5] * 10
mshape = ["o", "s", "^", "^", "^", "o", "v", "+", "D", "s", "^", "v", "<", ">", "d", "*"]
mcolor = ['r', '#00D200', '#0000FF', '#FF00FF', '#FF8000', '#000000', "b", "k", 'gold', 'salmon', 'goldenrod', 'violet']
isolid = [True, False, False, True, True, False, False, False, False, False]
msizes = [20, 8, 10, 10, 50, 50, 50, 100, 25, 50, 100, 10, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
mstyle = ["-", "--", "-", "--", ":", "-", "--", "-.", ":", "--", "-."]
mwidth = [2.5] * 10
###################################################################################
################################ set custom colormap #############################
numbin = 100

# 为DSN结果使用rainbow colormap
cmap_rainbow0 = plt.get_cmap("gist_rainbow_r")
minval_rainbow = 0.2
maxval_rainbow = 1
colors_rainbow = cmap_rainbow0(np.linspace(minval_rainbow, maxval_rainbow, numbin))
cmap_rainbow = mpl.colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_rainbow0.name, a=minval_rainbow, b=maxval_rainbow), colors_rainbow)

# 为误差结果使用灰度colormap
cmap_gray0 = plt.get_cmap("Greys")
minval_gray = 0
maxval_gray = 1
colors_gray = cmap_gray0(np.linspace(minval_gray, maxval_gray, numbin))
cmap_gray = mpl.colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_gray0.name, a=minval_gray, b=maxval_gray), colors_gray)

###################################################################################

if __name__ == "__main__":
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    pdir1 = os.path.dirname(cwd)
    pdir2 = os.path.dirname(pdir1)
    filePath = os.path.join(pdir1)
    workPath = os.path.join(cwd)
    fig_Path = os.path.join(cwd, "figure_res")
    g_suffix = ["pdf", "png"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
    ###################################################################################
    ############################# profile file settings ###############################
    numStep = 50000
    fieldName1 = "pred"
    fieldName2 = "true"
    fieldName3 = "res"
    str_numStep = "{}".format(numStep)
    solverPath = os.path.join(filePath)
    fieldPath = os.path.join(filePath, str_numStep)
    ###################################################################################
    # x & y labels
    xlabel = r"$x$"
    ylabel = r"$y$"
    ###################################################################################

    # 设置要绘制的时间点 - 方便修改
    selected_time = 1  # 可以选择 0, 1, 2, 3, 4, 5, 6, 7 等

    print(f"正在绘制 t = {selected_time} 时刻的结果...")

    # 加载基础数据
    data = loadmat("../cylinder_nektar_wake.mat")
    U_star = data["U_star"]  # N x 2 x T
    P_star = data["p_star"]  # N x T
    t_star = data["t"]  # T x 1
    X_star = data["X_star"]  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]

    # 创建插值网格
    x_grid = np.linspace(1, 8, 200)
    y_grid = np.linspace(-2, 2, 80)
    XX_grid, YY_grid = np.meshgrid(x_grid, y_grid)

    # 获取指定时间点的数据
    t_n = selected_time
    # 构建原始数据点
    X = X_star[:, 0]
    Y = X_star[:, 1]
    xy_points = np.c_[X, Y]

    # 定义模型和物理量
    models = ['pinn', 'pikan', 'piwt']
    variables = ['u', 'v', 'p']

    # 为每个变量定义范围（DSN使用）
    var_ranges = {
        'u': {'min': -1, 'max': 1.5, 'inc': 0.5, 'label': r"$u$"},
        'v': {'min': -0.5, 'max': 0.5, 'inc': 0.25, 'label': r"$v$"},
        'p': {'min': -0.5, 'max': 0, 'inc': 0.1, 'label': r"$p$"}
    }

    # 创建子图注释
    dsn_labels = [
        r'(a) DNS u', r'(b) DNS v', r'(c) DNS p'
    ]
    # 第二、三、四行：PINN、PIKAN、PIWT结果
    subplot_labels = [
        r'(d) PINN:$u$', r'(e) PINN:$v$', r'(f) PINN:$p$',
        r'(g) PIKAN:$u$', r'(h) PIKAN:$v$', r'(i) PIKAN:$p$',
        r'(j) PIWF:$u$', r'(k) PIWF:$v$', r'(l) PIWF:$p$'
    ]

    # 使用两个独立的GridSpec来精确控制第一行和第二行之间的间距
    fig = plt.figure(figsize=(22, 16))

    # 定义高度比例：第一行高度，间距，后面三行总高度
    # 这里设置间距高度为0.08（相对高度），可以根据需要调整
    gap_height = 0.08  # 第一行和第二行之间的间距大小，增大这个值可以增加距离

    # 创建第一个GridSpec用于第一行（DSN结果）
    gs1 = gridspec.GridSpec(1, 3, figure=fig,
                            top=0.88, bottom=0.88 - 0.2,  # 第一行的位置
                            left=0.08, right=0.95,
                            wspace=0.1)

    # 创建第二个GridSpec用于后面三行（PINN、PIKAN、PIWT结果）
    # 第二行开始的位置 = 第一行底部位置 - 间距高度
    first_row_bottom = 0.88 - 0.2  # 第一行底部位置
    second_row_top = first_row_bottom - gap_height  # 第二行顶部位置

    gs2 = gridspec.GridSpec(3, 3, figure=fig,
                            top=second_row_top, bottom=0.08,  # 后面三行的位置
                            left=0.08, right=0.95,
                            hspace=0.08, wspace=0.1)  # hspace控制后面三行之间的间距

    # 创建子图
    axes = np.zeros((4, 3), dtype=object)

    # 第一行子图
    for j in range(3):
        axes[0, j] = fig.add_subplot(gs1[0, j])

    # 后面三行子图
    for i in range(3):
        for j in range(3):
            axes[i + 1, j] = fig.add_subplot(gs2[i, j])

    ###################################################################################
    # 误差图设置 - colorbar固定为10^-1, 10^-0.5, 10^0
    nlevel = 101
    # COLORBAR固定的上下限：始终为10^-1 到 10^0
    colorbar_min_log = -1  # 固定为-1
    colorbar_max_log = 1  # 固定为0
    # 在线性空间创建等间距的level（在对数尺度下等间距）
    levels_log = np.linspace(colorbar_min_log, colorbar_max_log, nlevel)
    # 转换为实际数值用于contourf
    levels = 10 ** levels_log

    # Colorbar (cb) settings
    cb_w = 0.83  # colorbar长度
    cb_h = 0.02
    cb_fmt = mpl.ticker.StrMethodFormatter("{x:g}")
    cb_FontSize = fontSize
    cb_label = r"Er(%)"  # 改为百分比误差标签

    # 存储所有contour对象用于colorbar
    contour_plots = []

    # 首先绘制第一行：DSN结果
    for col in range(3):
        ax = axes[0, col]
        var = variables[col]

        var_range = var_ranges[var]
        var_min_dsn = var_range['min']
        var_max_dsn = var_range['max']
        var_inc_dsn = var_range['inc']

        dsn_filename = f"{var}_trueNS_T{selected_time}.npy"
        try:
            dsn_data = np.load('./' + dsn_filename)
            if np.any(np.isnan(dsn_data)) or np.any(np.isinf(dsn_data)):
                dsn_data = np.nan_to_num(dsn_data, nan=0.0, posinf=0.0, neginf=0.0)
            field_interp = griddata(xy_points, dsn_data.flatten(), (XX_grid, YY_grid), method='cubic')
            if np.any(np.isnan(field_interp)) or np.any(np.isinf(field_interp)):
                field_interp = np.nan_to_num(field_interp, nan=0.0, posinf=0.0, neginf=0.0)
            levels_dsn = np.linspace(var_min_dsn, var_max_dsn, nlevel)
            CS_dsn = ax.contourf(XX_grid, YY_grid, field_interp, vmin=var_min_dsn, vmax=var_max_dsn,
                                 levels=levels_dsn, cmap=cmap_rainbow, extend="both")
        except Exception as e:
            field_interp = np.zeros_like(XX_grid)
            CS_dsn = ax.contourf(XX_grid, YY_grid, field_interp, cmap=cmap_rainbow)

        cmap_rainbow.set_over(colors_rainbow[-1])
        cmap_rainbow.set_under(colors_rainbow[0])
        CS_dsn.set_cmap(cmap_rainbow)

        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)

        xlimit = [1, 8]
        ylimit = [-2, 2]
        ax.set_xlim(xlimit[0], xlimit[1])
        ax.set_ylim(ylimit[0], ylimit[1])

        if col == 0:
            ax.set_ylabel(ylabel)
            ax.yaxis.set_tick_params(which='major', size=7, width=2.5, direction='in', right=False)
            ax.yaxis.set_tick_params(which='minor', size=4, width=2.5, direction='in', right=False)
            ax.yaxis.set_tick_params(pad=8)
        else:
            ax.set_ylabel('')
            ax.yaxis.set_tick_params(which='major', size=0, width=0, direction='in', right=False)
            ax.yaxis.set_tick_params(which='minor', size=0, width=0, direction='in', right=False)
            ax.set_yticklabels([])
            ax.yaxis.set_visible(False)

        ax.set_xlabel('')
        ax.xaxis.set_tick_params(which='major', size=0, width=0, direction='in', top=False)
        ax.xaxis.set_tick_params(which='minor', size=0, width=0, direction='in', top=False)
        ax.set_xticklabels([])
        ax.xaxis.set_visible(False)

        xinterval = 1
        yinterval = 1
        yticks = np.arange(ylimit[0], ylimit[1] + yinterval, yinterval)
        if col == 0:
            ax.set_yticks(yticks)
            ytickLabels = ["{:.0f}".format(yticks[0])] + ["{:.1f}".format(yticks[i]) for i in range(1, len(yticks))]
            ax.set_yticklabels(["%g" % y for y in yticks])
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        else:
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

        ax.text(0.02, 0.02, dsn_labels[col], transform=ax.transAxes,
                fontsize=fontSize - 2, ha='left', va='bottom', color='black')

        cb_ax_dsn = fig.add_axes([0.1 + col * 0.3, 0.91, 0.25, 0.02])
        num_ticks = int((var_max_dsn - var_min_dsn) / var_inc_dsn) + 1
        cb_ticks_dsn = np.linspace(var_min_dsn, var_max_dsn, num_ticks)
        cb_ticks_dsn = np.round(cb_ticks_dsn, 2)

        cb_dsn = fig.colorbar(CS_dsn, ticks=cb_ticks_dsn, cax=cb_ax_dsn, format=cb_fmt,
                              orientation='horizontal', extendfrac=0)
        cb_dsn.ax.tick_params(labelsize=fontSize - 4, length=12, width=0.4)
        cb_dsn.ax.text(s=var_range['label'], x=-0.05, y=0.3, va="center", ha="right",
                       transform=cb_ax_dsn.transAxes, fontsize=fontSize - 2)

        # 为DNS的colorbar添加外边界（黑边）
        cb_dsn.outline.set_edgecolor('black')
        cb_dsn.outline.set_linewidth(1.5)

    # 绘制第2-4行：误差图 - 使用对数尺度（去掉等值线）
    for idx, (ax, label) in enumerate(zip(axes[1:].flat, subplot_labels)):
        row = idx // 3
        col = idx % 3
        model = models[row]
        var = variables[col]

        filename = f"{var}_resNS_T{selected_time}_{model}.npy"
        try:
            pred_data = np.load('./' + filename)
            if np.any(np.isnan(pred_data)) or np.any(np.isinf(pred_data)):
                pred_data = np.nan_to_num(pred_data, nan=0.0, posinf=0.0, neginf=0.0)
            # 确保数据为正数，避免log10出错
            pred_data = np.maximum(pred_data, 1e-10)
            field_interp = griddata(xy_points, pred_data.flatten(), (XX_grid, YY_grid), method='cubic')
            if np.any(np.isnan(field_interp)) or np.any(np.isinf(field_interp)):
                field_interp = np.nan_to_num(field_interp, nan=0.0, posinf=0.0, neginf=0.0)
            # 再次确保为正数
            field_interp = np.maximum(field_interp, 1e-10)
        except Exception as e:
            field_interp = np.ones_like(XX_grid) * 1e-10

        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)

        ax.set_xlim(xlimit[0], xlimit[1])
        ax.set_ylim(ylimit[0], ylimit[1])

        if col == 0:
            ax.set_ylabel(ylabel)
            ax.yaxis.set_tick_params(which='major', size=7, width=2.5, direction='in', right=False)
            ax.yaxis.set_tick_params(which='minor', size=4, width=2.5, direction='in', right=False)
            ax.yaxis.set_tick_params(pad=8)
        else:
            ax.set_ylabel('')
            ax.yaxis.set_visible(False)

        if row == 2:
            ax.set_xlabel(xlabel)
            ax.xaxis.set_tick_params(which='major', size=7, width=2.5, direction='in', top=False)
            ax.xaxis.set_tick_params(which='minor', size=4, width=2.5, direction='in', top=False)
            ax.xaxis.set_tick_params(pad=8)
        else:
            ax.set_xlabel('')
            ax.xaxis.set_visible(False)

        xticks = np.arange(xlimit[0], xlimit[1] + xinterval, xinterval)
        if row == 2:
            ax.set_xticks(xticks)
            xtickLabels = ["{:.0f}".format(xticks[0])] + ["{:.2f}".format(xticks[i]) for i in range(1, len(xticks))]
            ax.set_xticklabels(["%g" % x for x in xticks])
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        if col == 0:
            ax.set_yticks(yticks)
            ytickLabels = ["{:.0f}".format(yticks[0])] + ["{:.1f}".format(yticks[i]) for i in range(1, len(yticks))]
            ax.set_yticklabels(["%g" % y for y in yticks])
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        ax.text(0.02, 0.02, label, transform=ax.transAxes,
                fontsize=fontSize - 2, ha='left', va='bottom', color='black')

        # 使用对数尺度的contourf - colorbar范围固定为10^-1到10^0
        norm = LogNorm(vmin=10 ** colorbar_min_log, vmax=10 ** colorbar_max_log)
        CS = ax.contourf(XX_grid, YY_grid, field_interp,
                         levels=levels,  # 使用对数等间距的levels
                         norm=norm,
                         cmap=cmap_gray,
                         extend="both")

        # ========== 去掉等值线部分 ==========
        # 注释掉原来的等值线绘制代码
        # contour_levels_log = np.linspace(contour_min_log, contour_max_log, 3)  # 3条等值线
        # contour_levels = 10 ** contour_levels_log
        # # 过滤掉超出colorbar范围的等值线（可选，只显示在colorbar范围内的）
        # contour_levels = [l for l in contour_levels if l >= 10 ** colorbar_min_log and l <= 10 ** colorbar_max_log]
        # if len(contour_levels) > 0:
        #     contour_lines = ax.contour(XX_grid, YY_grid, field_interp,
        #                                levels=contour_levels,
        #                                colors='black', linewidths=1.0)
        #     # 修改为百分比形式，数据本身已经乘以100，所以直接加百分号
        #     ax.clabel(contour_lines, inline=True, fontsize=fontSize - 8,
        #               fmt=lambda x: f'{x:.1f}%')
        # ========== 等值线已去掉 ==========

        contour_plots.append(CS)
        cmap_gray.set_over(cmap_gray(1.0))
        cmap_gray.set_under(cmap_gray(0.0))
        CS.set_cmap(cmap_gray)

    # 创建误差图的colorbar - 固定为10^-1, 10^-0.5, 10^0，均匀分布
    if len(contour_plots) > 0:
        cb_ax = fig.add_axes([0.12, 0.63, cb_w, cb_h])

        # 设置colorbar的刻度：使用固定的数值
        cb_ticks_values = [0.1, 0.316227766, 1.0]  # 实际数值（10^-1, 10^-0.5, 10^0）
        cb_ticks_labels = [r'$10^{-1}$', r'$10^{-0.5}$', r'$10^{0}$']

        # 创建colorbar，但暂时不设置ticks
        cb = fig.colorbar(contour_plots[0], cax=cb_ax,
                          orientation='horizontal', extendfrac=0)

        # 获取colorbar的axes对象
        cb_ax_obj = cb.ax

        # 强制更新绘图
        plt.draw()

        # 获取colorbar的x轴范围（实际坐标）
        xlim = cb_ax_obj.get_xlim()

        # 精确计算三个位置的坐标
        # 左端位置
        pos_left = xlim[0]
        print('aaaaa',pos_left)
        # 中间位置
        pos_center = (xlim[0] + xlim[1]) / 2.0
        # 右端位置
        pos_right = xlim[1]

        # 设置刻度和标签到精确的位置
        cb_ax_obj.set_xticks([pos_left, 0.9, pos_right])
        cb_ax_obj.set_xticklabels(cb_ticks_labels)
        cb_ax_obj.tick_params(labelsize=cb_FontSize, length=22, width=0.4,pad=8)

        # 设置colorbar的标签
        cb.ax.text(s=cb_label, x=-0.03, y=0.3, va="center", ha="right",
                   transform=cb_ax.transAxes, fontsize=cb_FontSize + 2)

        # 为误差图的colorbar添加外边界（黑边）
        cb.outline.set_edgecolor('black')
        cb.outline.set_linewidth(1.5)

    for suffix in g_suffix:
        figName = f"combined_NS_T{selected_time}_with_DNS_no_contours.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=fig_DPI, bbox_inches='tight')
        print(f"已保存: {figFile}")
    plt.close()