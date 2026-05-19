import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import griddata
import data
from scipy.io import loadmat
import scipy.io as sio
from matplotlib.colors import LogNorm

mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fig_DPI = 600
fontSize = 26  # 从26改为32，放大文字
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
# 使用灰度colormap
cmap0 = plt.get_cmap("Greys")
# truncated the standard colormap
minval = 0
maxval = 1
colors = cmap0(np.linspace(minval, maxval, numbin))
cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap0.name, a=minval, b=maxval), colors)
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
    g_suffix = ["svg", "pdf", "png"]
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
    ylabel = r"$t$"
    ###################################################################################

    # 1. Read mesh for interpolation
    _, _, _, _, _, input, label_h, label_u, _, _ = data.load_SWEdata()
    x = input[:, 0:1]
    t = input[:, 1:2]
    x_point = x.flatten()
    t_point = t.flatten()
    xt_point = np.c_[x_point, t_point]

    # Create mesh grid for contour
    nn = 500
    x = np.linspace(0, 1200, nn)
    t = np.linspace(0, 3600, nn)
    X, T = np.meshgrid(x, t)

    # 定义文件列表 - 第一行h，第二行hu
    file_list = [
        'h_res_FD.npy',  # 第一行第一列 (a) FD:h
        'h_res_PINN.npy',  # 第一行第二列 (b) PINN:h
        'h_res_PIKAN.npy',  # 第一行第三列 (c) PIKAN:h
        'h_res_PIWT.npy',  # 第一行第四列 (d) PIWT:h
        'hu_res_FD.npy',  # 第二行第一列 (e) FD:hu
        'hu_res_PINN.npy',  # 第二行第二列 (f) PINN:hu
        'hu_res_PIKAN.npy',  # 第二行第三列 (g) PIKAN:hu
        'hu_res_PIWT.npy'  # 第二行第四列 (h) PIWT:hu
    ]

    # 子图注释
    subplot_labels = [
        r'(a) FD: $h$',
        r'(b) PINN: $h$',
        r'(c) PIKAN: $h$',
        r'(d) PIWF: $h$',
        r'(e) FD: $hu$',
        r'(f) PINN: $hu$',
        r'(g) PIKAN: $hu$',
        r'(h) PIWF: $hu$'
    ]

    # 创建8个子图的布局
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    ###################################################################################
    # 4. Contour level settings - 改为对数坐标，范围10^-3到10^2
    nlevel = 101
    # 对数坐标的范围 -3 到 2 (对应 10^-3 到 10^2)
    var_min_log = 0  # 对应 10^-3
    var_max_log = 2  # 对应 10^2

    # 对数坐标的刻度值 - 增加到10^2
    log_ticks = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    log_tick_labels = [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$']

    # 5. Colorbar (cb) settings
    cb_w = 0.8
    cb_h = 0.02
    cb_fmt = mpl.ticker.StrMethodFormatter("{x:g}")
    cb_FontSize = fontSize
    cb_label = r"Er(%)"  # 百分比误差标签

    # 存储所有contour对象用于colorbar
    contour_plots = []

    # 循环绘制8个子图
    for idx, (ax, file_name, label) in enumerate(zip(axes.flat, file_list, subplot_labels)):
        # 数据处理：直接读取误差数据（已经是百分比形式）
        u_pred = np.load('./' + file_name)
        UU_star = griddata(xt_point, u_pred.flatten(), (X, T), method='cubic')
        fieldData1 = UU_star

        # 将数据转换为正数，避免log计算问题
        fieldData1 = np.maximum(fieldData1, 1e-10)

        # Hide the top and right spines of the axis
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        # Set the axis box line width
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)

        # (2) set the x & y limits
        xlimit = [0, 1200]
        ylimit = [0, 3600]
        ax.set_xlim(xlimit[0], xlimit[1])
        ax.set_ylim(ylimit[0], ylimit[1])

        # 坐标轴控制：只保留最左侧和最下侧的刻度和标签
        if idx == 0 or idx == 4:  # 最左侧的子图 (a和e) - 保留y轴
            ax.set_ylabel(ylabel)
            ax.yaxis.set_tick_params(which='major', size=7, width=2.5, direction='in', right=False)
            ax.yaxis.set_tick_params(which='minor', size=4, width=2.5, direction='in', right=False)
            ax.yaxis.set_tick_params(pad=8)
        else:  # 其他子图 - 完全隐藏y轴刻度和标签
            ax.set_ylabel('')
            ax.yaxis.set_tick_params(which='major', size=0, width=0, direction='in', right=False)
            ax.yaxis.set_tick_params(which='minor', size=0, width=0, direction='in', right=False)
            ax.set_yticklabels([])
            ax.yaxis.set_visible(False)

        if idx >= 4:  # 第二行子图 (e,f,g,h) - 保留x轴
            ax.set_xlabel(xlabel)
            ax.xaxis.set_tick_params(which='major', size=7, width=2.5, direction='in', top=False)
            ax.xaxis.set_tick_params(which='minor', size=4, width=2.5, direction='in', top=False)
            ax.xaxis.set_tick_params(pad=8)
        else:  # 第一行子图 (a,b,c,d) - 完全隐藏x轴刻度和标签
            ax.set_xlabel('')
            ax.xaxis.set_tick_params(which='major', size=0, width=0, direction='in', top=False)
            ax.xaxis.set_tick_params(which='minor', size=0, width=0, direction='in', top=False)
            ax.set_xticklabels([])
            ax.xaxis.set_visible(False)

        # 设置刻度和标签（仅对有显示的轴）
        xinterval = 200
        yinterval = 600
        xticks = np.arange(xlimit[0], xlimit[1] + xinterval, xinterval)
        yticks = np.arange(ylimit[0], ylimit[1] + yinterval, yinterval)
        xticks = np.around(xticks, decimals=2)
        yticks = np.around(yticks, decimals=2)

        if idx >= 4:  # 第二行子图设置x轴刻度
            ax.set_xticks(xticks)
            xtickLabels = ["{:.0f}".format(xticks[0])] + ["{:.2f}".format(xticks[i]) for i in range(1, len(xticks))]
            ax.set_xticklabels(xtickLabels)

        if idx == 0 or idx == 4:  # 最左侧子图设置y轴刻度
            ax.set_yticks(yticks)
            ytickLabels = ["{:.0f}".format(yticks[0])] + ["{:.1f}".format(yticks[i]) for i in range(1, len(yticks))]
            ax.set_yticklabels(ytickLabels)

        ########  set the minor tick locator (仅对显示刻度的轴) ########
        if idx >= 4:  # 第二行子图保留x轴minor
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        else:
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

        if idx == 0 or idx == 4:  # 最左侧子图保留y轴minor
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        else:
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

        ########  set the tick label format   ########
        formatter = mpl.ticker.StrMethodFormatter("{x:g}")
        if idx >= 4:  # 第二行子图
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if idx == 0 or idx == 4:  # 最左侧子图
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        # 在子图左上角添加注释 - 带白色半透明背景
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=fontSize - 2, ha='left', va='top')

        contour_var2 = fieldData1
        # 使用对数归一化，范围从10^-3到10^2
        varmin_ = 10 ** var_min_log  # 10^-3
        varmax_ = 10 ** var_max_log  # 10^2

        # 创建对数间隔的levels
        levels_log = np.logspace(var_min_log, var_max_log, nlevel)

        # 使用LogNorm和灰度colormap绘制contour
        CS2 = ax.contourf(X, T, contour_var2,
                          norm=LogNorm(vmin=varmin_, vmax=varmax_),
                          levels=levels_log, cmap=cmap1,
                          extend="both")
        contour_plots.append(CS2)

        # 设置colormap的超出范围的颜色
        colors_array = cmap1(np.linspace(0, 1, 256))
        cmap1.set_over(colors_array[-1])  # 超出最大值用最后一个颜色（深灰）
        cmap1.set_under(colors_array[0])  # 超出最小值用第一个颜色（浅灰）
        CS2.set_cmap(cmap1)

    # (6) set the colorbar —— 放在顶部，占满整个宽度
    cb_ax = fig.add_axes([0.15, 0.94, cb_w, cb_h])
    # 设置对数坐标的colorbar刻度，显示10^-3到10^2
    cb = fig.colorbar(contour_plots[0], ticks=log_ticks, cax=cb_ax,
                      format=mpl.ticker.LogFormatterMathtext(),
                      orientation='horizontal', extendfrac=0)
    cb.ax.tick_params(labelsize=cb_FontSize, length=22, width=0.4,pad=8)
    # 设置colorbar刻度标签
    cb.ax.set_xticklabels(log_tick_labels)
    # colorbar标签放大
    cb.ax.text(s=cb_label, x=-0.03, y=0.3, va="center", ha="right",
               transform=cb_ax.transAxes, fontsize=cb_FontSize + 2)
    cb.outline.set_edgecolor('white')
    # 添加colorbar边界（黑色边框）
    cb.outline.set_edgecolor('black')
    cb.outline.set_linewidth(2.0)

    # 调整子图之间的间距
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.08, wspace=0.1, hspace=0.08)

    for suffix in g_suffix:
        figName = ("combined_contour_h_"
                   "_greyscale_log.{}").format(suffix)
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=fig_DPI, bbox_inches='tight')
        print("Width x Height: ", fig.get_size_inches(), "suffix: ", suffix)
    plt.close()