import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
import scipy.io as sio
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import numpy.fft as fft

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

def compute_fft_2d(data_2d):
    """
    计算二维快速傅里叶变换，返回频谱幅度
    """
    # 进行2D FFT
    fft_result = fft.fft2(data_2d)
    # 将零频移到中心
    fft_shifted = fft.fftshift(fft_result)
    # 计算幅度谱
    magnitude = np.abs(fft_shifted)
    return magnitude


def compute_relative_error(pred_fft, true_fft):
    """
    计算相对误差（百分比）
    公式: |pred - true| / |true| * 100
    避免除零错误
    """
    # 添加小量避免除零
    epsilon = 1e-10
    relative_error = np.abs(pred_fft - true_fft) / (np.abs(true_fft) + epsilon) * 100
    return relative_error


def get_wavenumber_grid(shape, dx, dy):
    """
    生成波数网格 (k_x, k_y)
    shape: (ny, nx) 网格形状
    """
    ny, nx = shape
    # 计算波数坐标 (rad/m)
    kx = fft.fftfreq(nx, dx) * 2 * np.pi
    ky = fft.fftfreq(ny, dy) * 2 * np.pi
    # 使用fftshift使零频在中心
    KX, KY = np.meshgrid(fft.fftshift(kx), fft.fftshift(ky))
    return KX, KY


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
    # x & y labels (波数空间的标签)
    xlabel = r"$k_x$ "
    ylabel = r"$k_y$ "
    ###################################################################################

    # ==================== 超参数设置 ====================
    selected_time = 7  # 可以选择 0, 1, 2, 3, 4, 5, 6, 7 等
    print(f"正在绘制 t = {selected_time} 时刻的傅里叶空间相对误差结果...")
    # ==================================================

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

    # 网格形状 (ny, nx)
    ny, nx = XX_grid.shape

    # 计算网格间距
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    # 生成波数网格 (注意形状是 (ny, nx))
    KX, KY = get_wavenumber_grid(XX_grid.shape, dx, dy)

    # 设置波数显示范围（避免显示过大的波数）
    kx_limit = [-20, 20]
    ky_limit = [-20, 20]

    # 创建波数空间的掩码（只显示指定范围内的波数）
    kx_mask = (KX >= kx_limit[0]) & (KX <= kx_limit[1])
    ky_mask = (KY >= ky_limit[0]) & (KY <= ky_limit[1])
    combined_mask = kx_mask & ky_mask

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
    # 第二、三、四行：PINN、PIKAN、PIWT的傅里叶空间相对误差
    subplot_labels = [
        r'(d) PINN:$u$ ', r'(e) PINN:$v$ ', r'(f) PINN:$p$ ',
        r'(g) PIKAN:$u$ ', r'(h) PIKAN:$v$ ', r'(i) PIKAN:$p$ ',
        r'(j) PIWF:$u$ ', r'(k) PIWF:$v$ ', r'(l) PIWF:$p$ '
    ]

    # 使用两个独立的GridSpec来精确控制第一行和第二行之间的间距
    fig = plt.figure(figsize=(22, 16))

    # 定义高度比例：第一行高度，间距，后面三行总高度
    gap_height = 0.08  # 第一行和第二行之间的间距大小

    # 创建第一个GridSpec用于第一行（DSN结果）
    gs1 = gridspec.GridSpec(1, 3, figure=fig,
                            top=0.88, bottom=0.88 - 0.2,
                            left=0.08, right=0.95,
                            wspace=0.1)

    # 创建第二个GridSpec用于后面三行（PINN、PIKAN、PIWT的傅里叶空间相对误差）
    first_row_bottom = 0.88 - 0.2
    second_row_top = first_row_bottom - gap_height

    gs2 = gridspec.GridSpec(3, 3, figure=fig,
                            top=second_row_top, bottom=0.08,
                            left=0.08, right=0.95,
                            hspace=0.08, wspace=0.1)

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
    # 误差图设置 - colorbar改为10^0到10^2
    nlevel = 101
    # COLORBAR的上下限：10^0 到 10^2 (1% 到 100%)
    colorbar_min = 1  # 10^0 = 1%
    colorbar_max = 100  # 10^2 = 100%
    # 在线性空间创建等间距的level（在对数尺度下等间距）
    levels_log = np.linspace(np.log10(colorbar_min), np.log10(colorbar_max), nlevel)
    # 转换为实际数值用于contourf
    levels = 10 ** levels_log

    # Colorbar (cb) settings
    cb_w = 0.83  # colorbar长度
    cb_h = 0.02
    cb_fmt = mpl.ticker.StrMethodFormatter("{x:g}")
    cb_FontSize = fontSize
    cb_label = r"FFT Error(%)"  # 傅里叶空间相对误差标签

    # 存储所有contour对象用于colorbar
    contour_plots = []

    # 首先计算所有DNS的傅里叶频谱（作为真值）
    dns_fft_dict = {}
    for col in range(3):
        var = variables[col]
        dsn_filename = f"{var}_trueNS_T{selected_time}.npy"
        try:
            dsn_data = np.load('./' + dsn_filename)
            if np.any(np.isnan(dsn_data)) or np.any(np.isinf(dsn_data)):
                dsn_data = np.nan_to_num(dsn_data, nan=0.0, posinf=0.0, neginf=0.0)
            field_interp = griddata(xy_points, dsn_data.flatten(), (XX_grid, YY_grid), method='cubic')
            if np.any(np.isnan(field_interp)) or np.any(np.isinf(field_interp)):
                field_interp = np.nan_to_num(field_interp, nan=0.0, posinf=0.0, neginf=0.0)

            # 计算DNS的傅里叶频谱
            dns_fft = compute_fft_2d(field_interp)
            dns_fft_dict[var] = dns_fft

        except Exception as e:
            print(f"Error loading DNS {var}: {e}")
            dns_fft_dict[var] = None

    # 首先绘制第一行：DNS结果的傅里叶频谱（在波数空间显示）
    for col in range(3):
        ax = axes[0, col]
        var = variables[col]

        var_range = var_ranges[var]

        dns_fft = dns_fft_dict[var]

        if dns_fft is not None:
            # 使用对数尺度显示频谱
            dns_fft_log = np.log10(dns_fft + 1e-10)

            # 只显示指定波数范围内的数据
            dns_fft_log_masked = np.ma.masked_where(~combined_mask, dns_fft_log)

            # 设置显示范围
            fft_min = -2
            fft_max = 4
            levels_dsn = np.linspace(fft_min, fft_max, nlevel)
            CS_dsn = ax.contourf(KX, KY, dns_fft_log_masked,
                                 vmin=fft_min, vmax=fft_max,
                                 levels=levels_dsn, cmap=cmap_rainbow, extend="both")
        else:
            CS_dsn = ax.contourf(KX, KY, np.zeros_like(KX), cmap=cmap_rainbow)

        cmap_rainbow.set_over(colors_rainbow[-1])
        cmap_rainbow.set_under(colors_rainbow[0])
        CS_dsn.set_cmap(cmap_rainbow)

        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)

        # 设置波数显示范围
        ax.set_xlim(kx_limit[0], kx_limit[1])
        ax.set_ylim(ky_limit[0], ky_limit[1])

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

        # 设置波数空间的刻度
        kx_ticks = np.arange(-20, 21, 10)
        ky_ticks = np.arange(-20, 21, 10)

        if col == 0:
            ax.set_yticks(ky_ticks)
            ax.set_yticklabels([f"{y:.0f}" for y in ky_ticks])
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        else:
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

        ax.text(0.02, 0.02, dsn_labels[col], transform=ax.transAxes,
                fontsize=fontSize - 2, ha='left', va='bottom', color='black')

        # 为DNS的colorbar
        cb_ax_dsn = fig.add_axes([0.1 + col * 0.3, 0.91, 0.25, 0.02])
        num_ticks = 5
        cb_ticks_dsn = np.linspace(-2, 4, num_ticks)
        cb_ticks_dsn = np.round(cb_ticks_dsn, 1)

        cb_dsn = fig.colorbar(CS_dsn, ticks=cb_ticks_dsn, cax=cb_ax_dsn, format=cb_fmt,
                              orientation='horizontal', extendfrac=0)
        cb_dsn.ax.tick_params(labelsize=fontSize - 4, length=12, width=0.4)
        cb_dsn.ax.text(s=var_range['label'], x=-0.05, y=0.3, va="center", ha="right",
                       transform=cb_ax_dsn.transAxes, fontsize=fontSize - 2)

        cb_dsn.outline.set_edgecolor('black')
        cb_dsn.outline.set_linewidth(1.5)

    # 绘制第2-4行：傅里叶空间的相对误差图（在波数空间显示）
    for idx, (ax, label) in enumerate(zip(axes[1:].flat, subplot_labels)):
        row = idx // 3
        col = idx % 3
        model = models[row]
        var = variables[col]

        # 加载预测数据
        filename = f"{var}_predNS_T{selected_time}_{model}.npy"
        try:
            pred_data = np.load('./' + filename)
            if np.any(np.isnan(pred_data)) or np.any(np.isinf(pred_data)):
                pred_data = np.nan_to_num(pred_data, nan=0.0, posinf=0.0, neginf=0.0)

            # 插值到网格
            field_interp = griddata(xy_points, pred_data.flatten(), (XX_grid, YY_grid), method='cubic')
            if np.any(np.isnan(field_interp)) or np.any(np.isinf(field_interp)):
                field_interp = np.nan_to_num(field_interp, nan=0.0, posinf=0.0, neginf=0.0)

            # 计算预测结果的傅里叶频谱
            pred_fft = compute_fft_2d(field_interp)

            # 获取DNS的傅里叶频谱
            true_fft = dns_fft_dict[var]

            if true_fft is not None:
                # 计算相对误差（百分比）
                relative_error = compute_relative_error(pred_fft, true_fft)

                # 限制相对误差的范围（1% 到 100%）
                relative_error = np.clip(relative_error, colorbar_min, colorbar_max)

                # 只显示指定波数范围内的数据
                error_data_masked = np.ma.masked_where(~combined_mask, relative_error)
            else:
                error_data_masked = np.ma.masked_all_like(KX)

        except Exception as e:
            print(f"Error processing {model} {var}: {e}")
            error_data_masked = np.ma.masked_all_like(KX)

        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)

        # 设置波数显示范围
        ax.set_xlim(kx_limit[0], kx_limit[1])
        ax.set_ylim(ky_limit[0], ky_limit[1])

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

        # 设置波数空间的刻度
        if row == 2:
            ax.set_xticks(kx_ticks)
            ax.set_xticklabels([f"{x:.0f}" for x in kx_ticks])
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        if col == 0:
            ax.set_yticks(ky_ticks)
            ax.set_yticklabels([f"{y:.0f}" for y in ky_ticks])
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        ax.text(0.02, 0.02, label, transform=ax.transAxes,
                fontsize=fontSize - 2, ha='left', va='bottom', color='black')

        # 使用对数尺度的contourf - colorbar范围固定为10^0到10^2
        norm = LogNorm(vmin=colorbar_min, vmax=colorbar_max)
        CS = ax.contourf(KX, KY, error_data_masked,
                         levels=levels,  # 使用对数等间距的levels
                         norm=norm,
                         cmap=cmap_gray,
                         extend="both")

        contour_plots.append(CS)
        cmap_gray.set_over(cmap_gray(1.0))
        cmap_gray.set_under(cmap_gray(0.0))
        CS.set_cmap(cmap_gray)

    # 创建误差图的colorbar - 让matplotlib自动处理对数刻度位置
    if len(contour_plots) > 0:
        cb_ax = fig.add_axes([0.12, 0.63, cb_w, cb_h])

        # 创建colorbar，使用第一个contour对象
        cb = fig.colorbar(contour_plots[0], cax=cb_ax,
                          orientation='horizontal', extendfrac=0)

        # 设置colorbar的刻度（让matplotlib自动计算正确的对数位置）
        cb.set_ticks([1, 10, 100])
        cb.set_ticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$'])

        # 设置刻度参数
        cb.ax.tick_params(labelsize=cb_FontSize, length=22, width=0.4,pad=8)

        # 设置colorbar的标签
        cb.ax.text(s=cb_label, x=-0.03, y=0.3, va="center", ha="right",
                   transform=cb_ax.transAxes, fontsize=cb_FontSize + 2)

        # 为误差图的colorbar添加外边界（黑边）
        cb.outline.set_edgecolor('black')
        cb.outline.set_linewidth(1.5)

    # 保存图片
    for suffix in g_suffix:
        figName = f"combined_NS_T{selected_time}_FFT_Error_wavenumber_space.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=fig_DPI, bbox_inches='tight')
        print(f"已保存: {figFile}")

    plt.close()
    print("傅里叶空间相对误差分析完成！")