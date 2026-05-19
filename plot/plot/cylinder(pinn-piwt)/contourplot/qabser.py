import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
import scipy.io as sio
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

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
cmap_name = 'myCmapName'
cmap0 = plt.get_cmap("gist_rainbow_r")
# truncated the standard colormap
minval = 0.2
maxval = 1

# 获取原始colormap的颜色（包含alpha通道）
colors_original = cmap0(np.linspace(minval, maxval, numbin))

# 创建修改后的colormap，使中间部分变为白色
colors_modified = colors_original.copy()
mid_idx = numbin // 2
white_range = int(numbin * 0.4)
start_idx = mid_idx - white_range // 2
end_idx = mid_idx + white_range // 2
start_idx = max(0, start_idx)
end_idx = min(numbin, end_idx)
for i in range(start_idx, end_idx):
    dist_to_center = abs(i - mid_idx) / (white_range / 2)
    if dist_to_center < 1:
        alpha = dist_to_center
        white_color = np.array([1, 1, 1, 1.0])
        colors_modified[i] = (1 - alpha) * white_color + alpha * colors_original[i]

cmap1 = mpl.colors.LinearSegmentedColormap.from_list(
    'trunc_white_center({n},{a:.2f},{b:.2f})'.format(n=cmap0.name, a=minval, b=maxval), colors_modified)

# 为DNS参考场使用rainbow colormap（与FFT误差代码一致）
cmap_rainbow0 = plt.get_cmap("gist_rainbow_r")
minval_rainbow = 0.2
maxval_rainbow = 1
colors_rainbow = cmap_rainbow0(np.linspace(minval_rainbow, maxval_rainbow, numbin))
cmap_rainbow = mpl.colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_rainbow0.name, a=minval_rainbow, b=maxval_rainbow), colors_rainbow)

# 为绝对误差使用灰度colormap
cmap_gray0 = plt.get_cmap("Greys")
minval_gray = 0
maxval_gray = 1
colors_gray = cmap_gray0(np.linspace(minval_gray, maxval_gray, numbin))
cmap_gray = mpl.colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_gray0.name, a=minval_gray, b=maxval_gray), colors_gray)

###################################################################################

def compute_vorticity(u, v, x, y):
    """
    计算涡量: ω = ∂v/∂x - ∂u/∂y
    """
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    dvdx = np.gradient(v, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    vorticity = dvdx - dudy
    return vorticity

def compute_q_criterion(u, v, x, y):
    """
    计算Q判据: Q = 0.5 * (||Ω||^2 - ||S||^2)
    """
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]
    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)
    S11 = dudx
    S12 = 0.5 * (dudy + dvdx)
    S22 = dvdy
    Omega12 = 0.5 * (dvdx - dudy)
    S_norm2 = S11**2 + 2*S12**2 + S22**2
    Omega_norm2 = 2*Omega12**2
    Q = 0.5 * (Omega_norm2 - S_norm2)
    return Q

def compute_absolute_error(pred, true):
    """
    计算物理空间的绝对误差: |pred - true|
    """
    return np.abs(pred - true)

if __name__ == "__main__":
    ###################################################################################
    ############################# file Path of input data #############################
    cwd = os.getcwd()
    pdir1 = os.path.dirname(cwd)
    pdir2 = os.path.dirname(pdir1)
    filePath = os.path.join(pdir1)
    workPath = os.path.join(cwd)
    fig_Path = os.path.join(cwd, "figure_vorticity_Q_combined")
    g_suffix = ["pdf", "png"]
    ###################################################################################
    ############################ create the graph directory ###########################
    if not os.path.exists(fig_Path):
        os.makedirs(fig_Path)
    ###################################################################################
    ############################# profile file settings ###############################
    numStep = 50000
    str_numStep = "{}".format(numStep)
    solverPath = os.path.join(filePath)
    fieldPath = os.path.join(filePath, str_numStep)
    ###################################################################################
    # x & y labels
    xlabel = r"$x$"
    ylabel = r"$y$"
    ###################################################################################

    # 设置要绘制的时间点
    aim_T = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    selected_time = 1  # 可以选择 0, 1, 2, 3, 4, 5, 6, 7

    # 定义模型顺序
    models = ['DNS', 'PINN', 'PIKAN', 'PIWT']
    quantities = ['Vorticity', 'Q-criterion']

    # 为每个物理量设置colorbar范围
    quantity_ranges = {
        'Vorticity': {'min': -5, 'max': 5, 'inc': 2, 'label': r"$\omega$"},
        'Q-criterion': {'min': -1, 'max': 1, 'inc': 0.5, 'label': r"$Q$"}
    }

    # 创建子图注释 - 2行4列的顺序：
    # 第一行：DNS Vorticity, PINN Vorticity, PIKAN Vorticity, PIWT Vorticity
    # 第二行：DNS Q-criterion, PINN Q-criterion, PIKAN Q-criterion, PIWT Q-criterion
    subplot_labels = [
        r'(a) DNS Vorticity', r'(b) PINN Vorticity',
        r'(c) PIKAN Vorticity', r'(d) PIWT Vorticity',
        r'(e) DNS Q-criterion', r'(f) PINN Q-criterion',
        r'(g) PIKAN Q-criterion', r'(h) PIWT Q-criterion'
    ]

    # 保持每个子图的长宽比与原来一致
    fig, axes = plt.subplots(2, 4, figsize=(38, 10))

    ###################################################################################
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

    nlevel = 101

    # Colorbar设置 - 改为竖直放置在右侧
    cb_width = 0.01
    cb_height = 0.4
    cb_fmt = mpl.ticker.StrMethodFormatter("{x:g}")
    cb_FontSize = fontSize

    print(f"正在绘制 t = {selected_time} 时刻的涡量和Q判据...")

    # 获取指定时间点的原始数据点
    t_n = selected_time
    X = X_star[:, 0]
    Y = X_star[:, 1]
    xy_points = np.c_[X, Y]

    # 存储所有contour对象用于colorbar
    contour_plots_vorticity = []
    contour_plots_q = []

    # 循环绘制8个子图（2行4列）
    for idx, ax in enumerate(axes.flat):
        row = idx // 4  # 0: 第一行(涡量), 1: 第二行(Q判据)
        col = idx % 4   # 0: DNS, 1: PINN, 2: PIKAN, 3: PIWT

        model = models[col]
        quantity = quantities[row]

        q_range = quantity_ranges[quantity]
        q_min = q_range['min']
        q_max = q_range['max']
        q_inc = q_range['inc']
        cb_label = q_range['label']

        label = subplot_labels[idx]

        # 加载速度数据
        if model == 'DNS':
            u_filename = f"u_trueNS_T{t_n}.npy"
            v_filename = f"v_trueNS_T{t_n}.npy"
        else:
            u_filename = f"u_predNS_T{t_n}_{model.lower()}.npy"
            v_filename = f"v_predNS_T{t_n}_{model.lower()}.npy"

        try:
            u_data = np.load(u_filename)
            v_data = np.load(v_filename)

            u_interp = griddata(xy_points, u_data.flatten(), (XX_grid, YY_grid), method='cubic')
            v_interp = griddata(xy_points, v_data.flatten(), (XX_grid, YY_grid), method='cubic')

            if quantity == 'Vorticity':
                field_interp = compute_vorticity(u_interp, v_interp, XX_grid, YY_grid)
            else:
                field_interp = compute_q_criterion(u_interp, v_interp, XX_grid, YY_grid)

        except FileNotFoundError as e:
            print(f"文件不存在: {e}")
            field_interp = np.zeros_like(XX_grid)

        # 设置坐标轴样式
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

        # 坐标轴控制：只保留最左侧列的y轴和最下侧行的x轴
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

        if row == 1:
            ax.set_xlabel(xlabel)
            ax.xaxis.set_tick_params(which='major', size=7, width=2.5, direction='in', top=False)
            ax.xaxis.set_tick_params(which='minor', size=4, width=2.5, direction='in', top=False)
            ax.xaxis.set_tick_params(pad=8)
        else:
            ax.set_xlabel('')
            ax.xaxis.set_tick_params(which='major', size=0, width=0, direction='in', top=False)
            ax.xaxis.set_tick_params(which='minor', size=0, width=0, direction='in', top=False)
            ax.set_xticklabels([])
            ax.xaxis.set_visible(False)

        xinterval = 1
        yinterval = 1
        xticks = np.arange(xlimit[0], xlimit[1] + xinterval, xinterval)
        yticks = np.arange(ylimit[0], ylimit[1] + yinterval, yinterval)
        xticks = np.around(xticks, decimals=2)
        yticks = np.around(yticks, decimals=2)

        if row == 1:
            ax.set_xticks(xticks)
            xtickLabels = ["{:.0f}".format(xticks[0])] + ["{:.2f}".format(xticks[i]) for i in range(1, len(xticks))]
            ax.set_xticklabels(xtickLabels)

        if col == 0:
            ax.set_yticks(yticks)
            ytickLabels = ["{:.0f}".format(yticks[0])] + ["{:.1f}".format(yticks[i]) for i in range(1, len(yticks))]
            ax.set_yticklabels(ytickLabels)

        if row == 1:
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        else:
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())

        if col == 0:
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        else:
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

        formatter = mpl.ticker.StrMethodFormatter("{x:g}")
        if row == 1:
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if col == 0:
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        ax.text(0.02, 0.02, label, transform=ax.transAxes,
                fontsize=fontSize - 2, ha='left', va='bottom',
                color='black')

        levels = np.linspace(q_min, q_max, nlevel)
        CS = ax.contourf(XX_grid, YY_grid, field_interp, vmin=q_min, vmax=q_max,
                         levels=levels, cmap=cmap1, extend="both")

        if quantity == 'Vorticity':
            contour_plots_vorticity.append(CS)
        else:
            contour_plots_q.append(CS)

        cmap_current = CS.get_cmap()
        cmap_current.set_over(colors_modified[-1])
        cmap_current.set_under(colors_modified[0])
        CS.set_cmap(cmap_current)

    # 创建两个竖直colorbar放在右侧
    cb_ax_vorticity = fig.add_axes([0.95, 0.53, cb_width, cb_height])
    cb_ticks_vorticity = np.arange(-5, 6, 2)
    cb_vorticity = fig.colorbar(contour_plots_vorticity[0], ticks=cb_ticks_vorticity, cax=cb_ax_vorticity,
                                format=cb_fmt, orientation='vertical', extendfrac=0)
    cb_vorticity.ax.tick_params(labelsize=cb_FontSize, length=12, width=1.5)
    cb_vorticity.ax.text(s=r"$\omega$", x=0.3, y=1.05, va="center", ha="center",
                         transform=cb_ax_vorticity.transAxes, fontsize=cb_FontSize + 2)
    cb_vorticity.outline.set_edgecolor('black')
    cb_vorticity.outline.set_linewidth(2.0)

    cb_ax_q = fig.add_axes([0.95, 0.08, cb_width, cb_height])
    cb_ticks_q = np.arange(-1, 1.1, 0.5)
    cb_q = fig.colorbar(contour_plots_q[0], ticks=cb_ticks_q, cax=cb_ax_q,
                        format=cb_fmt, orientation='vertical', extendfrac=0)
    cb_q.ax.tick_params(labelsize=cb_FontSize, length=12, width=1.5)
    cb_q.ax.text(s=r"$Q$", x=0.3, y=1.05, va="center", ha="center",
                 transform=cb_ax_q.transAxes, fontsize=cb_FontSize + 2)
    cb_q.outline.set_edgecolor('black')
    cb_q.outline.set_linewidth(2.0)

    plt.subplots_adjust(left=0.08, right=0.93, top=0.95, bottom=0.08, wspace=0.05, hspace=0.05)

    for suffix in g_suffix:
        figName = f"vorticity_Q_T{selected_time}_combined.{suffix}"
        figFile = os.path.join(fig_Path, figName)
        plt.savefig(figFile, dpi=fig_DPI, bbox_inches='tight')
        print(f"Width x Height: {fig.get_size_inches()}, suffix: {suffix}")
        print(f"已保存: {figFile}")
    plt.close()

    ###################################################################################
    # 新增：绘制物理空间绝对误差云图（布局与FFT误差代码一致，4行2列）
    ###################################################################################
    print("正在绘制绝对误差云图...")
    # 定义需要计算绝对误差的模型（排除DNS）
    error_models = ['PINN', 'PIKAN', 'PIWT']
    # 物理量列表（涡量和Q判据）
    phys_quantities = ['Vorticity', 'Q-criterion']
    # 每个物理量的显示标签和colorbar范围（用于DNS参考场）
    phys_ranges = {
        'Vorticity': {'min': -5, 'max': 5, 'label': r"$\omega$"},
        'Q-criterion': {'min': -1, 'max': 1, 'label': r"$Q$"}
    }

    # 先计算所有模型（DNS及误差模型）在所选时刻的涡量和Q判据场，存入字典
    fields = {}
    for model in ['DNS'] + error_models:
        fields[model] = {}
        for qty in phys_quantities:
            if model == 'DNS':
                u_file = f"u_trueNS_T{t_n}.npy"
                v_file = f"v_trueNS_T{t_n}.npy"
            else:
                u_file = f"u_predNS_T{t_n}_{model.lower()}.npy"
                v_file = f"v_predNS_T{t_n}_{model.lower()}.npy"
            try:
                u_data = np.load(u_file)
                v_data = np.load(v_file)
                u_interp = griddata(xy_points, u_data.flatten(), (XX_grid, YY_grid), method='cubic')
                v_interp = griddata(xy_points, v_data.flatten(), (XX_grid, YY_grid), method='cubic')
                if qty == 'Vorticity':
                    field = compute_vorticity(u_interp, v_interp, XX_grid, YY_grid)
                else:
                    field = compute_q_criterion(u_interp, v_interp, XX_grid, YY_grid)
                fields[model][qty] = field
            except Exception as e:
                print(f"计算 {model} {qty} 时出错: {e}")
                fields[model][qty] = np.zeros_like(XX_grid)

    # 设置绝对误差图的参数（使用对数色阶，范围 0.01 ~ 1，可根据实际数据调整）
    # 涡量绝对误差通常小于1，Q判据绝对误差通常小于0.5，因此上限设为1足够
    nlevel_err = 101
    err_min = 0.01      # 10^-2
    err_max = 1.0       # 10^0
    levels_log = np.linspace(np.log10(err_min), np.log10(err_max), nlevel_err)
    levels_err = 10 ** levels_log

    # 创建图形，使用GridSpec精确控制行间距
    fig_err = plt.figure(figsize=(22, 16))
    gap_height = 0.08
    gs1 = gridspec.GridSpec(1, 2, figure=fig_err,
                            top=0.88, bottom=0.88 - 0.2,
                            left=0.08, right=0.95,
                            wspace=0.1)
    first_row_bottom = 0.88 - 0.2
    second_row_top = first_row_bottom - gap_height
    gs2 = gridspec.GridSpec(3, 2, figure=fig_err,
                            top=second_row_top, bottom=0.08,
                            left=0.08, right=0.95,
                            hspace=0.08, wspace=0.1)

    axes_err = np.zeros((4, 2), dtype=object)
    for j in range(2):
        axes_err[0, j] = fig_err.add_subplot(gs1[0, j])
    for i in range(3):
        for j in range(2):
            axes_err[i+1, j] = fig_err.add_subplot(gs2[i, j])

    dns_labels = [r'(a) DNS Vorticity', r'(b) DNS Q-criterion']
    err_labels = [
        r'(c) PINN:Vorticity (Abs. Error)', r'(d) PINN:Q-criterion (Abs. Error)',
        r'(e) PIKAN:Vorticity (Abs. Error)', r'(f) PIKAN:Q-criterion (Abs. Error)',
        r'(g) PIWT:Vorticity (Abs. Error)',  r'(h) PIWT:Q-criterion (Abs. Error)'
    ]

    # 第一行：DNS参考场
    for col, qty in enumerate(phys_quantities):
        ax = axes_err[0, col]
        field_dns = fields['DNS'][qty]
        q_min = phys_ranges[qty]['min']
        q_max = phys_ranges[qty]['max']
        levels_dns = np.linspace(q_min, q_max, nlevel)
        CS_dns = ax.contourf(XX_grid, YY_grid, field_dns,
                             vmin=q_min, vmax=q_max,
                             levels=levels_dns, cmap=cmap_rainbow, extend="both")
        cmap_rainbow.set_over(colors_rainbow[-1])
        cmap_rainbow.set_under(colors_rainbow[0])
        CS_dns.set_cmap(cmap_rainbow)

        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)
        ax.set_xlim(1, 8)
        ax.set_ylim(-2, 2)

        if col == 0:
            ax.set_ylabel(ylabel)
            ax.yaxis.set_tick_params(which='major', size=7, width=2.5, direction='in', right=False)
            ax.yaxis.set_tick_params(which='minor', size=4, width=2.5, direction='in', right=False)
            ax.yaxis.set_tick_params(pad=8)
        else:
            ax.set_ylabel('')
            ax.yaxis.set_visible(False)

        ax.set_xlabel('')
        ax.xaxis.set_visible(False)

        xticks = np.arange(1, 9, 1)
        yticks = np.arange(-2, 3, 1)
        if col == 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{y:.0f}" for y in yticks])
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.text(0.02, 0.02, dns_labels[col], transform=ax.transAxes,
                fontsize=fontSize-2, ha='left', va='bottom', color='black')

        cb_ax_dns = fig_err.add_axes([0.1 + col*0.5, 0.91, 0.25, 0.02])
        num_ticks = 5
        cb_ticks_dns = np.linspace(q_min, q_max, num_ticks)
        cb_ticks_dns = np.round(cb_ticks_dns, 1)
        cb_dns = fig_err.colorbar(CS_dns, ticks=cb_ticks_dns, cax=cb_ax_dns,
                                  format=mpl.ticker.StrMethodFormatter("{x:g}"),
                                  orientation='horizontal', extendfrac=0)
        cb_dns.ax.tick_params(labelsize=fontSize-4, length=12, width=0.4)
        cb_dns.ax.text(s=phys_ranges[qty]['label'], x=-0.05, y=0.3, va="center", ha="right",
                       transform=cb_ax_dns.transAxes, fontsize=fontSize-2)
        cb_dns.outline.set_edgecolor('black')
        cb_dns.outline.set_linewidth(1.5)

    # 下面三行：绝对误差图（下限0.01，上限1，对数色阶）
    contour_err_list = []
    for idx, (ax, label) in enumerate(zip(axes_err[1:].flat, err_labels)):
        row = idx // 2
        col = idx % 2
        model = error_models[row]
        qty = phys_quantities[col]

        pred_field = fields[model][qty]
        true_field = fields['DNS'][qty]
        abs_err = compute_absolute_error(pred_field, true_field)
        # 截断到 [err_min, err_max]
        abs_err = np.clip(abs_err, err_min, err_max)

        norm = LogNorm(vmin=err_min, vmax=err_max)
        CS_err = ax.contourf(XX_grid, YY_grid, abs_err,
                             levels=levels_err, norm=norm,
                             cmap=cmap_gray, extend="both")
        contour_err_list.append(CS_err)
        cmap_gray.set_over(cmap_gray(1.0))
        cmap_gray.set_under(cmap_gray(0.0))
        CS_err.set_cmap(cmap_gray)

        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)
        ax.set_xlim(1, 8)
        ax.set_ylim(-2, 2)

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
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{x:.0f}" for x in xticks])
            ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        else:
            ax.set_xlabel('')
            ax.xaxis.set_visible(False)

        if col == 0:
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{y:.0f}" for y in yticks])
            ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

        ax.text(0.02, 0.02, label, transform=ax.transAxes,
                fontsize=fontSize-2, ha='left', va='bottom', color='black')

    # 公共colorbar（刻度：0.01, 0.1, 1，显示为10^{-2}, 10^{-1}, 10^{0}）
    if len(contour_err_list) > 0:
        cb_ax_err = fig_err.add_axes([0.12, 0.63, 0.83, 0.02])
        cb_err = fig_err.colorbar(contour_err_list[0], cax=cb_ax_err,
                                  orientation='horizontal', extendfrac=0)
        # 设置对数刻度位置
        cb_err.set_ticks([0.01, 0.1, 1.0])
        cb_err.set_ticklabels([r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
        cb_err.ax.tick_params(labelsize=cb_FontSize, length=22, width=0.4)
        cb_err.ax.text(s=r"Absolute Error", x=-0.03, y=0.3, va="center", ha="right",
                       transform=cb_ax_err.transAxes, fontsize=cb_FontSize+2)
        cb_err.outline.set_edgecolor('black')
        cb_err.outline.set_linewidth(1.5)

    for suffix in g_suffix:
        figName_err = f"absolute_error_vorticity_Q_T{selected_time}.{suffix}"
        figFile_err = os.path.join(fig_Path, figName_err)
        plt.savefig(figFile_err, dpi=fig_DPI, bbox_inches='tight')
        print(f"绝对误差图已保存: {figFile_err}")
    plt.close(fig_err)
    print("所有图形绘制完成！")