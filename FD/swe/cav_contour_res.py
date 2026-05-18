import os
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from scipy.interpolate import griddata
import data
from scipy.io import loadmat
import scipy.io as sio
mpl.use('PS')

###################################################################################
##### The property of drawing plot using matplotlib #####
fig_DPI = 600
fontSize = 26
########## font: Time New Roman ##########
config = {
    "text.usetex": False,
    "font.family": "Times New Roman",
    "font.size": fontSize,
    "mathtext.fontset": 'cm',

    "xtick.direction":"in",
    "ytick.direction":"in",
}
mpl.rcParams.update(config)
# mpl.mathtext.FontConstantsBase.sup1 = 0.5
# mpl.mathtext.FontConstantsBase.sub1 = 0.4
# mpl.mathtext.FontConstantsBase.sub2 = 0.4
###################################################################################
##### The property of drawing graphs #####
lcolor = ['k','r','k','r','#FF0000', '#00D200', '#0000FF', '#FF00FF', '#000000', '#000000', '#000000', 'salmon', 'violet', 'yellowgreen']
lstyle = ["--", "--", "-", "--", ":", "-", "--", "-.", ":", "--", "-."]
lwidth = [2.5]*10
mshape = ["o","s", "^", "^","^", "o", "v", "+", "D", "s", "^", "v", "<", ">", "d", "*"]
mcolor = ['r', '#00D200', '#0000FF', '#FF00FF', '#FF8000', '#000000', "b", "k", 'gold', 'salmon', 'goldenrod', 'violet']
isolid = [True, False, False, True, True, False, False, False, False, False]
msizes = [20, 8, 10, 10, 50, 50, 50, 100, 25, 50, 100, 10, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
mstyle = ["-", "--", "-", "--", ":", "-", "--", "-.", ":", "--", "-."]
mwidth = [2.5]*10
###################################################################################
################################ set custom colormap #############################
numbin = 100
cmap_name = 'myCmapName'
# colors = ['blue','cyan','lawngreen','yellow','orange','red']  # Blue -> Green -> Red
# cmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=numbin)
cmap0 = plt.get_cmap("gist_rainbow_r")
# truncated the standard colormap
minval = 0.2
maxval = 1
colors = cmap0(np.linspace(minval, maxval, numbin))
cmap1 = mpl.colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap0.name, a=minval, b=maxval),colors)
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
    #x & y labels
    xlabel = r"$x$"
    ylabel = r"$t$"
    # title  = r"veloicty contour"
    ###################################################################################
    # 1. Read mesh
    _, _, _, _, _, input, label_h, label_u, _, _ = data.load_SWEdata()
    x = input[:, 0:1]
    t = input[:, 1:2]
    # data = np.loadtxt("../Re1000.dat", skiprows=2, max_rows=2601)
    # x = data[:, 0:1]
    # y = data[:, 1:2]
    x_point = x.flatten()
    t_point = t.flatten()
    xt_point = np.c_[x_point, t_point]
    #data = sio.loadmat("./dataset/Allen_Cahn.mat")
    nn = 500
    x = np.linspace(0, 1200, nn)
    t = np.linspace(0, 3600, nn)
    X, T = np.meshgrid(x, t)

    max_layer = 4

    for aaa in ['h', 'hu']:




        u_pred = np.log10(np.load(f"swe_res_{aaa}.npy")+1e-10)
        # u_true = np.load("u_trueAC.npy")
        # u_res = np.load("u_resAC.npy")
        UU_star = griddata(xt_point, u_pred.flatten(), (X, T), method='cubic')

        #print(contour_X.shape)
        fieldData1 = UU_star # shape: (T, N)
        # fieldData2 = u_true.T  # shape: (T, N)
        # fieldData3 = u_res.T  # shape: (T, N)
        print(np.max(fieldData1),np.min(fieldData1))
        contour_X = X
        contour_Y = T

        # contour_X = np.load("x.npy")
        # contour_Y = np.load("y.npy",)
        # fieldData = np.load("U.npy")
        ###################################################################################
        # 4. Contour level settings
        nlevel = 101
        if aaa == 'h':

            var_min = -2 #[0,-1e-6,-1e-6]
            var_max = 1#[0.5,1e-6,1e-6]
            var_inc = 0.5#[0.1,5e-7,5e-7]

        if aaa == 'hu':
            var_min = -2  # [0,-1e-6,-1e-6]
            var_max = 1  # [0.5,1e-6,1e-6]
            var_inc = 0.5  # [0.1,5e-7,5e-7]

        # 5. Colorbar (cb) settings
        cb_w  = 0.6
        cb_h  = 0.04
        cb_fmt = mpl.ticker.StrMethodFormatter("{x:g}")
        cb_FontSize = fontSize
        cb_label = r"$log_{10}(E_{r})$"#, r"$\mathdefault{\bar U_y}$", r"$\mathdefault{\bar U_z}$"]



        fig, ax = plt.subplots(figsize=(8, 8))
        # Hide the top and right spines of the axis
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        # Set the axis box line width
        ax.spines['bottom'].set_linewidth(2.5)
        ax.spines['left'].set_linewidth(2.5)
        ax.spines['top'].set_linewidth(2.5)
        ax.spines['right'].set_linewidth(2.5)
        # Tick Parameters: Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=7, width=2.5, direction='in', top=False)
        ax.xaxis.set_tick_params(which='minor', size=4, width=2.5, direction='in', top=False)
        ax.yaxis.set_tick_params(which='major', size=7, width=2.5, direction='in', right=False)
        ax.yaxis.set_tick_params(which='minor', size=4, width=2.5, direction='in', right=False)
        ax.xaxis.set_tick_params(pad=8)
        ax.yaxis.set_tick_params(pad=8)

        # (2) set the x & y limits
        xlimit = [0, 1200]
        ylimit = [0, 3600]
        ax.set_xlim(xlimit[0], xlimit[1])
        ax.set_ylim(ylimit[0], ylimit[1])

        # (3) set the x & y labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        # plt.title(title, x=0.5, y=0.92, fontsize = fontSize)

        # (4) set the x & y tickes and ticklabels
        xinterval = 200
        yinterval = 600
        xticks = np.arange(xlimit[0], xlimit[1] + xinterval, xinterval)
        yticks = np.arange(ylimit[0], ylimit[1] + yinterval, yinterval)
        xticks = np.around(xticks, decimals=2)
        yticks = np.around(yticks, decimals=2)
        numXticks = len(xticks)
        numYticks = len(yticks)
        # xtickLabels = xticks
        # ytickLabels = yticks
        xtickLabels = ["{:.0f}".format(xticks[0])] + ["{:.2f}".format(xticks[i]) for i in range(1, numXticks)]
        ytickLabels = ["{:.0f}".format(yticks[0])] + ["{:.1f}".format(yticks[i]) for i in range(1, numYticks)]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels(xtickLabels)
        ax.set_yticklabels(ytickLabels)
        ########  set the minor tick locator  ########
        # 设置次刻度
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(40))  # 设置 x 轴次刻度间隔为 0.5
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(60))  # 设置 y 轴次刻度间隔为 0.1

        # ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        # ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())
        ########  set the tick label format   ########
        formatter = mpl.ticker.StrMethodFormatter("{x:g}")
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

        contour_var2 = fieldData1
        varmin_ = var_min
        varmax_ = var_max
        varinc_ = var_inc
        levels = np.linspace(varmin_, varmax_, nlevel)
        CS2 = ax.contourf(contour_X, contour_Y, contour_var2, vmin=varmin_, vmax=varmax_, levels=levels, cmap=cmap1,
                          extend="both")
        cmap = CS2.get_cmap()
        cmap.set_over(colors[-1])
        cmap.set_under(colors[0])
        CS2.set_cmap(cmap)

        # (6) set the colorbar —— add axis: left, bottom, width, height
        cb_ax = fig.add_axes([0.27, 0.94, cb_w, cb_h])
        cb_ticks = np.arange(varmin_, varmax_ + varinc_, varinc_)
        cb = fig.colorbar(CS2, ticks=cb_ticks, cax=cb_ax, format=cb_fmt, orientation='horizontal', extendfrac=0)
        cb.ax.tick_params(labelsize=cb_FontSize, length=22, width=0.4)
        cb.ax.text(s=cb_label, x=-0.3, y=0.3, va="center", ha="left", transform=cb.ax.transAxes, fontsize=cb_FontSize)
        cb.outline.set_edgecolor('white')

        for suffix in g_suffix:
            figName = "contour_{}_{}.{}".format(aaa, fieldName1, suffix)
            figFile = os.path.join(fig_Path, figName)
            plt.savefig(figFile, dpi=fig_DPI, bbox_inches='tight')
            print("Width x Hight: ", fig.get_size_inches(), "suffix: ", suffix)
        plt.close()