import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error


def showPlotFile(xfName="", prfx="./xpydm.zip/data/"):
    data_uri = open(prfx + xfName, 'rb').read().encode('base64').replace('\n', '')
    img_tag = '%html <img src="data:image/png;base64,{0}">'.format(data_uri)
    print(img_tag)


def show_heatmap(xdf_input, target_col, xselekt_cols):
    f, ax = plt.subplots(figsize=(12, 10))
    _ = sns.heatmap(xdf_input[[target_col] + xselekt_cols].corr())


def show_corrplot_df(xABC, save2file=False, xcmap=plt.cm.BrBG, xrotation=0.0, yrotation=0.0,
                     xtitle="Korrelationen"):
    f = plt.figure(dpi=120)  # figsize=(12, 10)
    data = xABC.corr()
    plt.matshow(data, fignum=f.number, cmap=xcmap)  #
    plt.xticks(list(range(xABC.shape[1])), xABC.columns, fontsize=10, rotation=xrotation)
    plt.yticks(list(range(xABC.shape[1])), xABC.columns, fontsize=10, rotation=yrotation)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.title(xtitle)
    plt.tight_layout()
    for (i, j), z in np.ndenumerate(data):
        plt.text(j, i, '{:0.1f}'.format(z))
    if save2file:
        plt.savefig("./xpydm.zip/data/outfig.png", dpi=100)
        showPlotFile(xfName="outfig.png")
    else:
        plt.show()


def show_gridhist(xdf, xby=(5, 4), save2file=False):
    """ xby wie groß sollte der Grid sein (5,4) => 5 Zeilen, 4 Spalten"""
    xcolumns = xdf.columns
    print("[x] Histogramme für:", xcolumns.tolist())
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    if xcolumns.shape[0] > (xby[0] * xby[1]):
        print("[x] xby zu niedrig eingestellt")
        return
    fig, ax = plt.subplots(xby[1], xby[0], figsize=(12, 8), dpi=120)
    fig.tight_layout()
    for i in range(xby[1]):
        for j in range(xby[0]):
            k = i * xby[0] + j
            if k < xcolumns.shape[0]:
                ax[i, j].hist(xdf[xcolumns[k]])
                ax[i, j].set_title(xcolumns[k])
    plt.tight_layout()
    if save2file:
        plt.savefig("./xpydm.zip/data/outfig.png", dpi=120)
        showPlotFile(xfName="outfig.png")
    else:
        plt.show()


def do_pairplot(xdf, xselekt, save2file=False):
    """
        # Pairplot für enge Auswahl Features
    """
    from matplotlib.artist import setp
    xstrMap = lambda xstr: [x for x in xstr]
    xstrPad = lambda xstr: ''.join(([' '] * (10 - len(xstr))) + xstrMap(xstr)) if len(xstr) < 10 else xstr
    axs = pd.plotting.scatter_matrix(xdf[xselekt].dropna(),
                                     figsize=(16, 8),
                                     marker='o', hist_kwds={'bins': 25}, s=8, alpha=.4)
    for row in axs:
        for j in range(len(row)):
            subplot = row[j]
            setp(subplot.get_xticklabels(), rotation=0)
            setp(subplot.get_yticklabels(), rotation=0)
            ylabel = subplot.get_ylabel()
            ylabel = ylabel[:10] + (".." if len(ylabel) >= 10 else "")
            xlabel = subplot.get_xlabel()
            yticks = [str(item) for item in subplot.get_yticks()]
            if len([x for x in yticks if x != ""]) != 0:
                subplot.set_yticklabels([xstrPad(str(np.round(float(lbl), 2))) for lbl in yticks])
            subplot.set_ylabel(ylabel, rotation=90, fontdict={'fontsize': 10, 'fontweight': 'bold'})
            subplot.set_xlabel(xlabel, rotation=0, fontdict={'fontsize': 10, 'fontweight': 'bold'})
    plt.tight_layout()
    if save2file:
        plt.savefig("./xpydm.zip/data/outfig.png", dpi=120)
        showPlotFile(xfName="outfig.png")
    else:
        plt.show()


def show_hist_lmres(xdf_cols_stats):
    """
        Gegeben ein DataFrame plot Histogrammen
    :param xdf_cols_stats:
    :return:
    """
    xdf_cols_stats.hist(bins=25, grid=False)
    plt.autoscale(enable=True)
    # plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches(18, 16)
    plt.show()


def regression_ist_vs_pred(ist_werte, voraussage, target_col, jitter_on=(1,1)):
    """
        Regressionsresultate in Streudiagramm visualisieren
    :param ist_werte:
    :param voraussage:
    :param target_col:
    :param jitter_on:
    :return:
    """
    def rand_jitter(arr):
        stdev = .05 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None,
               vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
        return plt.scatter(rand_jitter(x) if jitter_on[0] == 1 else x,
                           rand_jitter(y) if jitter_on[1] == 1 else y,
                           s=s, c=c, marker=marker, cmap=cmap, norm=norm,
                           vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)

    if True:
        r2_perf = r2_score(ist_werte, voraussage)
        rmse_perf = np.sqrt(mean_squared_error(ist_werte, voraussage))
        print(r2_perf, rmse_perf)
        f, ax = plt.subplots(figsize=(10, 8))
        # plt.scatter(x=df_voraussage[target_col].values, y=xpred, s=4, alpha=0.3);
        jitter(x=ist_werte, y=voraussage, s=2, alpha=0.2)
        plt.xlabel(target_col)
        plt.ylabel("Voraussage ({})".format(target_col))
        plt.title("IST-Werte vs Voraussage mit Jitter")
        plt.show()