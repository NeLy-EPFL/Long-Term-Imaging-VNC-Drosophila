import os
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import matplotlib
matplotlib.use('agg')  # use non-interactive backend for pdf plotting
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.stats import ttest_ind, mannwhitneyu
from skimage.color import label2rgb
from PIL import ImageColor

from twoppp import utils, load, rois, denoise
from twoppp import plot as myplt
from twoppp.pipeline import PreProcessFly

from revision_flies import all_flies, high_caff_flies, high_caff_main_fly, low_caff_flies, low_caff_main_fly, sucr_flies, sucr_main_fly
from preprocessing_parameters import params
from make_caff_figures import compute_summary_stats, compute_twop_dfs, compute_rest_maps, make_dff_maps, make_detailled_wave_plots, make_trial_wave_plot, main_figure, sup_fig_all_waves

FILE_PATH = os.path.realpath(__file__)
CAFF_PATH, _ = os.path.split(FILE_PATH)
OUTPUT_PATH = os.path.join(CAFF_PATH, "outputs")
utils.makedirs_safe(OUTPUT_PATH)

colors = [myplt.DARKBLUE, myplt.DARKBLUE_CONTRAST, myplt.DARKCYAN, myplt.DARKGREEN, myplt.DARKGREEN_CONTRAST,
          myplt.DARKYELLOW, myplt.DARKORANGE, myplt.DARKPINK, myplt.DARKRED, myplt.DARKPURPLE,
          myplt.DARKBROWN, myplt.DARKGRAY, myplt.BLACK]


def trial_wave_figure():
    fig = plt.figure(figsize=(15,10))
    layout = """
    AAAAABBBBBCCCCC
    DDDDDEEEEEFFFFF
    GGGGGHHHHHIIIII
    
    """
    
    ax_dict = fig.subplot_mosaic(layout)

    overwrite = False
    n_norm = 2
    normalise = 0.99

    # row 1: trial wave plots
    ylim = make_trial_wave_plot(high_caff_main_fly, ax=ax_dict["C"], legend=False, use_denoised=False,
                                overwrite=overwrite, normalise=normalise, n_norm=n_norm)

    _ = make_trial_wave_plot(low_caff_main_fly, ylim=ylim, ax=ax_dict["B"], legend=False,
                                overwrite=overwrite, normalise=normalise, n_norm=n_norm)

    _ = make_trial_wave_plot(sucr_main_fly, ylim=ylim, ax=ax_dict["A"], legend=True, use_denoised=False,
                                overwrite=overwrite, normalise=normalise, n_norm=n_norm)

    _ = make_trial_wave_plot(high_caff_flies[1], ylim=ylim, ax=ax_dict["F"], legend=False, use_denoised=False,
                                overwrite=overwrite, normalise=normalise, n_norm=n_norm)

    _ = make_trial_wave_plot(low_caff_flies[1], ylim=ylim, ax=ax_dict["E"], legend=False, use_denoised=False,
                                overwrite=overwrite, normalise=normalise, n_norm=n_norm)

    _ = make_trial_wave_plot(sucr_flies[1], ylim=ylim, ax=ax_dict["D"], legend=True, use_denoised=False,
                                overwrite=overwrite, normalise=normalise, n_norm=n_norm)

    _ = make_trial_wave_plot(high_caff_flies[2], ylim=ylim, ax=ax_dict["I"], legend=False, use_denoised=False,
                                overwrite=overwrite, normalise=normalise, n_norm=n_norm)

    _ = make_trial_wave_plot(low_caff_flies[2], ylim=ylim, ax=ax_dict["H"], legend=False, use_denoised=False,
                                overwrite=overwrite, normalise=normalise, n_norm=n_norm)

    _ = make_trial_wave_plot(sucr_flies[2], ylim=ylim, ax=ax_dict["G"], legend=True, use_denoised=False,
                                overwrite=overwrite, normalise=normalise, n_norm=n_norm)
 
    
    fig.tight_layout()
    return fig

def plot_significance_bracket(ax, x, p, h=10.25, dx=0.2, dy=0.1, fontsize=8):
    if p >= 0.05:
        sign = 0
    elif p < 0.05 and p >= 0.01:
        sign = 1
    else:
        sign = np.floor(np.abs(np.log10(p))-2*np.finfo(float).eps)

    x1, x2, y, h, col = x-dx, x+dx, h, dy, 'k'
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    ax.text((x1+x2)*.5, y+h, "*"*sign+"ns"*np.logical_not(sign), ha='center', va='bottom', color=col, fontsize=fontsize)

def statistics_figure(flies, normalise=0.99, n_norm=2, test=mannwhitneyu):
    # conditions, indices = np.unique([fly.paper_condition for fly in flies], return_index=True)
    for i_fly, fly in enumerate(flies):
        with open(fly.trials_mean_dff, "rb") as f:
            flies[i_fly].dffs = pickle.load(f)
        
        if normalise:
            flies[i_fly].low = np.quantile(flies[i_fly].dffs[:,:n_norm], (1-normalise)/2)
            flies[i_fly].high = np.quantile(flies[i_fly].dffs[:,:n_norm], (1+normalise)/2)
            flies[i_fly].dffs = (flies[i_fly].dffs - flies[i_fly].low) / (flies[i_fly].high - flies[i_fly].low)

    # concatenate pre-feeding, during feeding and then two consecutive post feeding trials each
    i_concat = [[0,1], [2], [3,4], [5,6], [7,8], [9,10]]
    N_test = len(i_concat)
    quantile_concat = np.zeros((len(flies), len(i_concat)))

    for i_fly, fly in enumerate(flies):
        dffs = deepcopy(fly.dffs)
        for i_c, concat in enumerate(i_concat):
            this_dff = dffs[:, concat]
            quantile_concat[i_fly, i_c] = np.quantile(this_dff, q=(1+normalise)/2)
    
    quantile_high = quantile_concat[["high" in fly.paper_condition for fly in flies]].T
    quantile_low = quantile_concat[["low" in fly.paper_condition for fly in flies]].T
    quantile_sucr = quantile_concat[["sucr" in fly.paper_condition for fly in flies]].T

    np.savetxt(os.path.join(OUTPUT_PATH, f"stats_highcaff_{datestring}.csv"), quantile_high, delimiter=",")
    np.savetxt(os.path.join(OUTPUT_PATH, f"stats_lowcaff_{datestring}.csv"), quantile_low, delimiter=",")
    np.savetxt(os.path.join(OUTPUT_PATH, f"stats_sucrose_{datestring}.csv"), quantile_sucr, delimiter=",")

    # compute p values of statistical tests
    ps_high_vs_low = []
    for i in range(1,N_test):
        p = test(quantile_high[i,:], quantile_low[i,:], alternative="greater").pvalue
        ps_high_vs_low.append(p)
    
    ps_high_vs_sucr = []
    for i in range(1,N_test):
        p = test(quantile_high[i,:], quantile_sucr[i,:], alternative="greater").pvalue
        ps_high_vs_sucr.append(p)

    fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.plot(np.arange(-1,5)-0.23, quantile_sucr[:,0], myplt.DARKGRAY, marker=".", linestyle="",
        label="sucrose")
    ax.plot(np.arange(-1,5)-0.2, quantile_sucr[:,1], myplt.DARKGRAY, marker=".", linestyle="",
        label=None)
    ax.plot(np.arange(-1,5)-0.17, quantile_sucr[:,2], myplt.DARKGRAY, marker=".", linestyle="",
        label=None)

    ax.plot(np.arange(-1,5)-0.03, quantile_low[:,0], myplt.BLACK, marker=".", linestyle="",
        label="low caffeine")
    ax.plot(np.arange(-1,5), quantile_low[:,1], myplt.BLACK, marker=".", linestyle="",
        label=None)
    ax.plot(np.arange(-1,5)+0.03, quantile_low[:,2], myplt.BLACK, marker=".", linestyle="",
        label=None)

    ax.plot(np.arange(-1,5)+0.17, quantile_high[:,0], myplt.DARKRED, marker=".", linestyle="",
        label="high caffeine")
    ax.plot(np.arange(-1,5)+0.2, quantile_high[:,1], myplt.DARKRED, marker=".", linestyle="",
        label=None)
    ax.plot(np.arange(-1,5)+0.23, quantile_high[:,2], myplt.DARKRED, marker=".", linestyle="",
        label=None)
    # ax.plot([-1,4], [1, 1], "k")  # , linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, loc="center left")
    # ax.spines['bottom'].set_position(("data",1))
    ax.spines['bottom'].set_position(("data",0))
    ax.set_xticks(np.arange(-1,5))
    ax.set_xticklabels(["<21 min before", "during", "<9 min", "<19 min", "<29 min", "<38 min after"], fontsize=8)
    ax.set_xlabel("time relative to beginning of feeding")
    ax.set_yticks(np.arange(10))
    ax.set_ylabel("maximum normalised fluorescence", fontsize=8)  #  relative to before feeding

    for i_test in range(N_test-1):
        plot_significance_bracket(ax, x=i_test, p=ps_high_vs_sucr[i_test])
        plot_significance_bracket(ax, x=i_test+0.1, p=ps_high_vs_low[i_test], dx=0.1, h=9.5)

    return fig

def spontaneous_fluctuation_figure(fly, i_trial=0, normalise=0.99, n_norm=2, N_denoise=30):
    opflow_df = fly.trials[i_trial].opflow_df
    twop_df = fly.trials[i_trial].twop_df

    opflow_df = pd.read_pickle(opflow_df)
    twop_df = pd.read_pickle(twop_df) 

    with open(fly.trials_mean_dff, "rb") as f:
        fly.dffs = pickle.load(f)
        
    if normalise:
        fly.low = np.quantile(fly.dffs[:,:n_norm], (1-normalise)/2)
        fly.high = np.quantile(fly.dffs[:,:n_norm], (1+normalise)/2)
        fly.dffs = (fly.dffs - fly.low) / (fly.high - fly.low)

    #remove the start and end samples that would be taken away by the denoising algorithm
    dff = fly.dffs[N_denoise:-(N_denoise+1), i_trial]
    rho = np.corrcoef(dff, twop_df.rest.values.astype(int))[0,1]

    fig, ax = plt.subplots(1,1, figsize=(6,4))
    ax.plot(twop_df.t.values, dff, color="k", label="mean fluorescence")
    # plt.plot(twop_df.t.values, twop_df.walk.values)
    myplt.shade_categorical(twop_df.rest.values.astype(int), x=twop_df.t.values, colors=["None", "k"], ax=ax, labels=[None, "animal stationary"])
    ax.legend(frameon=True, fontsize=8, edgecolor="w")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("nomalised fluorescence (a.u.)")
    ax.set_title(f"Mean fluoresence in CC encodes stationarity: r={rho:.2f}")

    return fig


def main():
    global datestring
    datestring = datetime.now().strftime("%Y%m%d_%H%M")
    
    with PdfPages(os.path.join(OUTPUT_PATH, f"_wave_figures_revision_{datestring}.pdf")) as pdf:  # nnorm3_
        fig = trial_wave_figure()
        # fig.savefig(os.path.join(OUTPUT_PATH, f'_wave_figures_revision_{datestring}_1.eps'), format='eps')
        # fig.savefig(os.path.join(OUTPUT_PATH, f'_wave_figures_revision_{datestring}_1.pdf'), transparent=True)
        pdf.savefig(fig)
        
        fig = statistics_figure(flies=deepcopy(all_flies))
        # fig.savefig(os.path.join(OUTPUT_PATH, f'_wave_figures_revision_{datestring}_2.eps'), format='eps')
        # fig.savefig(os.path.join(OUTPUT_PATH, f'_wave_figures_revision_{datestring}_2.pdf'), transparent=True)
        pdf.savefig(fig)
        
        fig = spontaneous_fluctuation_figure(deepcopy(all_flies[0]))
        # fig.savefig(os.path.join(OUTPUT_PATH, f'_wave_figures_revision_{datestring}_3.eps'), format='eps')
        # fig.savefig(os.path.join(OUTPUT_PATH, f'_wave_figures_revision_{datestring}_3.pdf'), transparent=True)
        pdf.savefig(fig)
    

if __name__ == "__main__":
    main()