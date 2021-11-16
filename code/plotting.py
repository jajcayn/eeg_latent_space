"""
Plotting helpers.
"""


import string

import matplotlib as mpl
import mne
import numpy as np
from matplotlib import pyplot as plt

MNE_LOGGING_LEVEL = "WARNING"
mne.set_log_level(MNE_LOGGING_LEVEL)


def plot_eeg_topomaps(
    topomaps,
    mne_info,
    xlabels=None,
    title="",
    plot_minmax_vec=False,
    fname=None,
    **kwargs,
):
    """
    Plots microstate maps.

    :param topomaps: microstate topographies to plot, no states x channels
    :type topomaps: np.ndarray
    :param mne_info: info from mne as per channels and locations
    :type mne_info: `mne.io.meas_info.Info`
    :param xlabels: labels for topomaps maps, usually correlation with
        template
    :type xlabels: list[str]
    :param title: title for the plot
    :type title: str
    :param plot_minmax_vec: whether to plot vector between minimum and maximum
        loading of the topographies
    :type plot_minmax_vec: bool
    :param fname: filename for the plot, if None, will show
    :type fname: str|None
    """

    plt.figure(figsize=((np.ceil(topomaps.shape[0] / 2.0)) * 5, 12))

    ms_names = list(string.ascii_uppercase)[: topomaps.shape[0]]

    if xlabels is None:
        xlabels = ["" for i in range(topomaps.shape[0])]

    for i, t, xlab in zip(range(topomaps.shape[0]), ms_names, xlabels):
        plt.subplot(2, int(np.ceil(topomaps.shape[0] / 2.0)), i + 1)
        mne.viz.plot_topomap(topomaps[i, :], mne_info, show=False, contours=10)

        if plot_minmax_vec:
            max_sen = np.argmax(topomaps[i, :])
            min_sen = np.argmin(topomaps[i, :])
            pos_int = mne.channels.layout._find_topomap_coords(
                mne_info, picks="eeg"
            )
            plt.gca().plot(
                [pos_int[min_sen, 0], pos_int[max_sen, 0]],
                [pos_int[min_sen, 1], pos_int[max_sen, 1]],
                "ko-",
                markersize=7,
                lw=2.2,
            )
        plt.title(t, fontsize=25)
        plt.xlabel(xlab, fontsize=22)

    plt.suptitle(title, fontsize=30)

    if fname is None:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", dpi=150, **kwargs)
    plt.close()


def plot_corr_matrix_w_maps(
    corr_mat,
    maps,
    mne_info,
    colorbar=True,
    cmap="bwr",
    minmax=[-1.0, 1.0],
    cbar_label="",
    dividers=[],
    **kwargs,
):
    """
    Plots correlation matrix with topomaps as labels.

    :param corr_mat: correlation matrix to plot
    :type corr_mat: np.ndarray
    :param maps: topomaps to use as labels, num maps x channels
    :type maps: np.ndarray
    :param mne_info: info from mne as per channels and locations
    :type mne_info: `mne.io.meas_info.Info`
    :param colorbar: whether to plot colorbar
    :type colorbar: bool
    :param cmap: colormap to use
    :type cmap: str
    :param minmax: minimum and maximum of the colorbar
    :type minmax: list|tuple
    :param cbar_label: label for the colorbar
    :type cbar_label: str
    :param dividers: black dividers into correlation matrix
    :type dividers: list[int]
    :**kwargs: optional keyword arguments:
        figsize - figure size | tuple
        divider_color - color for dividers | str
    :return: figure instance
    :rtype: `matplotlib.figure.Figure`
    """
    fig = plt.figure(figsize=kwargs.get("figsize", (7, 7)))
    n_maps = corr_mat.shape[0]
    assert maps.shape[0] == n_maps
    gs = fig.add_gridspec(n_maps + 1, n_maps + 1)

    # correlation matrix
    ax_main = fig.add_subplot(gs[:-1, 1:])
    ax_main.set_yticklabels([])
    ax_main.set_xticklabels([])
    ax_main.imshow(corr_mat, cmap=cmap, vmin=minmax[0], vmax=minmax[1])
    divider_color = kwargs.get("divider_color", "k")
    for divider in dividers:
        ax_main.axhline(divider - 0.5, color=divider_color)
        ax_main.axvline(divider - 0.5, color=divider_color)

    # maps as labels
    for lab in range(n_maps):
        ax_ylabel = fig.add_subplot(gs[lab, 0])
        mne.viz.plot_topomap(
            maps[lab, :],
            mne_info,
            show=False,
            contours=10,
            axes=ax_ylabel,
        )
        ax_xlabel = fig.add_subplot(gs[-1, lab + 1])
        mne.viz.plot_topomap(
            maps[lab, :],
            mne_info,
            show=False,
            contours=10,
            axes=ax_xlabel,
        )

    if colorbar:
        cax = fig.add_axes([1.0, 0.5, 0.02, 0.3])
        cbar = mpl.colorbar.ColorbarBase(
            cax,
            cmap=plt.get_cmap(cmap),
            norm=mpl.colors.Normalize(vmin=minmax[0], vmax=minmax[1]),
            orientation="vertical",
        )
        cbar.set_label(cbar_label)

    return fig
