#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    some helper functions used in multiple scripts
"""

import numpy as np
from scipy.io import loadmat
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# import matplotlib.collections as mcoll


# ---------------------------------------------------------------------
# math functions
# ---------------------------------------------------------------------

def length(vec):
    return np.sqrt(np.sum(np.asarray(vec) ** 2))


def normalized(vec):

    return np.asarray(vec) / length(vec)


def wilcoxon(x, y,
             exact=True,
             verbose=False):
    """try to using R's exact wilcoxon test via rpy2 if available"""

    from scipy import stats

    if exact:

        try:
            import rpy2.robjects as robjects
            from rpy2.robjects.packages import importr

            rstats = importr('stats')
            paired = len(x) == len(y)
            res = rstats.wilcox_test(robjects.FloatVector(x),
                                     robjects.FloatVector(y),
                                     exact=True,
                                     paired=paired)
            dd = {k[0]: k[1] for k in res.items()}

            if not paired:
                h = {u[0]: u[1] for u in dd['statistic'].items()}['W']
            else:
                h = dd['statistic'][0]
            pval = {u[0]: u[1] for u in dd['p.value'].items()}[None]

            if verbose:
                print("Used R's wilcox_test. pvalue=:", pval)

        except ImportError:
            print("Could not import rpy2. Using scipy's test instead.")
            h, pval = stats.wilcoxon(x, y)

    else:
        h, pval = stats.wilcoxon(x, y)

    return h, pval


def set_font_axes(ax,
                  add_size=0,
                  size_ticks=6,
                  size_labels=8,
                  size_text=8,
                  size_title=8,
                  family='Arial'):
    # set font family and size for given axes

    # title
    ax.title.set_fontname(family)
    if size_title is not None:
        ax.title.set_fontsize(size_title + add_size)

    if size_ticks is not None:
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=size_ticks + add_size)

    # labels
    ax.xaxis.label.set_fontname(family)
    ax.yaxis.label.set_fontname(family)
    if hasattr(ax, 'zaxis'):
        ax.zaxis.label.set_fontname(family)

    if size_labels is not None:

        ax.xaxis.label.set_fontsize(size_labels + add_size)
        ax.yaxis.label.set_fontsize(size_labels + add_size)
        if hasattr(ax, 'zaxis'):
            ax.zaxis.label.set_fontsize(size_labels + add_size)

    for at in ax.texts:
        at.set_fontname(family)
        if size_text is not None:
            at.set_fontsize(size_text + add_size)


def adjust_axes(ax,
                tick_length=2,
                tick_direction='out',
                spine_width=0.5,
                pad=2):
    # set axes ticks, label padding, and spine width

    if tick_length is not None:
        ax.tick_params(axis='both',
                       which='major',
                       length=tick_length)

    ax.tick_params(axis='both',
                   which='both',
                   direction=tick_direction)

    if pad is not None:
        ax.tick_params(axis='both',
                       which='both',
                       pad=pad)

    for s in ax.spines:
        spine = ax.spines[s]
        if spine.get_visible():
            spine.set_linewidth(spine_width)


def simple_xy_axes(ax):
    """Remove top and right spines/ticks from matplotlib axes"""

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def simple_twinx_axes(ax):
    """Remove top and right spines/ticks from matplotlib axes"""

    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')


def shift_ticklabels(ax, shift, which='x'):
    """shift ticklabels for given axis"""

    import types

    if which == 'x':
        labels = ax.xaxis.get_majorticklabels()
    elif which == 'y':
        labels = ax.yaxis.get_majorticklabels()

    if isinstance(shift, (int, float)):
        shift = len(labels) * [shift]

    for i, (label, dx) in enumerate(zip(labels, shift)):

        label.dx = dx
        label.set_x = types.MethodType(lambda self, x:
                                       plt.Text.set_x(self, x+self.dx),
                                       label)


# ---------------------------------------------------------------------
# signal processing-related functions
# ---------------------------------------------------------------------

def filter_data(data, fs, f_lower=300., f_upper=6000., order=2,
                method='filtfilt', column_wise=False, filt_type='bandpass'):
    """simple bandbass filtering of electrode signals"""

#    if f_lower > 0:
    if filt_type == 'bandpass':
        Wn = (f_lower / fs * 2, f_upper / fs * 2)
        b, a = signal.butter(order, Wn, btype='bandpass', analog=False,
                             output='ba')
    elif filt_type == 'lowpass':
        Wn = f_upper / fs * 2
        b, a = signal.butter(order, Wn, btype='lowpass', analog=False,
                             output='ba')
    elif filt_type == 'highpass':
        Wn = f_lower / fs * 2
        b, a = signal.butter(order, Wn, btype='highpass', analog=False,
                             output='ba')
    else:
        raise ValueError('Unknown filter type: %s' % filt_type)

    if method == 'filtfilt':
        filt_fun = signal.filtfilt
    elif method == 'lfilter':
        filt_fun = signal.lfilter
    else:
        raise ValueError('Invalid filtering method: ' + method)

    if column_wise:
        data_ = np.zeros_like(data)
        for i in range(data.shape[1]):
            data_[:, i] = filt_fun(b, a, data[:, i])
    else:
        data = filt_fun(b, a, data, axis=0)

    return data


def corrlag(x, y, maxlag=1000, normalize=True, center=True):
    """correlation function with max. time lag"""

    assert x.shape[0] == y.shape[0]

    N = x.shape[0]
    i1 = N-1 - maxlag
    i2 = N-1 + maxlag+1

    if center:
        cxy = signal.correlate(x - x.mean(), y - y.mean(), 'full')
    else:
        cxy = signal.correlate(x, y, 'full')

    if normalize:
        cc = np.diag(np.corrcoef(x, y), 1)
        cxy = cxy / cxy[N-1] * cc

    return cxy[i1:i2], np.arange(-maxlag, maxlag+1)
