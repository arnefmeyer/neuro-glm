#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Poisson generalized linear model (GLM)
"""

import numpy as np


class Penalty(object):

    def __init__(self, name='param', value=0., grid=2**np.linspace(-3, 10, 6), w_ind=None):

        self.name = name
        self.value = value
        self.grid = grid
        self.w_ind = w_ind

    def apply(self, w):
        raise NotImplementedError()


class L2Penalty(Penalty):

    def __init__(self, *args, **kwargs):
        super(L2Penalty, self).__init__(*args, **kwargs)

    def apply(self, w):

        beta = self.value

        n_dim = w.shape[0]

        f = beta * 0.5 * w @ w
        g = beta * w
        h = beta * np.eye(n_dim)

        return f, g, h


class L1Penalty(Penalty):
    """Smooth approximation to L1 penalty (i.e. with well-defined derivative)

        see eqs. 3-5 in https://people.csail.mit.edu/romer/papers/SchFunRos_ECML07.pdf
    """

    def __init__(self, alpha=100, **kwargs):
        super(L1Penalty, self).__init__(**kwargs)
        self.alpha = alpha

    def smoothl1_f(self, w):
        return 1 / self.alpha * np.sum((np.log(1 + np.exp(-self.alpha * w)) + np.log(1 + np.exp(self.alpha * w))))

    def smoothl1_grad(self, w):
        return 1 / (1 + np.exp(-self.alpha * w)) - 1 / (1 + np.exp(self.alpha * w))

    def smoothl1_hess(self, w):
        return 2*self.alpha*np.exp(self.alpha*w) / (1 + np.exp(self.alpha*w))**2

    def apply(self, w):

        beta = self.value

        f = beta * self.smoothl1_f(w)
        g = beta * self.smoothl1_grad(w)
        h = beta * self.smoothl1_hess(w)

        return f, g, h


class RoughnessPenalty1D(Penalty):

    def __init__(self, *args, w_shape=10, **kwargs):
        super(RoughnessPenalty1D, self).__init__(*args, **kwargs)
        self.w_shape = w_shape

        n_dim = w_shape
        D = np.diag(np.ones((n_dim,))) + np.diag(-1 * np.ones((n_dim,)), 1)[:-1, :-1]
        self.DD = D.T @ D

    def apply(self, w):

        beta = self.value

        DD = self.DD
        f = beta * 0.5 * w @ DD @ w
        g = beta * DD @ w
        h = beta * DD

        return f, g, h


class RoughnessPenalty2D(Penalty):

    def __init__(self, w_shape=(8, 6), **kwargs):
        super(RoughnessPenalty2D, self).__init__(**kwargs)
        self.w_shape = w_shape

        n_rows, n_cols = self.w_shape
        n_dim = n_cols * n_rows
        DD = np.zeros((n_dim, n_dim))
        for i in range(n_rows):
            for j in range(n_cols):
                i0 = i * n_cols + j
                i1 = (i - 1) * n_cols + j
                i2 = (i + 1) * n_cols + j
                j1 = i * n_cols + j - 1
                j2 = i * n_cols + j + 1
                for k in [i1, i2, j1, j2]:
                    if 0 <= k < n_dim:
                        DD[i0, k] -= 1
                        DD[i0, i0] += 1
        self.DD = DD

    def apply(self, w):

        beta = self.value

        DD = self.DD
        f = beta * 0.5 * w @ DD @ w
        g = beta * DD @ w
        h = beta * DD

        return f, g, h


class RoughnessPenalty1DCircular(Penalty):

    def __init__(self, w_shape=10, **kwargs):
        super(RoughnessPenalty1DCircular, self).__init__(**kwargs)
        self.w_shape = w_shape

        n_dim = w_shape
        D = np.diag(np.ones((n_dim,))) + np.diag(-1 * np.ones((n_dim,)), 1)[:-1, :-1]
        DD = D.T @ D

        # correct the smoothing across first and last bin
        DD[0, :] = np.roll(DD[1, :], -1, axis=0)
        DD[-1, :] = np.roll(DD[-2, :], 1, axis=0)

        self.DD = DD

    def apply(self, w):

        beta = self.value

        DD = self.DD
        f = beta * 0.5 * w @ DD @ w
        g = beta * DD @ w
        h = beta * DD

        return f, g, h
