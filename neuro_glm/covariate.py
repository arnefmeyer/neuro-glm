#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Poisson generalized linear model (GLM)
"""


import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from neuro_glm import util


def segment(data, seg_len, shift, zero_padding=True):
    """Rearranges a vector by buffering overlapping segments as rows
    Parameters
    ==========
        data : array-like
            a vector (or flattened view) of the array to be segmented
        seg_len : int
            length of each segment in samples
        shift : int
            the segment shift in samples
        zero_padding : bool, optional
            append zeros to data if data does not contain enough
            samples to fill all segments. If set to False, the
            last segment will be omitted. Defaults to True.
    """
    total_len = data.shape[0]

    if zero_padding is True:
        num_seg = np.int(np.ceil((total_len - seg_len + shift) / shift))
    else:
        num_seg = np.int(np.floor((total_len - seg_len + shift) / shift))

    out = np.zeros([num_seg, seg_len])
    for i in range(num_seg):
        i0 = i * shift
        i1 = i0 + seg_len
        if i1 <= data.shape[0]:
            out[i, :] = data[i0:i1]

        else:
            j1 = data.shape[0] - i0
            out[i, :j1] = data[i0:]

    return out


class DesignMatrix(object):

    def __init__(self, timestamps, dt=None):

        if dt is None:
            dt = np.median(np.diff(timestamps))

        self.timestamps = timestamps
        self.dt = dt

        self.covariates = OrderedDict()
        self.penalties = OrderedDict()

    def add_covariate(self, name, cov, penalty=None):

        assert name not in self.covariates

        self.covariates[name] = cov
        self.penalties[name] = penalty

    @property
    def bin_edges(self):

        return np.append(self.timestamps - self.dt/2., self.timestamps[-1] + self.dt/2)

    def get_penalties_with_indices(self, names=None):

        if names is None:
            names = self.covariates.keys()

        i0 = 0
        for k in names:
            n_dim = self.covariates[k].ndim
            self.penalties[k].w_ind = i0 + np.arange(n_dim)
            i0 += n_dim

        return [self.penalties[k] for k in names]

    @property
    def valid_rows(self):

        V = np.vstack([cov.valid_rows if cov.valid_rows is not None else np.ones((cov.X.shape[0],), dtype=np.bool)
                       for cov in self.covariates.values()])
        return np.sum(V, axis=0) == V.shape[0]

    def as_array(self, names=None, ignore_invalid=False):

        if names is None:
            names = self.covariates.keys()

        if ignore_invalid:
            valid = self.valid_rows
            return np.concatenate([self.covariates[k].X[valid, :] for k in names], axis=1)
        else:
            return np.concatenate([self.covariates[k].X for k in names], axis=1)

    def split_weights(self, w, names=None):

        if names is None:
            names = self.covariates.keys()

        w_cov = OrderedDict()
        i0 = 0
        for k in names:
            n_dim = self.covariates[k].ndim
            w_cov[k] = self.covariates[k].apply_basis(w[i0 + np.arange(n_dim)])
            i0 += n_dim

        return w_cov

    def get_covariate_params(self, names=None):

        if names is None:
            names = self.covariates.keys()

        params = {}
        for k in names:

            if k in self.penalties and hasattr(self.penalties[k], 'w_shape'):
                w_shape = self.penalties[k].w_shape
            else:
                w_shape = len(self.covariates[k].bin_centers)
            params[k] = {'bin_centers': self.covariates[k].bin_centers,
                         'bin_edges': self.covariates[k].bin_edges,
                         'w_shape': w_shape,
                         'penalty': self.penalties[k].__class__.__name__ if k in self.penalties else None}

        return params


class Covariate(object):

    def __init__(self):

        self.X = None
        self.valid_rows = None
        self.basis = None

    # @property
    # def nobs(self):
    #     raise NotImplementedError()

    @property
    def ndim(self):
        raise NotImplementedError()

    def apply_basis(self, w):
        if self.basis is None:
            return w
        else:
            return w @ self.basis


class Pos2D(Covariate):

    def __init__(self, xy, x_range=(-1, 1), y_range=(-1, 1), bin_size=.1):
        super(Pos2D, self).__init__()

        self.x_range = x_range
        self.y_range = y_range
        self.bin_size = bin_size

        self.X, self.valid_rows = self._bin_data(xy)

    @property
    def bin_edges(self):

        nx = int(round((self.x_range[1] - self.x_range[0]) / self.bin_size))
        ny = int(round((self.y_range[1] - self.y_range[0]) / self.bin_size))
        x_edges = np.linspace(self.x_range[0], self.x_range[1], nx + 1)
        y_edges = np.linspace(self.y_range[0], self.y_range[1], ny + 1)

        return x_edges, y_edges

    @property
    def bin_centers(self):

        x_edges, y_edges = self.bin_edges
        x_centers = self.bin_size/2 + x_edges[:-1]
        y_centers = self.bin_size / 2 + y_edges[:-1]

        return x_centers, y_centers

    @property
    def ndim(self):

        x_c, y_c = self.bin_centers

        return len(x_c) * len(y_c)

    def _bin_data(self, xy):

        x_centers, y_centers = self.bin_centers
        nx, ny = len(x_centers), len(y_centers)

        binned = np.zeros((xy.shape[0], nx*ny))
        valid_rows = np.ones((xy.shape[0],), dtype=np.bool)
        for i, xy_i in enumerate(xy):
            if np.sum(np.isnan(xy_i)) == 0:
                ix = np.argmin(np.abs(x_centers - xy_i[0]))
                iy = np.argmin(np.abs(y_centers - xy_i[1]))
                index = iy * nx + ix
                binned[i, index] = 1
            else:
                valid_rows[i] = False

        return binned, valid_rows

    def plot(self, w, ax=None):

        if ax is None:
            fig, ax = plt.subplots()

        x_centers, y_centers = self.bin_centers
        W = np.reshape(w, (len(x_centers), len(y_centers)))
        w_max = np.max(np.abs(w))
        im = ax.imshow(W, vmin=-w_max, vmax=w_max, interpolation='nearest', cmap='RdBu_r',
                       origin='lower', extent=(self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks(self.x_range)
        ax.set_yticks(self.y_range)
        util.simple_xy_axes(ax)
        util.set_font_axes(ax)
        util.adjust_axes(ax)

        cb = plt.colorbar(im, ax=ax, location='bottom', orientation='horizontal')
        util.set_font_axes(cb.ax)

        return ax


class Pos1D(Covariate):

    def __init__(self, x, x_range=(-1, 1), bin_size=.1):
        super(Pos1D, self).__init__()

        self.x_range = x_range
        self.bin_size = bin_size

        self.X, self.valid_rows = self._bin_data(x)

    @property
    def bin_edges(self):

        nx = int(round((self.x_range[1] - self.x_range[0]) / self.bin_size))
        x_edges = np.linspace(self.x_range[0], self.x_range[1], nx + 1)

        return x_edges

    @property
    def bin_centers(self):

        return self.bin_size/2 + self.bin_edges[:-1]

    @property
    def ndim(self):

        return len(self.bin_centers)

    def _bin_data(self, x):

        x_centers = self.bin_centers
        nx = len(x_centers)

        x = np.copy(x)
        x[x < self.x_range[0]] = np.NaN
        x[x > self.x_range[-1]] = np.NaN

        binned = np.zeros((x.shape[0], nx))
        col_ind = np.floor(x / self.bin_size).astype(np.int)
        valid_rows = ~np.isnan(x)
        binned[valid_rows, col_ind[valid_rows]] = 1

        return binned, valid_rows

    def plot(self, w, ax=None):

        if ax is None:
            fig, ax = plt.subplots()

        x_centers = self.bin_centers

        w_max = np.max(np.abs(w))
        ax.plot(x_centers, w, '-')
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Weight')
        ax.set_xticks(self.x_range)
        ax.set_yticks([-w_max, 0, w_max])

        util.simple_xy_axes(ax)
        util.set_font_axes(ax)
        util.adjust_axes(ax)

        return ax


class HeadDir2D(Covariate):

    def __init__(self, hd, bin_size=15):
        super(HeadDir2D, self).__init__()

        self.bin_size = bin_size
        self.X, self.valid_rows = self._bin_data(hd)

    @property
    def bin_edges(self):

        n_bins = 360 / self.bin_size
        edges = np.arange(n_bins + 1) * self.bin_size

        return edges

    @property
    def bin_centers(self):

        return self.bin_edges[:-1] + self.bin_size/2.

    @property
    def ndim(self):

        return len(self.bin_centers)

    def _bin_data(self, phi):

        edges = self.bin_edges
        binned = np.zeros((phi.shape[0], len(edges)-1))
        col_ind = np.floor(phi / self.bin_size).astype(np.int)
        valid_rows = ~np.isnan(phi)
        binned[valid_rows, col_ind[valid_rows]] = 1

        return binned, valid_rows

    def plot(self, w, ax=None):

        centers = self.bin_centers

        wh = np.append(w, w[0])
        ch = np.append(centers, centers[0])

        ax.plot(np.deg2rad(ch), wh, color='tab:blue')
        ax.grid(True)
        ax.set_theta_zero_location('E')

        util.set_font_axes(ax)
        util.adjust_axes(ax)


class Speed(Covariate):

    def __init__(self, spd, s_range=(0, 30), bin_size=5):
        super(Speed, self).__init__()

        self.s_range = s_range
        self.bin_size = bin_size

        self.X, self.valid_rows = self._bin_data(spd)

    @property
    def bin_edges(self):

        n_bins = (self.s_range[1] - self.s_range[0]) / self.bin_size
        edges = self.s_range[0] + np.arange(n_bins + 1) * self.bin_size

        return edges

    @property
    def bin_centers(self):

        return self.bin_edges[:-1] + self.bin_size/2.

    @property
    def ndim(self):

        return len(self.bin_centers)

    def _bin_data(self, spd):

        spd = np.copy(spd)
        centers = self.bin_centers

        spd[spd < self.s_range[0]] = np.NaN
        spd[spd > self.s_range[1]] = np.NaN

        binned = np.zeros((spd.shape[0], len(centers)))
        col_ind = np.floor(spd / self.bin_size).astype(np.int)
        valid_rows = ~np.isnan(spd)
        binned[valid_rows, col_ind[valid_rows]] = 1

        return binned, valid_rows

    def plot(self, w, ax=None):

        centers = self.bin_centers
        y_max = 1.1 * max(np.abs(w))

        ax.plot(centers, w, '-', color='tab:blue')
        ax.set_xlabel('Body speed (cm/s)')
        ax.set_ylabel('Weight', labelpad=0)
        ax.set_xticks(self.s_range)
        ax.set_xlim(self.s_range[0], self.s_range[1])
        ax.set_ylim(-y_max, y_max)
        ax.set_yticks([-y_max, 0, y_max])

        util.set_font_axes(ax)
        util.simple_xy_axes(ax)
        util.adjust_axes(ax)


class Velocity(Covariate):

    def __init__(self, velocity, v_range=(-30, 30), bin_size=5):
        super(Velocity, self).__init__()

        self.v_range = v_range
        self.bin_size = bin_size

        self.X, self.valid_rows = self._bin_data(velocity)

    @property
    def bin_edges(self):

        n_bins = (self.v_range[1] - self.v_range[0]) / self.bin_size
        edges = self.v_range[0] + np.arange(n_bins + 1) * self.bin_size

        return edges

    @property
    def bin_centers(self):

        return self.bin_edges[:-1] + self.bin_size/2.

    @property
    def ndim(self):

        return len(self.bin_centers)

    def _bin_data(self, velocity):

        vel = np.copy(velocity)
        centers = self.bin_centers

        vel[vel < self.v_range[0]] = np.NaN
        vel[vel > self.v_range[1]] = np.NaN

        binned = np.zeros((vel.shape[0], len(centers)))
        col_ind = np.floor((vel - self.v_range[0]) / self.bin_size).astype(np.int)
        valid_rows = ~np.isnan(vel)
        binned[valid_rows, col_ind[valid_rows]] = 1

        return binned, valid_rows

    def plot(self, w, ax=None):

        centers = self.bin_centers
        y_max = 1.1 * max(np.abs(w))

        ax.plot(centers, w, '-', color='tab:blue')
        ax.set_xlabel('Body velocity (cm/s)')
        ax.set_ylabel('Weight', labelpad=0)
        ax.set_xticks(self.v_range)
        ax.set_xlim(self.v_range[0], self.v_range[1])
        ax.set_ylim(-y_max, y_max)
        ax.set_yticks([-y_max, 0, y_max])

        util.set_font_axes(ax)
        util.simple_xy_axes(ax)
        util.adjust_axes(ax)


class ThetaPhase(Covariate):

    def __init__(self, theta_phase, bin_size=15):
        super(ThetaPhase, self).__init__()

        self.bin_size = bin_size

        self.X, self.valid_rows = self._bin_data(theta_phase)

    @property
    def bin_edges(self):

        n_bins = 360 / self.bin_size
        edges = -180 + np.arange(n_bins + 1) * self.bin_size

        return edges

    @property
    def bin_centers(self):

        return self.bin_edges[:-1] + self.bin_size/2.

    @property
    def ndim(self):

        return len(self.bin_centers)

    def _bin_data(self, phi):

        edges = self.bin_edges + 180
        phi = phi + 180

        binned = np.zeros((phi.shape[0], len(edges)-1))
        col_ind = np.floor(phi / self.bin_size).astype(np.int)
        valid_rows = ~np.isnan(phi)
        binned[valid_rows, col_ind[valid_rows]] = 1

        return binned, valid_rows

    def plot(self, w, ax=None):

        centers = self.bin_centers
        w_max = 1.1*np.max(np.abs(w))

        ax.plot(centers, w, color='tab:blue')
        ax.set_xlabel('Theta phase (deg)')
        ax.set_ylabel('Weight')
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, 0, 180])
        ax.set_ylim(-w_max, w_max)

        util.simple_xy_axes(ax)
        util.set_font_axes(ax)
        util.adjust_axes(ax)


class HeadPitch(Covariate):

    def __init__(self, pitch, p_range=(-90, 60), bin_size=15):
        super(HeadPitch, self).__init__()

        self.p_range = p_range
        self.bin_size = bin_size

        self.X, self.valid_rows = self._bin_data(pitch)

    @property
    def bin_edges(self):

        n_bins = (self.p_range[1] - self.p_range[0]) / self.bin_size
        edges = self.p_range[0] + np.arange(n_bins + 1) * self.bin_size

        return edges

    @property
    def bin_centers(self):

        return self.bin_edges[:-1] + self.bin_size/2.

    @property
    def ndim(self):

        return len(self.bin_centers)

    def _bin_data(self, pitch):

        centers = self.bin_centers

        pitch = np.copy(pitch)
        pitch[pitch < self.p_range[0]] = np.NaN
        pitch[pitch > self.p_range[1]] = np.NaN
        pitch -= self.p_range[0]

        binned = np.zeros((pitch.shape[0], len(centers)))
        col_ind = np.floor(pitch / self.bin_size).astype(np.int)
        valid_rows = ~np.isnan(pitch)
        binned[valid_rows, col_ind[valid_rows]] = 1

        return binned, valid_rows

    def plot(self, w, ax=None):

        centers = self.bin_centers
        y_max = 1.1 * max(np.abs(w))

        ax.plot(centers, w, '-', color='tab:blue')
        ax.set_xlabel('Head pitch (deg)')
        ax.set_ylabel('Weight', labelpad=0)
        ax.set_xticks(self.p_range)
        ax.set_xlim(self.p_range[0], self.p_range[1])
        ax.set_ylim(-y_max, y_max)
        ax.set_yticks([-y_max, y_max])
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))

        util.set_font_axes(ax)
        util.simple_xy_axes(ax)
        util.adjust_axes(ax)


class SpikeHistory(Covariate):

    def __init__(self, counts, dt=.025, max_lag=1., n_spline_bases=10):
        super(SpikeHistory, self).__init__()

        self.dt = dt
        self.max_lag = max_lag
        self.n_spline_bases = n_spline_bases

        if n_spline_bases > 0:
            self.basis = self.create_raise_cosine_basis(n_bases=n_spline_bases)

        self.X = self._segment_data(counts)
        self.valid_rows = np.ones((self.X.shape[0],), dtype=np.bool)

    @property
    def bin_edges(self):

        # edges = np.arange(-self.max_lag-.5, 0) * self.dt
        centers = self.bin_centers
        edges = np.append(centers-self.dt/2, centers[-1]+self.dt/2)

        return edges

    @property
    def bin_centers(self):

        # if self.basis is not None:
        #     ind = np.argmax(self.basis, axis=1)
        #     centers = self.dt/2 + ind*self.dt
        # else:
        #     centers = self.dt/2 + np.arange(0, self.max_lag, self.dt)

        centers = self.dt/2 + np.arange(0, self.max_lag, self.dt)

        return centers

    @property
    def ndim(self):

        if self.basis is not None:
            return self.n_spline_bases
        else:
            return len(self.bin_centers)

    def create_raise_cosine_basis(self, b=0.1, n_bases=10):
        # https://egocodedinsol.github.io/raised_cosine_basis_functions/

        n_lag = int(round(self.max_lag / self.dt + .5))
        t = np.linspace(0, 1, n_lag)
        nt = np.log(t + b + np.finfo(float).eps)
        c_start = nt[0]
        c_end = nt[-1]
        d_b = (c_end - c_start) / (n_bases - 1.)
        c = np.arange(c_start, c_end+1, d_b)

        basis = np.zeros((n_bases, len(t)))
        for k in range(n_bases):
            # tmp = np.min(np.pi, (nt - c[k])*np.pi / d_b)
            tmp = (nt - c[k]) * np.pi / d_b
            tmp[tmp > np.pi] = np.pi
            tmp[tmp < -np.pi] = -np.pi
            basis[k, :] = (np.cos(tmp) + 1) / 2

        return basis

    def _segment_data(self, counts):

        n_lag = int(round(self.max_lag / self.dt + .5))
        S = segment(np.append(np.zeros((n_lag,)), counts), n_lag, 1)
        S = S[:len(counts), :]

        # flip such that most recent spiking is to the left
        S = S[:, ::-1]

        if self.basis is not None:
            # apply raise cosine basis functions
            S = S @ self.basis.T

        return S

    def plot(self, w, ax=None):

        centers = (self.bin_centers * 1000).astype(np.int)
        y_max = 1.1 * max(np.abs(w))

        ax.plot(centers, w, '-', color='tab:blue')
        ax.set_xlabel('Time before spike (ms)')
        ax.set_ylabel('Weight', labelpad=0)
        ax.set_xticks([0, int(self.max_lag*1000)])
        ax.set_xlim([0, int(self.max_lag*1000)])
        ax.set_ylim(-y_max, y_max)
        ax.set_yticks([-y_max, y_max])

        util.set_font_axes(ax)
        util.simple_xy_axes(ax)
        util.adjust_axes(ax)


class EgocentricBoundary(Covariate):

    def __init__(self, intersections, d_phi=30, r_range=(0, 30), d_r=3, method='all'):  # r_edges=[0, 1, 2, 4, 8, 16, 32, 64]):
        super(EgocentricBoundary, self).__init__()

        self.d_phi = d_phi
        self.r_range = r_range
        self.d_r = d_r
        self.method = method
        # self.r_edges = r_edges

        self.X, self.valid_rows = self._bin_data(intersections)

    @property
    def bin_edges(self):

        n_r = int(round((self.r_range[1] - self.r_range[0]) / self.d_r))
        r_edges = self.r_range[0] + np.arange(n_r + 1) * self.d_r
        phi_edges = -self.d_phi/2. + np.arange(0, 361, self.d_phi)

        return phi_edges, r_edges

    @property
    def bin_centers(self):

        phi_edges, r_edges  = self.bin_edges
        return phi_edges[:-1] + self.d_phi/2., r_edges[:-1] + self.d_r/2

    @property
    def ndim(self):

        phi_centers, r_centers = self.bin_centers
        return len(r_centers) * len(phi_centers)

    def _bin_data(self, intersections):

        assert self.method in ['all', 'closest']

        if self.method == 'closest':
            for i in range(intersections.shape[0]):
                minval = np.nanmin(intersections[i])
                v = intersections[i] == minval
                intersections[i, ~v] = np.NaN

        r_range = self.r_range
        d_r = self.d_r
        n_r = int(round((r_range[-1] - r_range[0]) / d_r))

        intersections = np.copy(intersections)
        intersections[intersections > r_range[-1]] = np.NaN

        n_phi = int(360 / self.d_phi)

        binned = np.zeros((intersections.shape[0], n_r, n_phi))
        for i in range(n_phi):
            col_ind = np.floor(intersections[:, i] / d_r).astype(np.int)
            valid_rows = ~np.isnan(intersections[:, i])
            binned[valid_rows, col_ind[valid_rows], i] = 1

        binned = np.reshape(binned, (binned.shape[0], -1))
        # n_valid = n_phi if self.method == 'all' else 1
        # valid_rows = np.sum(binned, axis=1) == n_valid
        valid_rows = np.sum(np.isnan(intersections), axis=1) < intersections.shape[1]

        return binned, valid_rows

    def plot(self, w, ax=None):

        # https://stackoverflow.com/questions/53081557/3d-polar-plot-on-matplotlib
        # https://towardsdatascience.com/polar-heatmaps-in-python-with-matplotlib-d2a09610bc55
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Wedge.html

        from matplotlib.patches import Wedge
        from matplotlib.collections import PatchCollection

        phi_centers, r_centers = self.bin_centers
        W = np.reshape(w, (len(r_centers), len(phi_centers)))
        w_max = np.max(np.abs(w))

        # create wedges
        patches = []
        colors = []
        norm = plt.Normalize(-w_max, w_max)
        cm = plt.cm.viridis

        for i, r in enumerate(r_centers):
            r2 = r + self.d_r / 2.
            for j, p in enumerate(phi_centers):
                p1 = p - self.d_phi / 2.
                p2 = p + self.d_phi / 2.

                wedge = Wedge(0, r2, p1, p2, width=self.d_r)
                patches.append(wedge)
                colors.append(cm(norm(W[i, j])))

        collection = PatchCollection(patches, linewidth=0, facecolors=colors)
        im = ax.add_collection(collection)

        # P, R = np.meshgrid(np.append(phi_centers, 0), r_centers)
        # WW = np.hstack((W, np.atleast_2d(W[:, -1]).T))
        # ax.pcolormesh(np.deg2rad(P), R, WW, vmin=-w_max, vmax=w_max)
        # ax.set_theta_zero_location('N')

        # np.savez('/home/arne/Desktop/ebc.npz', W=W, phi=phi_centers, r=r_centers)

        ax.axis('scaled')
        ax.axis('off')
        r_max = self.r_range[-1]
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.text(0, 1.15*r_max, 'Front', ha='center', va='center')
        ax.text(0, -1.15*r_max, 'Back', ha='center', va='center')
        ax.text(-1.25*r_max, 0, 'Left', ha='center', va='center')
        ax.text(1.25*r_max, 0, 'Right', ha='center', va='center')

        util.set_font_axes(ax, size_text=7)
        util.adjust_axes(ax)

        # cb = plt.colorbar(im, ax=ax, location='bottom', orientation='horizontal')
        # util.set_font_axes(cb.ax)


class ThetaAmplitude(Covariate):

    def __init__(self, lfp):
        super(ThetaAmplitude, self).__init__()

        self.X = lfp
        self.valid_rows = np.ones((lfp.shape[0],), dtype=np.bool)

    @property
    def bin_edges(self):

        return .5+np.arange(self.X.shape[1]+1)

    @property
    def bin_centers(self):

        return self.bin_edges[:-1] + .5

    @property
    def ndim(self):

        return len(self.bin_centers)

    def plot(self, w, ax=None):

        centers = self.bin_centers
        y_max = 1.1 * max(np.abs(w))

        ax.plot(centers, w, '-', color='tab:blue')
        ax.set_xlabel('Probe channel')
        ax.set_ylabel('Weight', labelpad=0)
        # ax.set_xticks(centers[0], centers[-1])
        # ax.set_xlim(self.p_range[0], self.p_range[1])
        ax.set_ylim(-y_max, y_max)
        ax.set_yticks([-y_max, y_max])
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))

        util.set_font_axes(ax)
        util.simple_xy_axes(ax)
        util.adjust_axes(ax)
