#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Arne F. Meyer <arne.f.meyer@gmail.com>
# License: GPLv3

"""
    Poisson generalized linear model (GLM)
"""


import numpy as np
from scipy import optimize
from sklearn.base import BaseEstimator
from scipy.special import factorial
from collections import OrderedDict

# -----------------------------------------------------------------------------
# GLM fitting codes
# -----------------------------------------------------------------------------

# try:
#     import torch
#     from torch import nn
#     from torch import optim
#     from torch.utils.data import TensorDataset
#     from torch.utils.data import DataLoader
#
#     device = torch.device(
#         "cuda") if torch.cuda.is_available() else torch.device("cpu")
#     print("using device:", device)
#
#     class PytorchGLM(nn.Module):
#
#         def __init__(self, n_dim):
#             super().__init__()
#             self.lin = nn.Linear(n_dim, 1)
#
#         def forward(self, xb):
#             return self.lin(xb)
#
#
#     def fit_glm(x, y, optimizer='LBFGS'):
#
#         x_train = torch.tensor(x.astype(np.float32)).to(device)
#         y_train = torch.tensor(y.reshape(-1, 1).astype(np.float32)).to(device)
#
#         model = PytorchGLM(n_dim=x.shape[1])
#         model.to(device)
#
#         loss_func = torch.nn.functional.poisson_nll_loss
#
#         if optimizer.upper() == 'LBFGS':
#             opt = optim.LBFGS(model.parameters(), lr=1, max_iter=50)
#         elif optimizer.upper() == 'ADAM':
#             opt = optim.Adam(model.parameters(), lr=.001)
#
#         if type(opt) == optim.Adam:
#             epochs = 20000
#         else:
#             epochs = 10
#
#         t0 = time.time()
#         for epoch in range(epochs):
#
#             xb = x_train
#             yb = y_train
#
#             def closure():
#                 opt.zero_grad()
#                 pred = model(xb)
#                 loss = loss_func(pred, yb, log_input=True)
#                 loss.backward()
#                 return loss
#
#             opt.step(closure)
#
#             if type(opt) == optim.Adam:
#                 if epoch % 1000 == 0:
#                     print("loss:", loss_func(model(xb), yb, log_input=True))
#             else:
#                 print("loss:", loss_func(model(xb), yb, log_input=True))
#
#         print("fitting time:", time.time() - t0)
#
#         w = model.lin.weight.detach().cpu().numpy().ravel()
#         w = np.reshape(w, k_true.shape)
#
#         return w
#
# except ModuleNotFoundError:
#     pass


# def create_derivative_penality_1d(w, beta):
#
#     n_dim = w.shape[0]
#     D = np.diag(np.ones((n_dim,))) + np.diag(-1 * np.ones((n_dim,)), 1)[:-1, :-1]
#     DD1 = np.dot(D1.T, D1)
#     J = beta * 0.5 * np.dot(w, np.dot(DD1, w))
#     J_g = beta * np.dot(DD1, w)
#     J_h = beta * DD1
#
#     return J, J_g, J_h


# def create_derivative_penality_1d_circ(w, beta):
#
#     n_dim = w.shape[0]
#     D = np.diag(np.ones((n_dim,))) + np.diag(-1 * np.ones((n_dim,)), 1)[:-1, :-1]
#     DD1 = np.dot(D1.T, D1)
#
#     # correct the smoothing across first and last bin
#     # DD1(1,:) = circshift(DD1(2,:),[0 -1]);
#     # DD1(end,:) = circshift(DD1(end-1,:),[0 1]);
#     DD1[0, :] = np.roll(DD1[1, :], -1, axis=1)
#     DD1[-1, :] = np.roll(DD1[-2, :], 1, axis=1)
#
#     J = beta * 0.5 * np.dot(w, np.dot(DD1, w))
#     J_g = beta * np.dot(DD1, w)
#     J_h = beta * DD1
#
#     return J, J_g, J_h


def fit_poisson_glm(params, x, y, penalties=[]):

    # Compute GLM filter output and condititional intensity
    z = x @ params
    lamda = np.exp(z)  # note: lambda is a python keyword

    # Compute negative log-likelihood
    neg_llh = -y @ z + np.nansum(lamda + np.log(factorial(y)))

    # gradient of neg. LLH
    # grad = -np.dot(y, x) + np.dot(x.T, lamda)
    grad = x.T @ (lamda - y)

    # # hessian
    # rx = x.T * y
    # hessian = rx @ x

    # add penalty terms
    w = params[1:]
    for p in penalties:

        p_f, p_grad, p_hess = p.apply(w[p.w_ind])
        neg_llh += p_f
        grad[1+p.w_ind] += p_grad

    return neg_llh, grad


def fit_poisson_glm_hessian(params, x, y, penalties=[]):

    # hessian
    rx = x.T * y
    hessian = rx @ x

    # add penalty terms
    w = params[1:]
    for p in penalties:

        p_f, p_grad, p_hess = p.apply(w[p.w_ind])
        hessian[1+p.w_ind][:, 1+p.w_ind] += p_hess

    return hessian


def compute_log_likelihood(estimator, x, y, return_all=False):

    mean_spike_cnt = np.mean(y)
    lamda = np.exp(np.dot(x, estimator.coef_) + estimator.intercept_)
    n = np.sum(y)
    fact_y = np.log(factorial(y))

    llh_estimator = -np.nansum(lamda - y * np.log(lamda) + fact_y) / n
    llh_mean_rate = -np.nansum(mean_spike_cnt - y * np.log(mean_spike_cnt) + fact_y) / n

    # log likelihood increase; log2 to convert from nats to bits
    llh_increase = np.log(2) * (llh_estimator - llh_mean_rate)

    if return_all:
        return llh_estimator, llh_mean_rate, llh_increase
    else:
        return llh_estimator


class PoissonGLM(BaseEstimator):

    def __init__(self, penalties=[], max_iter=100, tol=1e-4, verbose=False, **kwargs):

        super(PoissonGLM, self).__init__()

        # create penalties from kwargs; slightly hacky but this makes sure it's consistent
        # with sklearn's cross-validation/model selection classes
        if len(kwargs) > 0:

            if len(penalties) == 0:
                # parse from "get_params" data
                penalties = []
                param_names = [k for k in kwargs if
                               not k.endswith('_wind') and
                               not k.endswith('_class') and
                               not k.endswith('_wshape')]
                for param_name in param_names:
                    cls = kwargs[param_name + '_class']
                    w_ind = kwargs[param_name + '_wind']
                    param_value = kwargs[param_name]
                    if param_name + '_wshape' in kwargs:
                        w_shape = kwargs[param_name + '_wshape']
                        p = cls(name=param_name, value=param_value, w_shape=w_shape, w_ind=w_ind)
                    else:
                        p = cls(name=param_name, value=param_value, w_ind=w_ind)

                    penalties.append(p)

            else:
                # kwargs contains hyperparameters that should be fixed to speed up grid search
                for p in penalties:
                    if p.name in kwargs:
                        p.value = kwargs[p.name]

        self.penalties = penalties
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        self.coef_ = None
        self.intercept_ = None

    def get_params(self, deep=True):
        dd = {'max_iter': self.max_iter,
              'tol': self.tol,
              'verbose': self.verbose}
        for p in self.penalties:
            dd[p.name] = p.value
            dd[p.name + '_wind'] = p.w_ind
            dd[p.name + '_class'] = p.__class__
            if hasattr(p, 'w_shape'):
                dd[p.name + '_wshape'] = p.w_shape
        return dd

    def set_params(self, **kwargs):

        for p in self.penalties:
            if p.name in kwargs:
                p.value = kwargs[p.name]

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        return self

    def get_param_grid(self):

        return {p.name: p.grid for p in self.penalties}

    def get_hyperparams(self):

        return {p.name: p.value for p in self.penalties}

    def fit(self, x, y):

        n_obs, n_features = x.shape
        params = np.zeros((n_features + 1,), dtype=x.dtype)
        params[0] = np.nanmean(y)
        params[1:] = 1e-3 * np.random.randn(n_features)

        # add bias term to covariate matrix
        xx = np.hstack((np.ones((n_obs, 1)), x))

        res = optimize.minimize(fit_poisson_glm, params,
                                method="L-BFGS-B",
                                # method='Newton-CG',
                                jac=True,
                                hess=None,
                                options={"maxiter": self.max_iter,
                                         "iprint": (self.verbose > 0) - 1,
                                         "gtol": self.tol,
                                         "ftol": 1e3 * np.finfo(float).eps},
                                args=(xx, y, self.penalties))

        opt_params = res['x']

        self.coef_ = opt_params[1:]
        self.intercept_ = opt_params[0]

    def predict(self, x):

        assert self.coef_ is not None

        z = np.dot(x, self.coef_) + self.intercept_  # filter output
        rate = np.exp(z)  # conditional intensity

        return rate

    def score(self, x, y, return_all=False):

        return compute_log_likelihood(self, x, y, return_all=return_all)


def create_test_dataset(n=1000):

    # "place cell" with one localized bump
    n1 = 15
    w1 = gaussian(n1, .08*n1)

    ind = np.random.randint(0, n1, size=n)
    x1 = np.zeros((n, n1))
    x1[np.arange(n), ind] = 1

    # head direction cell
    n2 = 15
    w2 = gaussian(n2, .1*n2)
    w2 = np.roll(w2, int(round(n2/2)))  # move to the edge to test circular penalty
    # w2 = np.zeros((n2,))

    phi_unwrapped = np.linspace(0, 10*360, n) + np.random.rand(n)
    phi_wrapped = phi_unwrapped % 360
    ind = np.round(phi_wrapped / 360 * (n2 - 1)).astype(int)
    x2 = np.zeros((n, n2))
    x2[np.arange(n), ind] = 1

    # xx = np.concatenate((x1, x2), axis=1)
    y = np.exp(np.dot(x1, w1) + np.dot(x2, w2))
    # y = np.exp(np.dot(x2, w2)) / dt

    return w1, w2, x1, x2, y


if __name__ == '__main__':

    n = 20
    dt = .1

    w1, w2, x1, x2, y = create_test_dataset(n=n, dt=dt)
    x = np.concatenate((x1, x2), axis=1)

    # fit GLM for a fixed set up hyperparameters
    ind1 = np.arange(len(w2))
    ind2 = len(w1) + np.arange(len(w2))
    penalties = [RoughnessPenalty1D(ind1, 'beta1', 2),
                 L2Penalty(ind1, 'alpha1', .5),
                 RoughnessPenalty1DCircular(ind2, 'beta2', 2)]
    model = PoissonGLM(penalties=penalties, dt=dt)
    model.fit(x, y)

    model.predict()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(w1, 'r-')
    ax1.plot(model.coef_[:len(w1)], 'k-')
    ax2.plot(w2, 'r-')
    ax2.plot(model.coef_[len(w1):], 'k-')
    plt.show(block=True)
