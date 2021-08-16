from collections import OrderedDict
from typing import List
from consts import *

import pymc3 as pm
import numpy as np
import theano.tensor as tt


def build_logistic_regression_model(X_train, y_train):
    weights_init = [0.18, 0.18, 0.025, 0.020, 0.17,
                    0.08, 0.08, 0.025, 0.025, 0.035,
                    0.055, 0.025, 0.025]

    with pm.Model() as lr_model:

        lr_input = pm.Data(name='lr_input', value=X_train)
        lr_output = pm.Data(name='lr_output', value=y_train)

        # mu = pm.Normal(name="mu", mu=0.0, sigma=1.0, shape=(X_train.shape[1],))
        # sigma = pm.Normal(name="sigma", mu=0.0, sigma=1.0, shape=(X_train.shape[1], ))

        weights = pm.Normal(name='weights', mu=0, sigma=1, shape=(X_train.shape[1],))
        bias = pm.Normal(name='bias', mu=0, sigma=1)

        lr_preds = pm.math.sigmoid(pm.math.dot(lr_input, weights) + bias)

        out = pm.Bernoulli(
            name='out',
            p=lr_preds,
            observed=lr_output,
        )

    return lr_model


def build_heirarchical_logistic_regression_model(X_train, y_train):
    weights_init = np.random.randn(X_train.shape[1])

    with pm.Model() as lr_model:

        X_observed = pm.Data(name='X_observed', value=X_train)
        y_observed = pm.Data(name='y_observed', value=y_train)

        weights_mu = pm.Normal(name="weights_mu", mu=0.0, sigma=3)
        weights_sigma = pm.HalfNormal(name="weights_sigma", sigma=5.0)
        bias_mu = pm.Normal(name="bias_mu", mu=0.0, sigma=3)
        bias_sigma = pm.HalfNormal(name="bias_sigma", sigma=5.0)

        weights = pm.Normal(name='weights', mu=weights_mu, sigma=weights_sigma,
                            shape=(X_train.shape[1],), testval=weights_init)

        bias = pm.Normal(name='bias', mu=bias_mu, sigma=bias_sigma)

        # sigma = pm.HalfCauchy(name="sigma", beta=5)
#        sigma = pm.HalfCauchy(name="sigma", mu=0.0, sigma=1.0, shape=(X_train.shape[1], ))

        y = pm.Bernoulli(
            name='y',
            p=pm.math.sigmoid(pm.math.dot(X_observed, weights) + bias),
            observed=y_observed,
        )

    return lr_model


def build_bayesian_linear_regression_model(X_train, y_train, metric_names, sorter_names):
    coords = {'sorter_names': sorter_names, 'metric_names': metric_names, 'n_obs': np.arange(X_train.shape[0])}
    weights_offset_init = np.random.rand(len(sorter_names), len(metric_names))
    bias_offset_init = np.random.rand(len(sorter_names), len(metric_names))
    weights_mu_init = np.random.random(len(metric_names))
    bias_mu_init = np.random.random(len(metric_names))

    with pm.Model(coords=coords) as model:
        X_observed = pm.Minibatch(name='X_observed', data=X_train[metric_names].values, batch_size=batch_size)
        y_observed = pm.Minibatch(name='y_observed', data=y_train.values, batch_size=batch_size)
        sorter_idx = pm.Minibatch(name='sorter_idx', data=X_train['sorter_id'].values, batch_size=batch_size)

        weights_mu = pm.Normal(name="weights_mu", mu=0, sigma=3, dims='metric_names',  testval=weights_mu_init)
        weights_sigma = pm.HalfCauchy(name="weights_sigma", beta=5, dims='metric_names')

        bias_mu = pm.Normal(name="bias_mu", mu=0, sigma=1, dims='metric_names',  testval=bias_mu_init)
        bias_sigma = pm.HalfCauchy(name="bias_sigma", beta=3, dims='metric_names')

        weights_offset = pm.Normal(
            name='weights_offset',
            mu=0, sigma=3,
            testval=weights_offset_init,
            dims=('sorter_names', 'metric_names')
        )

        weights = pm.Deterministic(
            name='weights',
            var=(weights_offset * weights_sigma) + weights_mu,
            dims=('sorter_names', 'metric_names')
        )

        bias_offset = pm.Normal(
            name='bias_offset',
            mu=0, sigma=1,
            testval=bias_offset_init,
            dims=('sorter_names', 'metric_names')
        )

        bias = pm.Deterministic(
            name='bias',
            var=(bias_offset * bias_sigma) + bias_mu,
            dims=('sorter_names', 'metric_names')
        )


        tt.printing.Print('X_obs')(X_observed.shape)
        tt.printing.Print('Weights')(weights[sorter_idx])
        tt.printing.Print('Bias')(bias[sorter_idx].shape)

        agreement_score_estimate = pm.Deterministic(
            name='agreement_score_estimate',
            var=pm.math.sum(
                (X_observed * weights[sorter_idx]) + bias[sorter_idx],
                axis=1
            )
        )

        sigma = pm.HalfCauchy(name="sigma", beta=5)

        agreement_score_likelihood = pm.Normal(
            name='agreement_score_likelihood',
            mu=agreement_score_estimate,
            sigma=sigma,
            observed=y_observed,
            total_size=X_train.shape[0]
        )

    return model

def build_cholesky_bayesian_linear_regression_model(X_train, y_train, metric_names, sorter_names, batch_size=None):
    coords = {'sorter_names': sorter_names, 'metric_names': metric_names, 'n_obs': np.arange(X_train.shape[0]), "param": ['weights', 'bias']}
    weights_offset_init = np.random.random(len(metric_names))
    bias_offset_init = np.random.random(len(metric_names))

    with pm.Model(coords=coords) as model:
        if batch_size is not None:
            X_observed = pm.Minibatch(name='X_observed', data=X_train[metric_names].values, batch_size=batch_size)
            y_observed = pm.Minibatch(name='y_observed', data=y_train.values, batch_size=batch_size)
            sorter_idx = pm.Minibatch(name='sorter_idx', data=X_train['sorter_id'].values, batch_size=batch_size)
        else:
            X_observed = pm.Data(name='X_observed', value=X_train[metric_names].values)
            y_observed = pm.Data(name='y_observed', value=y_train.values)
            sorter_idx = pm.Data(name='sorter_idx', value=X_train['sorter_id'].values)

        sd_dist = pm.HalfCauchy.dist(5)

        weights_offset = pm.Normal(name="weights", mu=0.0, sigma=3, dims='metric_names', testval=weights_offset_init)
        bias_offset = pm.Normal(name="bias", mu=0.0, sigma=1, dims='metric_names', testval=bias_offset_init)

        chol, corr, stds = pm.LKJCholeskyCov(name='chol', n=2, eta=2.0, sd_dist=sd_dist, compute_corr=True)

        z = pm.Normal(name="z", mu=0, sigma=1, dims=('metric_names', 'param', 'sorter_names'))

        tt.printing.Print('weights')(weights_offset.shape)
        tt.printing.Print('chol')(chol.shape)
        tt.printing.Print('z')(z.shape)

        weight_bias_mv = pm.Deterministic("weight_bias_mv", tt.dot(chol, z).T, dims=('sorter_names', 'metric_names', 'param'))

        tt.printing.Print('weight_bias_mv')(weight_bias_mv.shape)
        tt.printing.Print('weight_bias_mv[sorter_idx, :, 0]')(weight_bias_mv[sorter_idx, :, 0].shape)

        agreement_score_estimate = pm.Deterministic(
            name='agreement_score_estimate',
            var=pm.math.sum(
                (weights_offset + weight_bias_mv[sorter_idx, :, 0]) * X_observed + (bias_offset + weight_bias_mv[sorter_idx, :, 1]),
                axis=1
            )
        )

        sigma = pm.HalfCauchy(name="sigma", beta=5.0)

        agreement_score_likelihood = pm.Normal(
            name='agreement_score_likelihood',
            mu=agreement_score_estimate,
            sigma=sigma,
            observed=y_observed,
            dims='n_obs',
            total_size=X_train.shape[0]
        )

        return model

