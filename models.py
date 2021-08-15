from collections import OrderedDict
from typing import List

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


def build_bayesian_linear_regression_model(X_train, y_train, metric_names, sorter_names, hierarchical=False):
    coords = {'sorter_name': sorter_names, 'metric_name': metric_names, 'obs_n': np.arange(X_train.shape[0])}
    weights_init = np.squeeze(np.random.rand(len(sorter_names), len(metric_names)))
    bias_init = np.squeeze(np.random.rand(len(sorter_names), len(metric_names)))
    weights_mu_init = np.random.random(len(metric_names))
    weights_sigma_init = np.random.random(len(metric_names))
    bias_mu_init = np.random.random(len(metric_names))
    bias_sigma_init = np.random.random(len(metric_names))

    BATCH_SIZE = 64


    with pm.Model(coords=coords) as model:
        X_observed = pm.Minibatch(name='X_observed', data=X_train[metric_names].values, batch_size=BATCH_SIZE)
        y_observed = pm.Minibatch(name='y_observed', data=y_train.values, batch_size=BATCH_SIZE)
        sorter_idx = pm.Minibatch(name='sorter_idx', data=X_train['sorter_id'].values, batch_size=BATCH_SIZE)

        if hierarchical:
            weights_mu = pm.Normal(name="weights_mu", dims='metric_name', mu=np.zeros(shape=len(metric_names)), sigma=np.ones(shape=len(metric_names))*5, testval=weights_mu_init)
            weights_sigma = pm.Normal(name="weights_sigma", dims='metric_name', mu=np.zeros(shape=len(metric_names)), sigma=np.ones(shape=len(metric_names))*5, testval=weights_sigma_init)
            bias_mu = pm.Normal(name="bias_mu", dims='metric_name', mu=np.zeros(shape=len(metric_names)), sigma=np.ones(shape=len(metric_names))*5, testval=bias_mu_init)
            bias_sigma = pm.Normal(name="bias_sigma", dims='metric_name', mu=np.zeros(shape=len(metric_names)), sigma=np.ones(shape=len(metric_names))*5, testval=bias_sigma_init)
        else:
            weights_mu = np.zeros(shape=len(metric_names))
            weights_sigma = np.zeros(shape=len(metric_names))*5
            bias_mu = np.zeros(shape=len(metric_names))
            bias_sigma = np.zeros(shape=len(metric_names))*5

        weights = pm.Normal(
            name='weights', mu=weights_mu, sigma=weights_sigma,
            dims=('sorter_name', 'metric_name'), testval=weights_init
        )

        bias = pm.Normal(
            name='bias', mu=bias_mu, sigma=bias_sigma,
            dims=('sorter_name', 'metric_name'), testval=bias_init
        )

        sigma = pm.HalfCauchy(name="sigma", beta=5)

        y_mu = pm.Deterministic(name='y_mu', var=pm.math.sum(X_observed * weights[sorter_idx]) + bias[sorter_idx].T)

        weight_print = tt.printing.Print("weight")(weights)
        bias_print = tt.printing.Print("bias")(bias)
        y_mu_print = tt.printing.Print("y_mu")(y_mu)

        y = pm.Normal(
            name='y',
            mu=y_mu,
            sigma=sigma,
            observed=y_observed,
        )

    return model


def build_bayesian_neural_network_model(X_train, y_train, hierarchical=False):
    weights_init = np.random.randn(X_train.shape[1])

    with pm.Model() as model:
        X_observed = pm.Data(name='X_observed', value=X_train)
        y_observed = pm.Data(name='y_observed', value=y_train)

        if hierarchical:
            weights_mu = pm.Normal(name="weights_mu", mu=0.0, sigma=3)
            weights_sigma = pm.HalfNormal(name="weights_sigma", sigma=1)
            bias_mu = pm.Normal(name="bias_mu", mu=0.0, sigma=3)
            bias_sigma = pm.HalfNormal(name="bias_sigma", sigma=1)
        else:
            weights_mu = 0
            weights_sigma = 1
            bias_mu = 0
            bias_sigma = 1

        # weights = pm.Normal(name='weights', mu=weights_mu, sigma=weights_sigma,
        #                     shape=(X_train.shape[1],), testval=weights_init)
        #
        # bias = pm.Normal(name='bias', mu=bias_mu, sigma=bias_sigma)

        sigma = pm.HalfCauchy(name="sigma", beta=1)
        # Weights from input to hidden layer

        weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(X_train.shape[1], 18))

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(18, 18))

        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=1, shape=(18,))

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(X_observed, weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

        y = pm.Normal(
            name='y',
            mu=act_out,
            sigma=sigma,
            observed=y_observed,
        )

    return model

def build_classification_pipeline(X_train, y_train, metric_names: List[str], sorter_names: List[str], hierarchical=False):
    weights_init = np.random.randn(X_train.shape[1])

    coords = {'sorter_name': sorter_names, 'obs_id': np.arange(X_train.shape[0])}

    with pm.Model(coords=coords) as model:
        sorter_idx = pm.Data("sorter_idx", X_train['sorter_id'].values, dims="obs_id")

        X_observed = pm.Data(name='X_observed', value=X_train[metric_names])
        y_observed = pm.Data(name='y_observed', value=y_train)

        if hierarchical:
            weights_mu = pm.Normal(name="weights_mu", mu=0.0, sigma=3)
            weights_sigma = pm.HalfNormal(name="weights_sigma", sigma=1)
            bias_mu = pm.Normal(name="bias_mu", mu=0.0, sigma=3)
            bias_sigma = pm.HalfNormal(name="bias_sigma", sigma=1)
        else:
            weights_mu = 0
            weights_sigma = 1
            bias_mu = 0
            bias_sigma = 1

        weights = pm.Normal(name='weights', mu=weights_mu, sigma=weights_sigma,
                            shape=(X_train[metric_names].shape[1],), testval=weights_init)

        bias = pm.Normal(name='bias', mu=bias_mu, sigma=bias_sigma)

        sigma = pm.HalfCauchy(name="sigma", beta=1)

        y = pm.Normal(
            name='y',
            mu=pm.math.dot(X_observed, weights[sorter_idx]) + bias,
            sigma=sigma,
            observed=y_observed,
            dims="obs_id"
        )

    return model

def get_posterior_predictive_on_unseen_data(trace, X_test, y_test):
    # change the value and shape of the data
    pm.set_data(
        {
            "X_observed": X_test,
            # use dummy values with the same shape:
            "y_observed": y_test,
        }
    )

    return pm.sample_posterior_predictive(trace)
