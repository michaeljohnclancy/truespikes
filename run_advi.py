from consts import METRIC_NAMES, SORTER_NAMES, RANDOM_STATE
from models import build_bayesian_linear_regression_model
from train import get_prepared_agreement_score_dataset, train

import numpy as np
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

X_train, y_train, X_test, y_test = get_prepared_agreement_score_dataset()

model = build_bayesian_linear_regression_model(
    X_train, y_train, metric_names=METRIC_NAMES,
    sorter_names=list(SORTER_NAMES.keys()),
    hierarchical=True
)

pm.model_to_graphviz(model)

with model:
    approx = pm.fit(1000000, callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-4)])
    step = pm.NUTS(scaling=approx.cov.eval(), is_cov=True)
    trace = pm.sample(start=approx.sample()[0], step=step, init='adapt_diag')#, return_inferencedata=True)
    # az.to_netcdf(data=trace, filename='/home/mclancy/truespikes/traces/hierarchical_linear_regression/trace.netcdf')

idata = az.from_pymc3(trace)
az.summary(idata, round_to=2)

az.plot_trace(trace)
plt.savefig('/home/mclancy/truespikes/figures/model_selection/bayesian_linear_regression/traces.pdf')

with model:
    pm.set_data(
        {
            "X_observed": X_test.drop(columns='sorter_id'),
            "sorter_idx": X_test['sorter_id'],
            "y_observed": y_test,
        }
    )
    ppc = pm.sample_posterior_predictive(
        trace, random_seed=RANDOM_STATE
    )

y_preds = [sample_pred.mean() for sample_pred in (1 / (np.exp(-ppc['y']) + 1)).T]

from sklearn.metrics import mean_squared_error

zeros = np.ones(shape=y_test.shape[0])
baseline_rmse = np.sqrt(mean_squared_error((1 / (np.exp(-y_test) + 1)), zeros))

rmse = np.sqrt(mean_squared_error((1 / (np.exp(-y_test) + 1)), y_preds))

print(baseline_rmse)
print(rmse)
