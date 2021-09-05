from consts import *
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import normalize, StandardScaler
from utils import preprocess_agreement_score_dataset, logit_transform, trim_ones_and_zeros, inverse_logit_transform, \
    inverse_trim_ones_and_zeros, squeezed_log_transform, squeezed_logit_transform

# ISOLATION_METRICS = ['isolation_distance', 'nn_hit_rate', 'nn_miss_rate', 'd_prime', 'l_ratio', 'silhouette_score']
ISOLATION_METRIC_NAMES = ['nn_hit_rate', 'nn_miss_rate', 'd_prime', 'l_ratio', 'silhouette_score']
metric_names = ISOLATION_METRIC_NAMES + ['firing_rate', 'presence_ratio', 'snr']

# Path to datastore
data_path = '/data/static_tetrode_dataset.hd5'


transformations = {'firing_rate': squeezed_log_transform,
                   'presence_ratio': lambda x: x == 1,
                   'd_prime': squeezed_log_transform,
                   'nn_hit_rate': squeezed_logit_transform,
                   'nn_miss_rate': squeezed_logit_transform,
                   'silhouette_score': squeezed_log_transform,
                   'l_ratio': squeezed_log_transform,
                   'snr': squeezed_log_transform,
                   'agreement_score': squeezed_logit_transform,
                   }

data = preprocess_agreement_score_dataset(data_path, transform_dict=transformations)


# Temp variable to help with seeing rough separation of good and and units
data['AS >= 0.5'] = data['agreement_score'] >= 0.5

sns.pairplot(
    data, vars=metric_names, hue='AS >= 0.5',
).savefig('figures/isolation_pairplot.pdf')
plt.clf()
print('done pairplot')

# Isolation distance difficult to use as it depends on the number of waveforms used in the calculation
for metric_name in metric_names:
    # sns.kdeplot(
    #     x=data[metric_name], hue=inverse_squeezed_logit_transform(data['agreement_score'])
    # ).figure.savefig(f'figures/{metric_name}_histogram.pdf')
    # plt.clf()
    #
    sns.scatterplot(
        x=data[metric_name], y=data['agreement_score']
    ).figure.savefig(f'figures/{metric_name}_agreement_score_scatterplot.pdf')
    plt.clf()
#
# for metric_name_1, metric_name_2 in itertools.combinations(metric_names, 2):
#     sns.scatterplot(
#         x=data[metric_name_1], y=data[metric_name_2], hue=inverse_squeezed_logit_transform(data['agreement_score'])
#     ).figure.savefig(f'figures/feature_scatters/{metric_name_1}_{metric_name_2}_agreement_score_scatterplot.pdf')
#     plt.clf()
