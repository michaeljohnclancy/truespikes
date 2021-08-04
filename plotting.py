from typing import Optional

import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import zscore

from utils import filter_dataframe_outliers


def plot_performance_matrix(performance_matrix, save_path=None, annotate=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    sns.heatmap(performance_matrix.T.to_numpy(dtype=float), annot=annotate, xticklabels=performance_matrix.columns, yticklabels=performance_matrix.columns, ax=ax)
    ax.set_ylabel('Train Set')
    ax.set_xlabel('Test set')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)


def plot_feature_histograms(
        dataset: pd.DataFrame,
        filter_by_n_deviations: Optional[float] = 100,
        fig_out: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
):

    fig, ax = plt.subplots(**kwargs)
    filter_dataframe_outliers(dataset, n_deviations=filter_by_n_deviations).hist(ax=ax)

    if title is not None:
        fig.suptitle(title)

    if fig_out is not None:
        plt.savefig(fig_out)
