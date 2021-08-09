from typing import Optional, List, Callable

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


# def plot_feature_histograms(
#         dataset: pd.DataFrame,
#         filter_by_n_deviations: Optional[float] = 100,
#         fig_out: Optional[str] = None,
#         title: Optional[str] = None,
#         **kwargs
# ):
#
#     fig, ax = plt.subplots(**kwargs)
#     filter_dataframe_outliers(dataset, n_deviations=filter_by_n_deviations).hist(ax=ax)
#
#     if title is not None:
#         plt.suptitle(title)
#
#     if fig_out is not None:
#         plt.savefig(fig_out)
#

def _plot_sns(
        plot_func: Callable,
        df: pd.DataFrame,
        feature_names: List[str],
        by: str,
        n_deviations: Optional[float] = None,
        title: Optional[str] = None,
        fig_out: Optional[str] = None,
        col_wrap: Optional[int] = None,
        **kwargs
):

    g = sns.FacetGrid(
        pd.melt(
            filter_dataframe_outliers(df=df, n_deviations=n_deviations),
            id_vars=[by],
            var_name='feature_name',
            value_vars=feature_names
        ), col='feature_name', hue=by, sharex=False, sharey=False,
        legend_out=True, col_wrap=col_wrap, gridspec_kws=kwargs
    )

    g.map(plot_func, 'value')
    g.add_legend()

    if title is not None:
        g.fig.suptitle(title)

    if fig_out is not None:
        plt.savefig(fig_out)


def plot_feature_histograms(
    df: pd.DataFrame,
    feature_names: List[str],
    bins: Optional[int] = 10,
    n_deviations: Optional[float] = None,
    by: Optional[str] = None,
    fig_out: Optional[str] = None,
    title: Optional[str] = None,
    **kwargs
):
    if by is None:
        fig, ax = plt.subplots(**kwargs)
        filter_dataframe_outliers(df, n_deviations=n_deviations).hist(ax=ax, bins=bins)

        if title is not None:
            fig.suptitle(title)

        if fig_out is not None:
            plt.savefig(fig_out)

    else:
        _plot_sns(
            plot_func=sns.histplot,
            df=df,
            feature_names=feature_names,
            by=by,
            n_deviations=n_deviations,
            fig_out=fig_out,
            title=title,
            **kwargs
        )


def plot_feature_kdes(
        df: pd.DataFrame,
        feature_names: List[str],
        n_deviations: Optional[float] = None,
        by: Optional[str] = None,
        fig_out: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
):

    if by is None:
        fig, ax = plt.subplots(**kwargs)
        filter_dataframe_outliers(df, n_deviations=n_deviations).hist(ax=ax)

        if title is not None:
            fig.suptitle(title)

        if fig_out is not None:
            plt.savefig(fig_out)

    _plot_sns(
        plot_func=sns.kdeplot,
        df=df,
        feature_names=feature_names,
        by=by,
        n_deviations=n_deviations,
        fig_out=fig_out,
        title=title,
        **kwargs
    )
