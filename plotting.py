import seaborn as sns

import matplotlib.pyplot as plt


def plot_performance_matrix(performance_matrix, save_path=None, annotate=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    sns.heatmap(performance_matrix.T.to_numpy(dtype=float), annot=annotate, xticklabels=performance_matrix.columns, yticklabels=performance_matrix.columns, ax=ax)
    ax.set_ylabel('Train Set')
    ax.set_xlabel('Test set')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
