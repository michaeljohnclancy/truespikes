import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def plot_performance_matrix(datasets, model, metric=None, save_path=None):
    # Takes a dict with keys=sorter_names and values=the associated dataset
    sorter_names = datasets.keys()

    accuracy_matrix = pd.DataFrame(index=sorter_names, columns=sorter_names)
    for train_sorter_name in sorter_names:
        model.fit(datasets[train_sorter_name]['X_train'], datasets[train_sorter_name]['y_train'])

        for test_sorter_name in sorter_names:
            if metric is None:
                accuracy_matrix[train_sorter_name][test_sorter_name] = model.score(datasets[test_sorter_name]['X_test'], datasets[test_sorter_name]['y_test'])
            else:
                y_preds = model.predict(datasets[test_sorter_name]['X_test'])
                accuracy_matrix[train_sorter_name][test_sorter_name] = metric(datasets[test_sorter_name]['y_test'], y_preds)


    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(accuracy_matrix.T.to_numpy(dtype=float), annot=True, xticklabels=sorter_names, yticklabels=sorter_names, ax=ax)
    ax.set_ylabel('Train Set')
    ax.set_xlabel('Test set')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)