from collections import defaultdict
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from sf_utils import SFStudySet
from utils import get_study_set_metrics_data
import pprint


def stratified_k_fold(model, X, y, n_splits=5):
    models = []
    scores = []

    kf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in kf.split(X, y):
        print('train -  {}   |   test -  {}'.format(
            np.bincount(y[train_index]), np.bincount(y[test_index])))

        model.fit(X.iloc[train_index], y.iloc[train_index])

        y_test_preds = model.predict(X.iloc[test_index])
        models.append(model)
        scores.append(f1_score(y.iloc[test_index], y_test_preds))

    return models, scores


def get_feature_importances(rfcs: List[RandomForestClassifier], scores: List[float]):
    importances = rfcs[np.argmax(scores)].feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfcs],
                 axis=0)
    return importances, std


def plot_rfc_feature_importances(rfcs, scores, metric_names, title='Feature importances', output=None):
    importances, std = get_feature_importances(rfcs, scores=scores)

    indices = np.argsort(importances)
    ordered_metrics = [metric_names[i] for i in indices]
    # Plot the feature importances of the forest
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.barh(np.array(range(len(importances))), importances[indices],xerr=std[indices], color='r', align='center')
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    ax.set_yticks(range(len(importances)))
    ax.set_ylim([-1, len(importances)])
    ax.set_yticklabels(ordered_metrics)
    plt.tight_layout()

    if output is not None:
        plt.savefig(output)


def rfc_feature_importance_analysis(
        study_set_names: List[str],
        metric_names: List[str],
        random_state: int = 0,
        fig_output: Optional[str] = None):

    metric_data = get_study_set_metrics_data(
        study_set_names=study_set_names,
        train_test_split=True,
        metric_names=metric_names,
        random_state=random_state
    )

    rfcs, scores = stratified_k_fold(
        model=RandomForestClassifier(random_state=random_state),
        X=metric_data['X_train'], y=metric_data['y_train'], n_splits=5
    )

    importance_ranked_metrics = ([[metric_names[i] for i in np.argsort(rfc.feature_importances_)] for rfc in rfcs])

    plot_rfc_feature_importances(rfcs=rfcs, scores=scores, metric_names=metric_names, output=fig_output)

    # best_estimator = rfcs[np.argmax(scores)]

    test_set_f1_scores = [f1_score(rfc.predict(metric_data["X_test"]), metric_data["y_test"]) for rfc in rfcs]
    pprint.pprint(f"F1 Score for each RFC on held out test set: {test_set_f1_scores}; "
                  f"mean={np.mean(test_set_f1_scores)}; std={np.std(test_set_f1_scores)}")

    pprint.pprint(f'Metrics ranked by importance: {importance_ranked_metrics}')


def split_study_sets_by_electrode_type(study_set_names: List[str] = None):
    if study_set_names is None:
        study_sets = SFStudySet.get_all_available_study_sets()
    else:
        study_sets = [SFStudySet.load(study_set_name=study_set_name) for study_set_name in study_set_names]

    study_sets_by_electrode_type = defaultdict(list)

    for study_set in study_sets:
        study_sets_by_electrode_type[str(study_set.info['electrode_type'])].append(study_set.name)

    for k, v in study_sets_by_electrode_type.items():
        print(f'Electrode type {k} found in study sets: {v}')

    return study_sets_by_electrode_type
