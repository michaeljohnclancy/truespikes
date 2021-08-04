import json
from collections import defaultdict
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import plot_tree
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from sf_utils import SFStudySet
from utils import get_study_set_metrics_data, get_study_metrics_data
import pprint


def stratified_k_fold(X, y, model_cls, model_args=None, n_splits=5):
    models = []
    scores = []
    if model_args is None:
        model_args = {}

    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    kf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in kf.split(X, y):
        print('train -  {}   |   test -  {}'.format(
            np.bincount(y[train_index]), np.bincount(y[test_index])))

        model = model_cls(**model_args)
        model.fit(X.iloc[train_index], y.iloc[train_index])

        y_test_preds = model.predict(X.iloc[test_index])
        models.append(model)
        scores.append(f1_score(y.iloc[test_index], y_test_preds))

    return models, scores


def get_feature_importances(rfcs: List[RandomForestClassifier], scores: List[float]):
    importances = rfcs[np.argmax(scores)].feature_importances_
    std = np.std([rfc.feature_importances_ for rfc in rfcs],
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


def plot_decision_tree(decision_tree, metric_names, fig_out: Optional[str] = None):
    fig = plt.figure(figsize=(25, 20))
    _ = plot_tree(decision_tree,
                  feature_names=metric_names,
                  class_names=['True unit', 'False Positive Unit'],
                  filled=True
                  )

    if fig_out is not None:
        fig.savefig(fig_out)


def rfc_feature_importance_analysis(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        metric_names: List[str],
        model_args: Optional[Dict] = None,
        fig_title: Optional[str] = 'Feature Importances',
        feature_importance_output: Optional[str] = None,
        tree_output: Optional[str] = None):

    rfcs, scores = stratified_k_fold(
        model_cls=RandomForestClassifier, model_args=model_args,
        X=X_train, y=y_train, n_splits=5
    )

    # importance_ranked_metrics = ([[metric_names[i] for i in np.argsort(rfc.feature_importances_)] for rfc in rfcs])

    plot_rfc_feature_importances(rfcs=rfcs, scores=scores, metric_names=metric_names, title=fig_title, output=feature_importance_output)

    plot_decision_tree(decision_tree=rfcs[np.argmax(scores)].estimators_[0], metric_names=metric_names, fig_out=tree_output)

    print(f'Trained on {X_train.shape[0]}; Tested on {X_test.shape[0]}')

    test_set_f1_scores = [f1_score(rfc.predict(X_test), y_test) for rfc in rfcs]
    pprint.pprint(f"F1 Score for each RFC on held out test set: {test_set_f1_scores}; "
                  f"mean={np.mean(test_set_f1_scores)}; std={np.std(test_set_f1_scores)}")


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


def split_recordings_by_probe_geometry(study_sets: List[SFStudySet], standardize_geometry: Optional[bool] = False) -> Dict[str, List[SFStudySet]]:
    results = defaultdict(list)
    for study_set in study_sets:
        for study in study_set.get_studies():
            for recording in study.get_recordings():
                key = str(recording.geom)
                if standardize_geometry:
                    key = np.array(key)
                    geom = key - key.min()
                    key = str(geom.tolist())
                results[key].append((recording.name, recording.study_name))

    for i, geometry in enumerate(results.keys()):
        fig, ax = plt.subplots()
        x, y = zip(*list(eval(geometry)))
        ax.scatter(x, y)
        plt.savefig(f'../figures/split_by_geometry/{i}.pdf')

    with open('../figures/split_by_geometry/geometries.json', 'a') as f:
        json.dump(results.values(), f, indent=4)

    return results
