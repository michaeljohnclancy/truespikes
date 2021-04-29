from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

LOG_PATH = Path('logs')
LOG_PATH.mkdir(exist_ok=True)


def parse_sf_results(
        sf_data, sorter_names: Optional[List[str]] = None, study_names: Optional[List[str]] = None,
        exclude_sorter_names: Optional[List[str]] = None, exclude_study_names: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None, exclude_metric_names: Optional[List[str]] = None,
        by_sorter: bool = False, by_study: bool = False, by_recording: bool = False, include_meta: bool = False,
        train_test_split: bool = True, with_agreement_scores: bool = False,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Parses the SpikeForest json formatted results, and can be filtered by sorter name, study name and metric."""
    if sorter_names is not None and exclude_sorter_names is not None:
        raise ValueError("Can't provide list of sorters to include and exclude!")
    elif study_names is not None and exclude_study_names is not None:
        raise ValueError("Can't provide list of studies to include and exclude!")
    elif metric_names is not None and exclude_metric_names is not None:
        raise ValueError("Can't provide list of metrics to include and exclude!")

    y_var = 'agreement_score' if with_agreement_scores else 'fp'
    group_by = None
    if (by_sorter and by_study) or (by_sorter and by_recording) or (by_study and by_recording):
        raise ValueError("Cannot split by more than one attribute")
    elif by_sorter:
        group_by = 'sorterName'
    elif by_study:
        group_by = 'studyName'
    elif by_recording:
        group_by = 'recordingName'

    with open(LOG_PATH / f'parse_sf_results-{datetime.now().strftime("%Y%m%d%H%M%S")}.log', 'w') as logfile:
        results = None
        for entry in sf_data:
            if ((study_names is None and exclude_study_names is None)
                or (study_names is not None and entry['studyName'] in study_names)
                or (exclude_study_names is not None and entry['studyName'] not in exclude_study_names)) \
                    and ((sorter_names is None and exclude_sorter_names is None)
                         or (sorter_names is not None and entry['sorterName'] in sorter_names)
                         or (exclude_sorter_names is not None and entry['sorterName'] not in exclude_sorter_names)
            ):
                try:
                    entry_df = pd.DataFrame(entry['quality_metric'])
                    if with_agreement_scores:
                        entry_df['agreement_score'] = pd.DataFrame(entry['ground_truth_comparison']['agreement_scores']).max(axis=0)

                    else:
                        entry_df['fp'] = np.array(entry['ground_truth_comparison']['best_match_21']) == -1

                    entry_df['sorterName'] = np.array([entry['sorterName']] * entry_df.shape[0])
                    entry_df['studyName'] = np.array([entry['studyName']] * entry_df.shape[0])
                    entry_df['recordingName'] = np.array([entry['recordingName']] * entry_df.shape[0])
                    if results is None:
                        results = entry_df
                    else:
                        results = results.append(entry_df)
                except Exception as e:
                    logfile.write(
                        f"{entry['studyName']} - {entry['recordingName']} - {entry['sorterName']} - {str(e)} \n")

    if metric_names is not None:
        results = results[metric_names + [y_var]]
    elif exclude_metric_names is not None:
        results.drop(columns=exclude_metric_names, inplace=True)

    results.dropna(inplace=True)

    if group_by is not None:
        results = _split_dataset(
            results, group_by=group_by, remove_meta=not include_meta,
            train_test_split=train_test_split, y_var_name=y_var
        )
    elif train_test_split:
        if include_meta:
            metrics = results.drop(columns=[y_var])
        else:
            metrics = results.drop(columns=[y_var, 'sorterName', 'studyName', 'recordingName'])

        y_data = results[y_var]
        X_train, X_test, y_train, y_test = model_selection.train_test_split(metrics, y_data, test_size=0.2, random_state=0)
        results = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    elif not include_meta:
        results = results.drop(columns=['sorterName', 'studyName', 'recordingName'])

    return results


def _split_dataset(df: pd.DataFrame, group_by: str, remove_meta: bool = True, train_test_split: bool = True, y_var_name='fp'):
    grouped = df.groupby(group_by)

    datasets = {}
    if train_test_split:
        datasets['all_data'] = {}
    for attr_name, df in grouped:
        if remove_meta:
            metrics = df.drop(columns=[y_var_name, 'sorterName', 'studyName', 'recordingName'])
        else:
            metrics = df.drop(columns=[y_var_name])

        y_data = df[y_var_name]

        if train_test_split:
            datasets[attr_name] = {}
            datasets[attr_name]['X_train'], datasets[attr_name]['X_test'], \
            datasets[attr_name]['y_train'], datasets[attr_name]['y_test'] = \
                model_selection.train_test_split(metrics, y_data, test_size=0.2, random_state=0)

            if 'X_train' not in datasets['all_data']:
                datasets['all_data']['X_train'] = datasets[attr_name]['X_train']
                datasets['all_data']['y_train'] = datasets[attr_name]['y_train']
                datasets['all_data']['X_test'] = datasets[attr_name]['X_test']
                datasets['all_data']['y_test'] = datasets[attr_name]['y_test']
            else:
                datasets['all_data']['X_train'] = np.vstack(
                    [datasets['all_data']['X_train'], datasets[attr_name]['X_train']])
                datasets['all_data']['y_train'] = np.hstack(
                    [datasets['all_data']['y_train'], datasets[attr_name]['y_train']])
                datasets['all_data']['X_test'] = np.vstack(
                    [datasets['all_data']['X_test'], datasets[attr_name]['X_test']])
                datasets['all_data']['y_test'] = np.hstack(
                    [datasets['all_data']['y_test'], datasets[attr_name]['y_test']])

        else:
            datasets[attr_name] = df
    return datasets


def get_performance_matrix(datasets, model, metric=None):
    # Takes a dict with keys=sorter_names and values=the associated dataset
    sorter_names = datasets.keys()

    performance_matrix = pd.DataFrame(index=sorter_names, columns=sorter_names)
    for train_sorter_name in sorter_names:
        model.fit(datasets[train_sorter_name]['X_train'], datasets[train_sorter_name]['y_train'])

        for test_sorter_name in sorter_names:
            if metric is None:
                performance_matrix[train_sorter_name][test_sorter_name] = model.score(
                    datasets[test_sorter_name]['X_test'], datasets[test_sorter_name]['y_test'])
            else:
                y_preds = model.predict(datasets[test_sorter_name]['X_test'])
                performance_matrix[train_sorter_name][test_sorter_name] = metric(datasets[test_sorter_name]['y_test'],
                                                                                 y_preds)

    return performance_matrix


class LogitRegression(LinearRegression):

    def fit(self, x, p):
        p = np.asarray(p)
        p = p * 1e-16 + 0.5 * 1e-16
        y = np.log(p / (1 - p))
        return super().fit(x, y)

    def predict(self, x):
        y = super().predict(x)
        return 1 / (np.exp(-y) + 1)