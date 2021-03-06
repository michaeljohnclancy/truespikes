from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Union, Callable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn import model_selection, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from consts import *
from sf_utils import get_metric_metadata, SFStudySet

LOG_PATH = Path('logs')
LOG_PATH.mkdir(exist_ok=True)


def get_study_set_metrics_data(study_set_names: List[str], metric_data: Dict = None, **kwargs
                               ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    return parse_sf_metrics(
        metric_data if metric_data is not None else get_metric_metadata(),
        study_names=[study_name
                     for study_set_name in study_set_names
                     for study_name in SFStudySet.load(study_set_name).get_study_names()
                     ],
        **kwargs
    )


def get_study_metrics_data(study_names: List[str], metric_data: Dict = None, **kwargs
                           ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    return parse_sf_metrics(
        metric_data if metric_data is not None else get_metric_metadata(),
        study_names=study_names,
        **kwargs
    )


def get_recording_metrics_data(recording_names: List[str], metric_data: Dict = None, **kwargs
                               ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    return parse_sf_metrics(
        metric_data if metric_data is not None else get_metric_metadata(),
        recording_names=recording_names,
        **kwargs
    )


def parse_sf_metrics(
        sf_data, sorter_names: Optional[List[str]] = None, recording_names: Optional[List[str]] = None,
        study_names: Optional[List[str]] = None, exclude_sorter_names: Optional[List[str]] = None,
        exclude_recording_names: Optional[List[str]] = None, exclude_study_names: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None, exclude_metric_names: Optional[List[str]] = None,
        by_sorter: bool = False, by_recording: bool = False, by_study: bool = False, include_meta: bool = False,
        train_test_split: bool = False, with_agreement_scores: bool = False, one_hot_encode_sorter_name: bool = False,
        transformation: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None, random_state: int = 0
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Parses the SpikeForest json formatted results, and can be filtered by sorter name, study name and metric."""
    if sorter_names is not None and exclude_sorter_names is not None:
        raise ValueError("Can't provide list of sorters to include and exclude!")
    elif recording_names is not None and exclude_recording_names is not None:
        raise ValueError("Can't provide list of recordings to include and exclude!")
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

    sorter_names, exclude_sorter_names, study_names, exclude_study_names, \
    recording_names, exclude_recording_names, metric_names, exclude_metric_names = convert_arguments_to_lowercase(
        sorter_names, exclude_sorter_names, study_names, exclude_study_names,
        recording_names, exclude_recording_names, metric_names, exclude_metric_names
    )

    with open(LOG_PATH / f'parse_sf_results-{datetime.now().strftime("%Y%m%d%H%M%S")}.log', 'w') as logfile:
        results = None
        for entry in sf_data:
            if ((study_names is None and exclude_study_names is None)
                or (study_names is not None and entry['studyName'].lower() in study_names)
                or (exclude_study_names is not None and entry['studyName'].lower() not in exclude_study_names)) \
                    and ((sorter_names is None and exclude_sorter_names is None)
                         or (sorter_names is not None and entry['sorterName'].lower() in sorter_names)
                         or (exclude_sorter_names is not None and
                             entry['sorterName'].lower() not in exclude_sorter_names)) \
                    and ((recording_names is None and exclude_recording_names is None)
                         or (recording_names is not None and entry['recordingName'].lower() in recording_names)
                         or (exclude_recording_names is not None and
                             entry['recordingName'].lower() not in exclude_recording_names)):

                try:
                    if not isinstance(entry['quality_metric'], dict):
                        raise LookupError(f'Quality metric calculations failed serverside: {entry["quality_metric"]}')
                    if not isinstance(entry['ground_truth_comparison'], dict):
                        raise LookupError(
                            f'Ground truth comparison failed serverside: {entry["ground_truth_comparison"]}')
                    entry_df = pd.DataFrame(entry['quality_metric'])
                    entry_df.columns = map(str.lower, entry_df.columns)
                    entry_df.dropna(inplace=True, axis=1)

                    if transformation is not None:
                        entry_df = transformation(entry_df)
                    if with_agreement_scores:
                        entry_df['agreement_score'] = pd.DataFrame(
                            entry['ground_truth_comparison']['agreement_scores']).max(axis=0)

                    else:
                        entry_df['fp'] = np.array(entry['ground_truth_comparison']['best_match_21']) == -1

                    entry_df['sorterName'] = np.array([entry['sorterName']] * entry_df.shape[0])
                    entry_df['studyName'] = np.array([entry['studyName']] * entry_df.shape[0])
                    entry_df['recordingName'] = np.array([entry['recordingName']] * entry_df.shape[0])
                    if results is None:
                        results = entry_df
                    else:
                        results = results.append(entry_df)
                except LookupError as e:
                    logfile.write(
                        f"{entry['studyName']} - {entry['recordingName']} - {entry['sorterName']} - {str(e)} \n")

    if metric_names is not None:
        results = results[metric_names + ['studyName', 'recordingName', 'sorterName'] + [y_var]]
    elif exclude_metric_names is not None:
        results.drop(columns=exclude_metric_names, inplace=True)

    if group_by is not None:
        results = _split_dataset(
            results, group_by=group_by, remove_meta=not include_meta,
            train_test_split=train_test_split, y_var_name=y_var
        )
    elif train_test_split:
        metrics = results.drop(columns=[y_var])
        if not include_meta:
            metrics = metrics.drop(columns=['studyName', 'recordingName'])
        if one_hot_encode_sorter_name:
            one_hot = pd.get_dummies(metrics[['sorterName']])
            metrics = metrics.drop(columns=['sorterName'])
            metrics = pd.concat([metrics, one_hot], axis=1)
        else:
            metrics = metrics.drop(columns=['sorterName'])

        y_data = results[y_var]
        X_train, X_test, y_train, y_test = model_selection.train_test_split(metrics, y_data, test_size=0.2,
                                                                            random_state=random_state)
        results = {'X_train': X_train.reset_index(drop=True), 'y_train': y_train.reset_index(drop=True),
                   'X_test': X_test.reset_index(drop=True), 'y_test': y_test.reset_index(drop=True)}
    elif not include_meta:
        results = results.drop(columns=['studyName', 'recordingName'])
        if one_hot_encode_sorter_name:
            one_hot = pd.get_dummies(results[['sorterName']])
            results = results.drop(columns=['sorterName'])
            results = pd.concat([results, one_hot], axis=1)
        else:
            results = results.drop(columns=['sorterName'])

        results.reset_index(drop=True, inplace=True)
    else:
        results.reset_index(drop=True, inplace=True)

    return results


def _split_dataset(df: pd.DataFrame, group_by: str, remove_meta: bool = True, train_test_split: bool = True,
                   y_var_name='fp', one_hot_encode_sorter_name: bool = False, random_state: int = 0):
    grouped = df.groupby(group_by)

    datasets = {}
    if train_test_split:
        datasets['all_data'] = {}
    for attr_name, df in grouped:
        attr_name: str = attr_name.lower()
        metrics = df.drop(columns=[y_var_name])
        if remove_meta:
            metrics = metrics.drop(columns=['studyName', 'recordingName'])
        if one_hot_encode_sorter_name:
            one_hot = pd.get_dummies(metrics['sorterName'])
            metrics = metrics.drop(columns=['sorterName'])
            metrics = metrics.join(one_hot)
        else:
            metrics = metrics.drop(columns=['sorterName'])

        metrics.reset_index(inplace=True)

        y_data = df[y_var_name]

        if train_test_split:
            datasets[attr_name] = {}
            datasets[attr_name]['X_train'], datasets[attr_name]['X_test'], \
            datasets[attr_name]['y_train'], datasets[attr_name]['y_test'] = \
                model_selection.train_test_split(metrics, y_data, test_size=0.2, random_state=random_state)

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


def convert_arguments_to_lowercase(
        sorter_names: Optional[List[str]] = None,
        recording_names: Optional[List[str]] = None,
        study_names: Optional[List[str]] = None,
        exclude_sorter_names: Optional[List[str]] = None,
        exclude_recording_names: Optional[List[str]] = None,
        exclude_study_names: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None,
        exclude_metric_names: Optional[List[str]] = None
):
    if sorter_names is not None:
        sorter_names = [sorter_name.lower() for sorter_name in sorter_names]
    if exclude_sorter_names is not None:
        exclude_sorter_names = [sorter_name.lower() for sorter_name in exclude_sorter_names]
    if study_names is not None:
        study_names = [study_name.lower() for study_name in study_names]
    if exclude_study_names is not None:
        exclude_study_names = [study_name.lower() for study_name in exclude_study_names]
    if recording_names is not None:
        recording_names = [recording_name.lower() for recording_name in recording_names]
    if exclude_recording_names is not None:
        exclude_recording_names = [recording_name.lower() for recording_name in exclude_recording_names]
    if metric_names is not None:
        metric_names = [metric_name.lower() for metric_name in metric_names]
    if exclude_recording_names is not None:
        exclude_metric_names = [metric_name.lower() for metric_name in exclude_metric_names]

    return sorter_names, exclude_sorter_names, study_names, exclude_study_names, \
           recording_names, exclude_recording_names, metric_names, exclude_metric_names


def get_performance_matrix(datasets: Dict[str, pd.DataFrame], model_cls, metric: Union[str, Callable] = None):
    # Takes a dict with keys=sorter_names and values=the associated dataset
    sorter_names = datasets.keys()

    performance_matrix = pd.DataFrame(index=sorter_names, columns=sorter_names)
    for train_sorter_name in sorter_names:
        model = model_cls()
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


# def score(study_set_names: List[str], model, one_hot_encode_sorter_name=False):
#     dataset = get_study_set_metrics_data(study_set_names, train_test_split=True, one_hot_encode_sorter_name=one_hot_encode_sorter_name)
#     print(dataset['X_train'].shape)
#
#     model.fit(dataset['X_train'], dataset['y_train'])
#
#     y_test_preds = model.predict(dataset['X_test'])
#     f1 = f1_score(dataset['y_test'], y_test_preds)
#
#     print(f'Linear SVC F1-Score is {f1}')
#     print(accuracy_score(dataset['y_test'], y_test_preds))
#

def filter_dataframe_outliers(df: pd.DataFrame, n_deviations: Optional[float] = None) -> pd.DataFrame:
    if n_deviations is None:
        return df

    return df[(
        np.abs(zscore(df.select_dtypes(include=['float'])) < n_deviations).all(axis=1)
    )]


def apply_standard_scalar(df: pd.DataFrame) -> pd.DataFrame:
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.StandardScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_scaled, columns=df.columns)


def stratified_sample_df(df, col, n_samples):
    # https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


def logit_transform(data):
    return np.log(data / (1 - data))


def inverse_logit_transform(data):
    return 1 / (np.exp(-data) + 1)


# Allows for log transformations x
def trim_ones_and_zeros(df):
    subset = df.copy()
    epsilon = subset[subset != 0].min()

    subset[subset == 1] -= epsilon
    subset[subset == 0] = epsilon
    return subset


def inverse_trim_ones_and_zeros(df):
    subset = df.copy()
    epsilon = subset[subset != 0].min()

    subset[subset == 1] += epsilon
    subset[subset == 0] -= epsilon

    return subset


def squeezed_logit_transform(df):
    return logit_transform(trim_ones_and_zeros(df))


def inverse_squeezed_logit_transform(df):
    return inverse_trim_ones_and_zeros(inverse_logit_transform(df))


def squeezed_log_transform(df):
    return np.log(df + 1)


def preprocess_agreement_score_dataset(
        agreement_score_dataset_path: Union[Path, str],
):
    if Path(agreement_score_dataset_path).exists():
        data_store = pd.HDFStore(agreement_score_dataset_path, mode='r')
        data = data_store['dataset'].copy()

        print("Loaded from cache")
    else:
        data_store = pd.HDFStore(agreement_score_dataset_path, mode='w')
        data: pd.DataFrame = get_study_metrics_data(
            study_names=STATIC_TETRODE_STUDY_NAMES,
            metric_names=METRIC_NAMES,
            random_state=RANDOM_STATE,
            sorter_names=SORTER_NAMES,
            include_meta=True,
            with_agreement_scores=True
        ).dropna(axis=0)

        data_store['dataset'] = data.copy()

    data_store.close()

    # Drop rows with null values.
    return data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]


def prepare_transformed_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_path = '/home/mclancy/truespikes/data/static_tetrode_dataset.hd5'
    data = preprocess_agreement_score_dataset(data_path)
    # data = stratified_sample_df(preprocess_agreement_score_dataset(data_path), 'sorterName', 1000)

    data['sorter_id'] = data[['sorterName']].applymap(lambda x: SORTER_NAMES.index(str(x).lower()))
    # data.drop(columns=['cumulative_drift', 'max_drift', 'studyName', 'recordingName', 'sorterName'], inplace=True)
    data.drop(
        columns=['cumulative_drift', 'max_drift', 'studyName', 'recordingName',
                 'sorterName', 'presence_ratio', 'l_ratio', 'isi_violation', 'amplitude_cutoff'],
        inplace=True
    )

    transformations = {'firing_rate': lambda x: np.log(x),
                       'isi_violation': lambda x: np.log(x + 0.00001),
                       'snr': lambda x: np.sqrt(x),
                       'presence_ratio': squeezed_logit_transform,
                       'amplitude_cutoff': lambda x: squeezed_logit_transform(2 * x),
                       'isolation_distance': lambda x: np.log(x + 0.00001),
                       'nn_hit_rate': squeezed_logit_transform,
                       'nn_miss_rate': squeezed_logit_transform,
                       'silhouette_score': lambda x: x,
                       'l_ratio': lambda x: np.log(x + 0.00001),
                       'd_prime': lambda x: np.log(x + 0.00001),
                       'agreement_score': squeezed_logit_transform,
                       }

    transformed_data = pd.DataFrame(index=data.index, columns=data.columns)
    for column in data.columns:
        if column in transformations:
            transformed_data[column] = transformations[column](data[column])
        else:
            transformed_data[column] = data[column]

    transformed_train_data, transformed_test_data = train_test_split(
        transformed_data, shuffle=True, test_size=0.2, stratify=transformed_data['sorter_id'])

    gaussian_metrics = ["firing_rate", "snr", "isolation_distance",
                        "silhouette_score", "nn_hit_rate",
                        "nn_miss_rate", "d_prime"
                        ]

    standard_scalar = StandardScaler()
    transformed_train_data[gaussian_metrics] = standard_scalar.fit_transform(transformed_train_data[gaussian_metrics])
    transformed_test_data[gaussian_metrics] = standard_scalar.transform(transformed_test_data[gaussian_metrics])

    # transformed_train_data = filter_dataframe_outliers(
    #     transformed_train_data, n_deviations=5
    # )
    transformed_train_data.reset_index(drop=True, inplace=True)
    transformed_test_data.reset_index(drop=True, inplace=True)

    return transformed_train_data, transformed_test_data
