import pandas as pd
import requests
import datetime
from typing import List, Dict, Optional

import sortingview as sv
import kachery_client as kc

SF_SORTING_URI = 'sha1://52f24579bb2af1557ce360ed5ccc68e480928285/sortings.json'
SF_METRIC_URI = 'sha1://b3444629251cafda919af535f0e9837279151c6e/spikeforest-full-gt-qm.json?' \
                'manifest=cf73c99d06c11e328e635e14dc24b8db7372db3d'


class SFSorting:
    _metric_data: Optional[Dict] = None

    @staticmethod
    def load(sorter_name: str, study_name: str, recording_name: str):
        try:
            sorting_metadata = _get_sorting_metadata(
                sorter_name=sorter_name,
                recording_name=recording_name,
                study_name=study_name
            )[0]
        except IndexError:
            raise ValueError('Invalid sorter name or sorting not found')

        return SFSorting.deserialize(sorting_metadata)

    def __init__(self, identifier: str, recording_name: str, study_name: str,
                 sorter_name: str, sorter_version: str, sorting_parameters: Dict,
                 firings_url: str, recording_url: str, ground_truth_url: str, run_date: datetime.datetime):
        self.identifier = identifier
        self.recording_name = recording_name
        self.study_name = study_name
        self.sorter_name = sorter_name
        self.sorter_version = sorter_version
        self.sorting_parameters = sorting_parameters
        self.firings_url = firings_url
        self.recording_url = recording_url
        self.ground_truth_url = ground_truth_url
        self.run_date = run_date

    def get_sorting_extractor(self) -> sv.LabboxEphysSortingExtractor:
        return _build_sorting_extractor(firings_url=self.firings_url, sample_rate_hz=self._get_sample_rate_hz())

    def get_metrics(self, metric_names: Optional[List[str]] = None) -> pd.DataFrame:
        self._update_metric_data()
        metrics = pd.DataFrame(self._metric_data['quality_metric'])
        if metric_names:
            metrics.drop(columns=set(metrics.keys()) - set(metric_names), inplace=True)
        return metrics

    def get_agreement_scores(self) -> pd.DataFrame:
        self._update_metric_data()
        return pd.DataFrame(self._metric_data['ground_truth_comparison']['agreement_scores'])

    def get_best_match_12(self) -> pd.Series:
        self._update_metric_data()
        return pd.Series(self._metric_data['ground_truth_comparison']['best_match_12'])

    def get_best_match_21(self) -> pd.Series:
        self._update_metric_data()
        return pd.Series(self._metric_data['ground_truth_comparison']['best_match_21'])

    def _get_sample_rate_hz(self) -> int:
        recording_dict = kc.load_json(self.recording_url)
        return int(recording_dict['params']['samplerate'])

    def _update_metric_data(self):
        if self._metric_data is None:
            try:
                self._metric_data = _get_metric_metadata(
                    sorter_name=self.sorter_name,
                    recording_name=self.recording_name,
                    study_name=self.study_name
                )[0]
            except IndexError:
                raise ValueError('Metrics not found')
        
    @staticmethod
    def deserialize(sorting: Dict):
        return SFSorting(
            identifier=sorting['_id'], recording_name=sorting['recordingName'], study_name=sorting['studyName'],
            sorter_name=sorting['sorterName'], sorter_version=sorting['processorVersion'],
            sorting_parameters=sorting['sortingParameters'], firings_url=sorting['firings'],
            recording_url=sorting['recordingUri'], ground_truth_url=sorting['sortingTrueUri'],
            run_date=sorting['startTime'])


class SFRecording:

    @staticmethod
    def load(study_set_name: str, study_name: str, recording_name: str):
        return SFRecording.deserialize(_get_study_set_metadata(study_set_name, study_name, recording_name))

    def __init__(self, name: str, study_name: str, study_set_name: str, sample_rate_hz: int, num_channels: int,
                 duration_sec: float, num_true_units: int, spike_sign: int, recording_url: str, sorting_true_url: str):
        self.name = name
        self.study_name = study_name
        self.study_set_name = study_set_name
        self.sample_rate_hz = sample_rate_hz
        self.num_channels = num_channels
        self.duration_sec = duration_sec
        self.num_true_units = num_true_units
        self.spike_sign = spike_sign
        self.recording_url = recording_url
        self.sorting_true_url = sorting_true_url

    def get_recording_extractor(self, download: Optional[bool] = False) -> sv.LabboxEphysRecordingExtractor:
        return sv.LabboxEphysRecordingExtractor(self.recording_url, download=download)

    def get_ground_truth(self) -> sv.LabboxEphysSortingExtractor:
        return sv.LabboxEphysSortingExtractor(self.sorting_true_url)

    def get_sorting(self, sorter_name: str) -> SFSorting:
        return SFSorting.load(sorter_name=sorter_name, recording_name=self.name, study_name=self.study_name)

    def get_all_sortings(self) -> List[SFSorting]:
        return [SFSorting.deserialize(sorting_dict) for sorting_dict in
                _get_sorting_metadata(recording_name=self.name, study_name=self.study_name)]

    @staticmethod
    def deserialize(recording_set: Dict):
        return SFRecording(name=recording_set['name'], study_name=recording_set['studyName'],
                           study_set_name=recording_set['studySetName'], sample_rate_hz=recording_set['sampleRateHz'],
                           num_channels=recording_set['numChannels'], duration_sec=recording_set['durationSec'],
                           num_true_units=recording_set['numTrueUnits'], spike_sign=recording_set['spikeSign'],
                           recording_url=recording_set['recordingUri'], sorting_true_url=recording_set['sortingTrueUri'])


class SFStudy:

    @staticmethod
    def load(study_set_name: str, study_name: str):
        return SFStudy.deserialize(_get_study_set_metadata(study_set_name, study_name))

    def __init__(self, name: str, study_set_name: str, recordings: List[SFRecording]):
        self.name = name
        self.recording_sets = recordings
        self.study_set_name = study_set_name

    def get_recording_names(self) -> List[str]:
        return [recording.name for recording in self.recording_sets]

    def get_recording(self, name) -> SFRecording:
        try:
            return [recording_set for recording_set in self.recording_sets
                    if recording_set.name.lower() == name.lower()][0]
        except IndexError:
            raise ValueError('Recording set not found')

    @staticmethod
    def deserialize(study: Dict):
        return SFStudy(name=study['name'], study_set_name=study['studySetName'],
                       recordings=[SFRecording.deserialize(recording) for recording in study['recordings']],
                       )


class SFStudySet:

    @staticmethod
    def load(study_set_name: str):
        return SFStudySet.deserialize(_get_study_set_metadata(study_set_name=study_set_name))

    def __init__(self, name: str, studies: List[SFStudy]):
        self.name = name
        self._studies = studies

    def get_study_names(self) -> List[str]:
        return [study.name for study in self._studies]

    def get_study(self, name: str) -> SFStudy:
        try:
            return [study for study in self._studies if study.name.lower() == name.lower()][0]
        except IndexError:
            raise ValueError('Study not found')

    @staticmethod
    def deserialize(study_set: Dict):
        return SFStudySet(name=study_set['name'],
                          studies=[SFStudy.deserialize(study) for study in study_set['studies']],
                          )


def _get_recent_study_set_url() -> str:
    return requests.get(
        'https://raw.githubusercontent.com/flatironinstitute/spikeforest_recordings/master/recordings/studysets'
    ).text


def _get_study_set_metadata(
        study_set_name: str,
        study_name: Optional[str] = None,
        recording_name: Optional[str] = None
) -> Dict:
    if study_set_name is None:
        raise ValueError("Must provide a study_set name!")
    if recording_name is not None and study_name is None:
        raise ValueError("If a recording name is provided, a study name must also be provided.")

    metadata = [
        m for m in kc.load_json(_get_recent_study_set_url())['StudySets']
        if m['name'].lower() == study_set_name.lower()
    ][0]

    if study_name is not None:
        metadata = [m for m in metadata['studies'] if m['name'].lower() == study_name.lower()][0]
    if recording_name is not None:
        metadata = [m for m in metadata['recordings'] if m['name'].lower() == recording_name.lower()][0]

    return metadata


def _get_sorting_metadata(
        sorter_name: Optional[str] = None,
        recording_name: Optional[str] = None,
        study_name: Optional[str] = None
) -> List[Dict]:
    return _filter_sf_metadata(
        metadata_url=SF_SORTING_URI,
        sorter_name=sorter_name,
        recording_name=recording_name,
        study_name=study_name
    )


def _get_metric_metadata(
        study_name: str,
        recording_name: str,
        sorter_name: str
) -> List[Dict]:
    return _filter_sf_metadata(
        metadata_url=SF_METRIC_URI,
        sorter_name=sorter_name,
        recording_name=recording_name,
        study_name=study_name
    )


def _filter_sf_metadata(
        metadata_url: str,
        sorter_name: Optional[str] = None,
        recording_name: Optional[str] = None,
        study_name: Optional[str] = None
) -> List[Dict]:
    metadata = kc.load_json(metadata_url)
    return [m for m in metadata if
            (sorter_name is None or m['sorterName'].lower() == sorter_name.lower())
            and (recording_name is None or m['recordingName'].lower() == recording_name.lower())
            and (study_name is None or m['studyName'].lower() == study_name.lower())]


def _build_sorting_extractor(firings_url: str, sample_rate_hz: int,
                             sorting_format: Optional[str] = 'mda') -> sv.LabboxEphysSortingExtractor:
    return sv.LabboxEphysSortingExtractor(
        {
            'sorting_format': sorting_format,
            'data': {
                'firings': firings_url,
                'samplerate': sample_rate_hz
            }
        }
    )
