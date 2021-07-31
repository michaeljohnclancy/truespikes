import requests
from typing import List, Dict, Optional

import sortingview as sv
import kachery_client as kc


SF_SORTING_URI = 'sha1://52f24579bb2af1557ce360ed5ccc68e480928285/sortings.json'
SF_METRIC_URI = 'sha1://b3444629251cafda919af535f0e9837279151c6e/spikeforest-full-gt-qm.json?' \
                'manifest=cf73c99d06c11e328e635e14dc24b8db7372db3d'


class RecordingSet:

    @staticmethod
    def load(study_set_name: str, study_name: str, recording_name: str):
        return RecordingSet.deserialize(_get_sf_metadata(study_set_name, study_name, recording_name))

    def __init__(self, name: str, study_name: str, study_set_name: str, sample_rate_hz: int, num_channels: int,
                 duration_sec: float, num_true_units: int, spike_sign: int, recording_uri: str, sorting_true_uri: str):
        self.name = name
        self.study_name = study_name
        self.study_set_name = study_set_name
        self.sample_rate_hz = sample_rate_hz
        self.num_channels = num_channels
        self.duration_sec = duration_sec
        self.num_true_units = num_true_units
        self.spike_sign = spike_sign
        self.recording_uri = recording_uri
        self.sorting_true_uri = sorting_true_uri

    def get_recording(self, download: Optional[bool] = False) -> sv.LabboxEphysRecordingExtractor:
        return sv.LabboxEphysRecordingExtractor(self.recording_uri, download=download)

    def get_ground_truth(self) -> sv.LabboxEphysSortingExtractor:
        return sv.LabboxEphysSortingExtractor(self.sorting_true_uri)

    def get_sorting(self, sorter_name: str) -> sv.LabboxEphysSortingExtractor:
        try:
            sorting_metadata = _get_sorting_metadata(
                sorter_name=sorter_name, recording_name=self.name, study_name=self.study_name)[0]
        except IndexError:
            raise ValueError('Invalid sorter name or sorting not found.')
        return self._build_sorting(sorting_metadata['firings'])

    def get_all_sortings(self) -> Dict[str, sv.LabboxEphysSortingExtractor]:
        sorting_metadata = _get_sorting_metadata(
            recording_name=self.name, study_name=self.study_name)
        return {
            sorting['sorterName']: self._build_sorting(firing_url=sorting['firings'])
            for sorting in sorting_metadata
        }

    def _build_sorting(self, firing_url: str) -> sv.LabboxEphysSortingExtractor:
        return sv.LabboxEphysSortingExtractor(
            _wrap_sorting_uri(firings_url=firing_url,
                              sample_rate=self.sample_rate_hz,
                              sorting_format='mda')
        )

    @staticmethod
    def deserialize(recording_set: Dict):
        return RecordingSet(name=recording_set['name'], study_name=recording_set['studyName'],
                            study_set_name=recording_set['studySetName'], sample_rate_hz=recording_set['sampleRateHz'],
                            num_channels=recording_set['numChannels'], duration_sec=recording_set['durationSec'],
                            num_true_units=recording_set['numTrueUnits'], spike_sign=recording_set['spikeSign'],
                            recording_uri=recording_set['recordingUri'], sorting_true_uri=recording_set['sortingTrueUri'])


class Study:

    @staticmethod
    def load(study_set_name: str, study_name: str):
        return Study.deserialize(_get_sf_metadata(study_set_name, study_name))

    def __init__(self, name: str, study_set_name: str, recordings: List[RecordingSet]):
        self.name = name
        self.recording_sets = recordings
        self.study_set_name = study_set_name

    def get_recording_names(self) -> List[str]:
        return [recording.name for recording in self.recording_sets]

    def get_recording_set(self, name) -> RecordingSet:
        try:
            return [recording_set for recording_set in self.recording_sets
                    if recording_set.name.lower() == name.lower()][0]
        except IndexError:
            raise ValueError('Recording set not found')

    def get_recording_sets(self) -> List[RecordingSet]:
        return self.recording_sets

    @staticmethod
    def deserialize(study: Dict):
        return Study(name=study['name'], study_set_name=study['studySetName'],
                     recordings=[RecordingSet.deserialize(recording) for recording in study['recordings']],
                     )


class StudySet:

    @staticmethod
    def load(study_set_name: str):
        return StudySet.deserialize(_get_sf_metadata(study_set_name=study_set_name))

    def __init__(self, name: str, studies: List[Study]):
        self.name = name
        self._studies = studies

    def get_study_names(self) -> List[str]:
        return [study.name for study in self._studies]

    def get_study(self, name: str) -> Study:
        try:
            return [study for study in self._studies if study.name.lower() == name.lower()][0]
        except IndexError:
            raise ValueError('Study not found')

    @staticmethod
    def deserialize(study_set: Dict):
        return StudySet(name=study_set['name'],
                        studies=[Study.deserialize(study) for study in study_set['studies']],
                        )


def _get_recent_study_set_url() -> str:
    return requests.get(
        'https://raw.githubusercontent.com/flatironinstitute/spikeforest_recordings/master/recordings/studysets'
    ).text


def _get_sf_metadata(study_set_name: str,
                     study_name: Optional[str] = None,
                     recording_name: Optional[str] = None) -> Dict:
    if study_set_name is None:
        raise ValueError("Must provide a study_set name!")
    if recording_name is not None and study_name is None:
        raise ValueError("If a recording name is provided, a substudy must also be provided.")

    metadata = [
        m for m in kc.load_json(_get_recent_study_set_url())['StudySets']
        if m['name'].lower() == study_set_name.lower()
    ][0]

    if study_name is not None:
        metadata = [m for m in metadata['studies'] if m['name'].lower() == study_name.lower()][0]
    if recording_name is not None:
        metadata = [m for m in metadata['recordings'] if m['name'].lower() == recording_name.lower()][0]

    return metadata


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


def _wrap_sorting_uri(firings_url: str, sample_rate: int, sorting_format: str) -> Dict:
    return {
        'sorting_format': sorting_format,
        'data': {
            'firings': firings_url,
            'samplerate': sample_rate
        }
    }
