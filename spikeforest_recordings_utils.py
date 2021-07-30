from typing import List, Dict, Optional

import sortingview as sv
import kachery_client as kc

import requests

SF_SORTING_URI = 'sha1://52f24579bb2af1557ce360ed5ccc68e480928285/sortings.json'


class RecordingSet:

    @staticmethod
    def load(study_set_name: str, study_name: str, recording_name: str):
        recording_set_metadata = _get_sf_metadata(study_set_name, study_name, recording_name)
        return RecordingSet(name=recording_set_metadata['name'],
                            study_name=recording_set_metadata['studyName'],
                            study_set_name=recording_set_metadata['studySetName'],
                            sample_rate_hz=recording_set_metadata['sampleRateHz'],
                            num_channels=recording_set_metadata['numChannels'],
                            duration_sec=recording_set_metadata['durationSec'],
                            num_true_units=recording_set_metadata['numTrueUnits'],
                            spike_sign=recording_set_metadata['spikeSign'],
                            recording_uri=recording_set_metadata['recordingUri'],
                            sorting_true_uri=recording_set_metadata['sortingTrueUri']
                            )

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


class Study:

    @staticmethod
    def load(study_set_name: str, study_name: str):
        study_metadata = _get_sf_metadata(study_set_name, study_name)
        return _parse_study(study_metadata)

    def __init__(self, name: str, study_set_name: str, recordings: List[Dict], self_reference: str):
        self._name = name
        self._recording_sets = recordings
        self._study_set_name = study_set_name
        self._self_reference = self_reference

    def get_recording_names(self) -> List[str]:
        return [recording['name'] for recording in self._recording_sets]

    def get_recording_set(self, name) -> RecordingSet:
        try:
            return [_parse_recording_set(recording)
                    for recording in self._recording_sets if recording['name'].lower() == name.lower()][0]
        except IndexError:
            raise ValueError('Recording set not found')

    def get_recording_sets(self) -> List[RecordingSet]:
        return [_parse_recording_set(recording_set) for recording_set in self._recording_sets]


class StudySet:

    @staticmethod
    def load(study_set_name: str):
        study_set_metadata = _get_sf_metadata(study_set_name=study_set_name)
        return StudySet(name=study_set_metadata['name'],
                        studies=study_set_metadata['studies'],
                        self_reference=study_set_metadata['self_reference']
                        )

    def __init__(self, name: str, studies: List[Dict], self_reference: str):
        self.name = name
        self._studies = studies
        self._self_reference = self_reference

    def get_study_names(self) -> List[str]:
        return [study['name'] for study in self._studies]

    def get_study(self, name: str) -> Study:
        try:
            return [_parse_study(study)
                    for study in self._studies if study['name'].lower() == name.lower()][0]
        except IndexError:
            raise ValueError('Study not found')


def _get_recent_study_set_url() -> str:
    return requests.get(
        'https://raw.githubusercontent.com/flatironinstitute/spikeforest_recordings/master/recordings/studysets'
    ).text


def _get_sf_metadata(study_set_name: Optional[str] = None,
                     study_name: Optional[str] = None,
                     recording_name: Optional[str] = None) -> Dict:
    if study_set_name is None:
        raise ValueError("Must provide a study_set name!")
    if recording_name is not None and study_name is None:
        raise ValueError("If a recording name is provided, a substudy must also be provided.")

    metadata = kc.load_json(_get_recent_study_set_url())['StudySets']
    if study_set_name is not None:
        metadata = [m for m in metadata if m['name'].lower() == study_set_name.lower()][0]
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
    sortings = kc.load_json(SF_SORTING_URI)
    return [sorting for sorting in sortings if
            (sorter_name is None or sorting['sorterName'].lower() == sorter_name.lower())
            and (recording_name is None or sorting['recordingName'].lower() == recording_name.lower())
            and (study_name is None or sorting['studyName'].lower() == study_name.lower())]


def _wrap_sorting_uri(firings_url: str, sample_rate: int, sorting_format: str):
    return {
        'sorting_format': sorting_format,
        'data': {
            'firings': firings_url,
            'samplerate': sample_rate
        }
    }


def _parse_recording_set(recording: Dict) -> RecordingSet:
    return RecordingSet(name=recording['name'], study_name=recording['studyName'],
                        study_set_name=recording['studySetName'], sample_rate_hz=recording['sampleRateHz'],
                        num_channels=recording['numChannels'], duration_sec=recording['durationSec'],
                        num_true_units=recording['numTrueUnits'], spike_sign=recording['spikeSign'],
                        recording_uri=recording['recordingUri'], sorting_true_uri=recording['sortingTrueUri'])


def _parse_study(study: Dict) -> Study:
    return Study(name=study['name'], study_set_name=study['studySetName'], recordings=study['recordings'],
                 self_reference=study['self_reference'])
