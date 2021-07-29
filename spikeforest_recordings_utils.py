from pathlib import Path
from typing import List, Dict, Optional

import sortingview as sv
import kachery_client as kc

STUDY_SET_METADATA_URL = 'sha1://f728d5bf1118a8c6e2dfee7c99efb0256246d1d3/studysets.json'


def _get_sf_metadata(study_set_name: Optional[str] = None,
                     study_name: Optional[str] = None,
                     recording_name: Optional[str] = None):
    if study_set_name is None:
        raise ValueError("Must provide a study_set name!")
    if recording_name is not None and study_name is None:
        raise ValueError("If a recording name is provided, a substudy must also be provided.")

    metadata = kc.load_json(STUDY_SET_METADATA_URL)['StudySets']
    if study_set_name is not None:
        metadata = [m for m in metadata if m['name'].lower() == study_set_name.lower()][0]
    if study_name is not None:
        metadata = [m for m in metadata['studies'] if m['name'].lower() == study_name.lower()][0]
    if recording_name is not None:
        metadata = [m for m in metadata['recordings'] if m['name'].lower() == recording_name.lower()][0]

    return metadata


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


class Study:

    @staticmethod
    def load(study_set_name: str, study_name: str):
        study_metadata = _get_sf_metadata(study_set_name, study_name)
        return Study(name=study_metadata['name'],
                     recordings=study_metadata['recordings'],
                     study_set_name=study_metadata['studySetName'],
                     self_reference=study_metadata['self_reference']
                     )

    def __init__(self, name: str, study_set_name: str, recordings: List[Dict], self_reference: str):
        self._name = name
        self._recordings = recordings
        self._study_set_name = study_set_name
        self._self_reference = self_reference

    def get_recording_names(self):
        return [recording['name'] for recording in self._recordings]

    def get_recording_set(self, name) -> RecordingSet:
        return [RecordingSet(name=recording['name'], study_name=recording['studyName'],
                             study_set_name=recording['studySetName'], sample_rate_hz=recording['sampleRateHz'],
                             num_channels=recording['numChannels'], duration_sec=recording['durationSec'],
                             num_true_units=recording['numTrueUnits'], spike_sign=recording['spikeSign'],
                             recording_uri=recording['recordingUri'], sorting_true_uri=recording['sortingTrueUri'])
                for recording in self._recordings if recording['name'].lower() == name.lower()][0]


class StudySet:
    _studies: List[Dict]
    description: str
    self_reference: str

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
        return [Study(name=study['name'], study_set_name=study['studySetName'], recordings=study['recordings'],
                      self_reference=study['self_reference']) for study in self._studies if study['name'] == name][0]