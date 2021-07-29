from pathlib import Path
from typing import List, Dict

import sortingview as sv
import kachery_client as kc
import git

spikeforest_recordings = Path('spikeforest_recordings/recordings')

git.cmd.Git(spikeforest_recordings).pull()

STUDY_SET_METADATA_URL = 'sha1://f728d5bf1118a8c6e2dfee7c99efb0256246d1d3/studysets.json'


class RecordingSet:
    def __init__(self, name: str, study_name: str, study_set_name: str, sample_rate_hz: int, num_channels: int, duration_sec: float, num_true_units: int, spike_sign: int, recording_uri: str, sorting_true_uri: str):
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

    def get_recording(self, download: bool) -> sv.LabboxEphysRecordingExtractor:
        return sv.LabboxEphysRecordingExtractor(self.recording_uri, download=download)

    def get_ground_truth(self) -> sv.LabboxEphysSortingExtractor:
        return sv.LabboxEphysSortingExtractor(self.sorting_true_uri)


class Study:
    def __init__(self, name: str, study_set_name: str, recordings: List[Dict], self_reference: str):
        self._name = name
        self._recordings = recordings
        self._study_set_name = study_set_name
        self._self_reference = self_reference

    def get_recording_names(self):
        return [recording['name'] for recording in self._recordings]

    def get_recording(self, name) -> RecordingSet:
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

    def __init__(self, name: str, studies: List[Dict], self_reference: str):
        self.name = name
        self._studies = studies
        self._self_reference = self_reference

    def get_study_names(self) -> List[str]:
        return [study['name'] for study in self._studies]

    def get_study(self, name:str) -> Study:
        return [Study(name=study['name'], study_set_name=study['studySetName'], recordings=study['recordings'], self_reference=study['self_reference']) for study in self._studies if study['name'] == name][0]


def get_study_set_metadata():
    return kc.load_json(STUDY_SET_METADATA_URL)


def get_study_set_info(study_set_name: str):
    return [study_set_metadata for study_set_metadata in get_study_set_metadata()
            if study_set_metadata['name'].lower() == study_set_name.lower()][0]


def get_kachery_recording_gt_path_pairs_for_study_set(
        study_set_name: str,
        sub_study_name: str = None,
        recording_name: str = None
):
    if recording_name is not None and sub_study_name is None:
        raise ValueError("If a recording name is provided, a substudy must also be provided.")

    recording_gt_pairs = {}
    for sub_study_set in (spikeforest_recordings / study_set_name).iterdir():
        if sub_study_set.is_dir() and (sub_study_name is None or sub_study_name == sub_study_set.stem):
            for gt_metadata in sub_study_set.glob('*.firings_true.json'):
                recording_gt_pair_name = gt_metadata.name.split('.')[0]
                if recording_name is None or recording_gt_pair_name == recording_name:
                    recording_gt_pairs[sub_study_set.stem + '_' + recording_gt_pair_name] =\
                        (sub_study_set/(recording_gt_pair_name + '.json'), gt_metadata)
    return recording_gt_pairs


def get_labbox_ephys_extractors(
        study_set_name: str,
        sub_study_name: str = None,
        recording_name: str = None,
        download_full_recording: bool = False
):
    recording_gt_path_pairs = get_kachery_recording_gt_path_pairs_for_study_set(
        study_set_name, sub_study_name, recording_name)

    return {k: (
        sv.LabboxEphysRecordingExtractor(str(v[0]), download=download_full_recording),
        sv.LabboxEphysSortingExtractor(str(v[1])))
            for k, v in recording_gt_path_pairs.items()}
