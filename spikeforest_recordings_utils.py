from pathlib import Path
import sortingview as sv
import git

spikeforest_recordings = Path('spikeforest_recordings/recordings')

git.cmd.Git(spikeforest_recordings).pull()


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
