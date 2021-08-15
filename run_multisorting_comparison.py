import tqdm
from consts import *
from sf_utils import SFStudy, SFStudySet
from spikecomparison.multisortingcomparison import MultiSortingComparison, compare_multiple_sorters

SORTER_NAMES = list(sorter_name.lower() for sorter_name in SORTER_NAMES.keys())
SORTER_NAMES.pop(0)

comparisons = {}
for study_set in tqdm.tqdm(SFStudySet.get_all_available_study_sets()):
    for study in tqdm.tqdm(study_set.get_studies()):
        if study.name.lower() in [name.lower() for name in STATIC_TETRODE_STUDY_NAMES]:
            for recording in tqdm.tqdm(study.get_recordings()):
                sortings = recording.get_sortings()
                sorter_names = [sorting.sorter_name for sorting in sortings]
                sorting_extractors = [sorting.get_sorting_extractor() for sorting in sortings]
                print(len(sorting_extractors))
                comparisons[recording.name] = compare_multiple_sorters(
                    sorting_list=sorting_extractors, name_list=SORTER_NAMES
                )


for recording_name, multi_sorting in comparisons.items():
    multi_sorting.dump(save_folder='/home/mclancy/truespikes/multisorter_comparison')

