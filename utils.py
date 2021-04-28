from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd

LOG_PATH = Path('logs')
LOG_PATH.mkdir(exist_ok=True)


def parse_sf_results(
        sf_data, sorter_names: Optional[List[str]] = None, study_names: Optional[List[str]] = None,
        exclude_sorter_names: Optional[List[str]] = None, exclude_study_names: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None, exclude_metric_names: Optional[List[str]] = None
):
    """Parses the SpikeForest json formatted results, and can be filtered by sorter name, study name and metric."""
    if sorter_names is not None and exclude_sorter_names is not None:
        raise ValueError("Can't provide list of sorters to include and exclude!")
    elif study_names is not None and exclude_study_names is not None:
        raise ValueError("Can't provide list of studies to include and exclude!")
    elif metric_names is not None and exclude_metric_names is not None:
        raise ValueError("Can't provide list of metrics to include and exclude!")

    with open(LOG_PATH / f'parse_sf_results-{datetime.now().strftime("%Y%m%d%H%M%S")}.log', 'w') as logfile:
        results_df = None
        for entry in sf_data:
            if ((study_names is None and exclude_study_names is None)
                or (study_names is not None and entry['studyName'] in study_names)
                or (exclude_study_names is not None and entry['studyName'] not in exclude_study_names)) \
                    and ((sorter_names is None and exclude_sorter_names is None)
                         or (sorter_names is not None and entry['sorterName'] in sorter_names)
                         or (exclude_sorter_names is not None and entry['sorterName'] not in exclude_sorter_names)
            ):
                try:
                    entry_df = pd.DataFrame(entry["quality_metric"])
                    entry_df["fp"] = np.array(entry['ground_truth_comparison']["best_match_21"]) == -1
                    if results_df is None:
                        results_df = entry_df
                    else:
                        results_df = results_df.append(entry_df)
                except Exception as e:
                    logfile.write(f"{entry['studyName']} - {entry['recordingName']} - {entry['sorterName']} - {str(e)} \n")

    if metric_names is not None:
        results_df = results_df[metric_names + ['fp']]
    elif exclude_metric_names is not None:
        results_df.drop(columns=exclude_metric_names, inplace=True)

    results_df.dropna(inplace=True)
    return results_df
