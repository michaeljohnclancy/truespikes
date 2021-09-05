import numpy as np

STUDY_NAMES = ['hybrid_static_tetrode', 'hybrid_static_siprobe',
               'LONG_STATIC_1200s_8c', 'LONG_STATIC_600s_8c', 'LONG_STATIC_300s_16c', 'LONG_STATIC_4800s_16c',
               'LONG_STATIC_300s_8c', 'LONG_STATIC_2400s_8c', 'LONG_STATIC_2400s_16c', 'LONG_STATIC_600s_16c',
               'LONG_STATIC_1200s_16c', 'LONG_STATIC_4800s_8c', 'synth_magland_noise20_K20_C8',
               'synth_magland_noise10_K10_C4', 'synth_magland_noise10_K10_C8', 'synth_magland_noise20_K10_C4',
               'synth_magland_noise20_K20_C4', 'synth_magland_noise20_K10_C8', 'synth_magland_noise10_K20_C8',
               'synth_magland_noise10_K20_C4', 'synth_mearec_tetrode_noise10_K20_C4', 'synth_mearec_tetrode_noise10_K10_C4',
               'synth_mearec_tetrode_noise20_K10_C4', 'synth_mearec_tetrode_noise20_K20_C4']

STATIC_SIPROBE_STUDY_NAMES = ['hybrid_static_siprobe', 'LONG_STATIC_1200s_8c', 'LONG_STATIC_600s_8c', 'LONG_STATIC_300s_16c',
                              'LONG_STATIC_4800s_16c', 'LONG_STATIC_300s_8c', 'LONG_STATIC_2400s_8c', 'LONG_STATIC_2400s_16c',
                              'LONG_STATIC_600s_16c', 'LONG_STATIC_1200s_16c', 'LONG_STATIC_4800s_8c',]

STATIC_TETRODE_STUDY_NAMES = ['hybrid_static_tetrode', 'synth_magland_noise20_K20_C8', 'synth_magland_noise10_K10_C4',
                              'synth_magland_noise10_K10_C8', 'synth_magland_noise20_K10_C4', 'synth_magland_noise20_K20_C4',
                              'synth_magland_noise20_K10_C8', 'synth_magland_noise10_K20_C8', 'synth_magland_noise10_K20_C4',
                              'synth_mearec_tetrode_noise10_K20_C4', 'synth_mearec_tetrode_noise10_K10_C4',
                              'synth_mearec_tetrode_noise20_K10_C4', 'synth_mearec_tetrode_noise20_K20_C4']

METRIC_NAMES = ["firing_rate", "presence_ratio", "isi_violation",
                 "amplitude_cutoff", "snr", "silhouette_score",
                "isolation_distance", "l_ratio", "nn_hit_rate",
                "nn_miss_rate", "d_prime"
                ]

GAUSSIAN_METRICS = ["firing_rate", "snr", "isolation_distance",
                    "silhouette_score", "nn_hit_rate",
                    "nn_miss_rate", "d_prime"
                    ]

NONGAUSSIAN_METRICS = ["presence_ratio", "isi_violation", "amplitude_cutoff", "l_ratio"]

# METRIC_NAMES = ["firing_rate", "presence_ratio", "isi_violation",
#                 "amplitude_cutoff", "snr", "max_drift", "cumulative_drift",
#                 "silhouette_score", "isolation_distance", "l_ratio",
#                 "nn_hit_rate", "nn_miss_rate", "d_prime"]

# SORTER_NAMES = {'herdingspikes2': 0, 'ironclust': 1, 'jrclust': 2,
#                 'kilosort': 3, 'kilosort2': 4, 'klusta': 5, 'mountainsort4': 6,
#                 'spykingcircus': 7, 'tridesclous': 8}

# SORTER_NAMES = {'ironclust': 0, 'jrclust': 1,
                # 'kilosort2': 2, 'klusta': 3, 'mountainsort4': 4,
                # 'spykingcircus': 5, 'tridesclous': 6}
SORTER_NAMES = ['ironclust', 'jrclust', 'kilosort2', 'mountainsort4', 'spykingcircus', 'tridesclous']

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)
