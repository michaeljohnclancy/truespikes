import sortingview as sv

recording_name = 'paired_kampff/2014_11_25_Pair_3_0'
recording_uri = 'sha1://a205f87cef8b7f86df7a09cddbc79a1fbe5df60f/2014_11_25_Pair_3_0.json'
sorting_uri = 'sha1://c656add63d85a17840980084a1ff1cdc662a2cd5/2014_11_25_Pair_3_0.firings_true.json'

recording = sv.LabboxEphysRecordingExtractor(recording_uri, download=False)
sorting_true = sv.LabboxEphysSortingExtractor(sorting_uri)

channel_ids = recording.get_channel_ids()
samplerate = recording.get_sampling_frequency()
num_timepoints = recording.get_num_frames()
print(f'{recording_name}')
print(f'Recording has {len(channel_ids)} channels and {num_timepoints} timepoints (samplerate: {samplerate})')

unit_ids = sorting_true.get_unit_ids()
spike_train = sorting_true.get_unit_spike_train(unit_id=unit_ids[0])
print(f'Unit {unit_ids[0]} has {len(spike_train)} events')