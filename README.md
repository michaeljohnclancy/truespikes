# Truespikes

Current implementation interfaces with spikeforest recordings via kachery to allow for easy browsing and retrieval of
raw recordings, ground truth firings and the associated sorting results.

## Installation

- Clone this repo
- Create some virtual environment (Optional)
- pip install -r requirements.txt
- Download kachery-daemon either in this environment, or globally in the system
- Install kachery-daemon
    - Follow this guide from "Running the daemon":
        - https://github.com/kacheryhub/kachery-doc/blob/main/doc/kacheryhub-markdown/hostKacheryNode.md#running-the-daemon
    - Then register your node with the spikeforest channel by following this guide:
        - https://github.com/flatironinstitute/spikeforest/blob/main/doc/join-spikeforest-download-channel.md
 - Test this installation by running the notebook "data_retrieval_example.ipynb"
        
### The study sets currently available 
['HYBRID_JANELIA', 'LONG_DRIFT', 'LONG_STATIC', 'MANUAL_FRANKLAB', 'PAIRED_BOYDEN', 'PAIRED_CRCNS_HC1', 'PAIRED_ENGLISH', 'PAIRED_KAMPFF', 'PAIRED_MEA64C_YGER', 'PAIRED_MONOTRODE', 'SYNTH_BIONET', 'SYNTH_MAGLAND', 'SYNTH_MEAREC_NEURONEXUS', 'SYNTH_MEAREC_TETRODE', 'SYNTH_MONOTRODE', 'SYNTH_VISAPY']
