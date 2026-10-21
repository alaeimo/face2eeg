# face2eeg/core/data/physio.py

"""
Load physiological data (.dat) with caching.
"""
import os
import pickle
import numpy as np
from scipy import signal
from .cache import _PHYSIO_CACHE

def _load_raw_physiological(participant_id, root_dir, data_subdir):
    """Load the raw .dat pickle exactly once per (participant, path)."""
    key = (participant_id, root_dir, data_subdir)
    if key in _PHYSIO_CACHE:
        return _PHYSIO_CACHE[key]

    if participant_id < 10:
        participant_id = "0" + str(participant_id)
    else:
        participant_id = str(participant_id)

    data_path = os.path.join(root_dir, data_subdir, f's{participant_id}.dat')
    with open(data_path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')

    _PHYSIO_CACHE[key] = raw
    return raw

def load_participant_physiological_data(participant_id, 
                                        experiments=None, 
                                        channels=None, 
                                        remove_pretrial=True, 
                                        resample=False, 
                                        new_sample_rate=128,
                                        root_dir='.', 
                                        data_subdir='Physiological_data'):
    '''
    description:
    ------------
    Read a physiological data of a participant
    parameters:
    ------------
    -param participant_id: (integer) a number between 1,22
    -param experiments: (tuple/None) a tuple of numbers between [0:40) for selected experiments or None for selecting all experiments
    -param channels: (tuple/None) if None return all channels data else return selected channels data
    -param remove_pretrial: if true: remove 3s pre trial physiological data and concat all trials
    -param resample: (boolean) if true: resample signals based on given new_sample_rate
    -param new_sample_rate: (integer) represents the new sampling rate of physiological data
    -param root_dir: (str) root directory containing physiological data
    -param data_subdir: (str) subdirectory name containing physiological .dat files
    
    return:
    ------------
    signals: (numpy array) array contents (trial,channel,signal_data):shape e.g. (40,40,8064)
    labels: (numpy array)  array contents (trial,label (valence, arousal, dominance, liking)):shape e.g. (40,4)
    '''

    physiological_data = _load_raw_physiological(participant_id, root_dir, data_subdir)
    signals = physiological_data['data']
    labels  = physiological_data['labels']

    signals_length = 63 #63 seconds with 3s pre_trial
    if remove_pretrial:
        #remove 3s pre_trial and get new data with shape of (40,40,7680)
        signals = np.array([signals[i,:,128*3:] for i in range(signals.shape[0])])
        signals_length = 60

    if experiments is not None:
        signals = signals[experiments,:,:] #get signals of given experiment ids
    if channels is not None:
        signals = signals[:,channels,:]  #get signals of given channels

    if resample:
        new_samples = new_sample_rate * signals_length
        signals = np.array([signal.resample(signals[i],new_samples,axis=1) for i in range(signals.shape[0])])

    if experiments is None:
        return signals, labels
    
    return signals, labels[experiments,:]