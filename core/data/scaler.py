# face2eeg/core/data/scaler.py

"""
Per-fold scaler fitting (z-score / minmax).
"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from .physio import load_participant_physiological_data

def fit_scaler(participants, 
               experiments, 
               channels, 
               method, 
               feature_range=(0,1), 
               remove_pretrial=True, 
               resample=False, 
               new_sample_rate=128,
               root_dir='.'):
    '''
    description:
    ------------
    create a scaler object and fit it based on given data and return it

    parameters:
    ------------
    -param participants: (tuple/list) list of participant ids
    -param experiments: (tuple/list) list of experiment ids
    -param channels: (tuple) EEG channels id
    -param method: (str) "z-score" or "minmax"
    -param feature_range: (tuple)-> (0,1) transform features by scaling each feature to a given range when using MinMaxScaler
    -param remove_pretrial: if true: remove 3s pre trial physiological data and concat all trials
    -param resample: (boolean) if true: resample signals based on given new_sample_rate
    -param new_sample_rate: (integer) represents the new sampling rate of physiological data
    -param root_dir: (str) root directory of physiological data

    return:
    ------------
    (object) scaler(StandardScaler or MinMaxScaler)
    '''

    #create a scaler object according to given method name
    scaler = None
    if method =='z-score':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range)

    #load each signal of given participants and experiments and partial_fit scaler according to the
    for participant_id in participants:
        signals, _ = load_participant_physiological_data(participant_id,
                                                         experiments,
                                                         channels,
                                                         remove_pretrial,
                                                         resample,
                                                         new_sample_rate, 
                                                         root_dir=root_dir)
        for i in range(len(experiments)):
            signal = signals[i].T
            scaler.partial_fit(signal)
    return scaler