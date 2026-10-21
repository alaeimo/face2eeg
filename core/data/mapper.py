# face2eeg/core/data/mapper.py
"""
experiment_id → trial_id mapping from experiment2trial.csv.
"""
import pandas as pd
from .cache import _EXP2TRIAL_MAPPER_CACHE

def get_exp2trial_mapper(file_path):
    '''
    description:
    ------------
    Load experiment2trial.csv file to get experiment_id to trial_id mapper

    parameters:
    ------------
    -param file_path: experiment2trial.csv path

    return:
    ------------
    trials[XX] : shape : (32, 40) : 40 trials of 32 participants: a list of YYs for face videos of participant number XX in sXX/sXX_trial_YY.avi, XX:Participant_id,YY: trial_id
    labels[XX] : shape: (32, 40, 4): Valence,Arousal,Dominance,Liking of 40 trials of 32 participants:
    '''
    if file_path in _EXP2TRIAL_MAPPER_CACHE:
        return _EXP2TRIAL_MAPPER_CACHE[file_path]
    df = pd.read_csv(file_path) #read csv file as a pandas DataFrame
    trials = df['Trial'] #extract trial numbers
    labels = df[['Valence','Arousal','Dominance','Liking']] #extract emotional ratings of each trial
    trials = trials.values #convert trials to the numpy array
    labels = labels.values #convert emotional ratings to the numpy array
    trials = trials.reshape((32,int(1280/32))) #reshape array of trial numbers to (32,40)
    labels = labels.reshape((32,int(1280/32),4))  #reshape array of emotional ratings to (32,40,4)
    _EXP2TRIAL_MAPPER_CACHE[file_path] = (trials, labels)
    return trials,labels
