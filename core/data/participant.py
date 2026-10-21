# face2eeg/core/data/participant.py

"""
Load a single participant's (video, EEG, labels) triple.
"""
from .physio import load_participant_physiological_data
from .video  import load_participant_facial_video

def load_participant_data(participant_id, 
                          experiment_id, 
                          channel, 
                          scaler, 
                          remove_pretrial=True, 
                          resample=False, 
                          new_sample_rate=128,
                          resize=True,
                          frame_size=(32, 32), 
                          video_normalize=True, 
                          video_centering=True,
                          signal_normalize=True,
                          root_dir='.'):
    '''
    description:
    ------------
    Load selected participant phsiological data, face video and emotional ratings according
    to given experiment_id and channel_id
    parameters:
    ------------
    -param participant_id: (integer) a number between 1,22
    -param experiment_id: (integer) a number between [0:40) that shows selected experiment index
    -param channel: (integer) selected channel index
    -param scaler: (object) scaler object
    -param remove_pretrial: if true: remove 3s pre trial physiological data and concat all trials
    -param resample: (boolean) if true: resample signals based on given new_sample_rate
    -param new_sample_rate: (integer) represents the new sampling rate of physiological data
    -param resize: (boolean) if true: resize frames based on given height and width
    -param frame_size: (tuple) represents the number of pixel rows and columns in the image
    -param video_normalize: (boolean) mean zero normalizing of each frame then dividing by 255.0
    -param video_centering:(boolean) calculate the mean pixel value across the each frame channels, then subtract it from frame
    -param signal_normalize: (boolean) z-score normalizing of signal based on all signals mean and std
    -param root_dir: (str) root directory containing physiological data and facial videos

    return:
    ------------
    video: (numpy ndarray) participant facial video
    signals: (numpy ndarray) participand eeg signal
    labels: (numpy ndarray) participant emotional rating
    '''
    signals, emotional_ratings = load_participant_physiological_data(participant_id, 
                                                                     [experiment_id],
                                                                     [channel],
                                                                     remove_pretrial,
                                                                     resample,
                                                                     new_sample_rate,
                                                                     root_dir=root_dir)
    facial_video,_ = load_participant_facial_video(participant_id,
                                                   experiment_id,
                                                   resize,
                                                   frame_size,
                                                   video_normalize, 
                                                   video_centering, 
                                                   root_dir=root_dir)
    if signal_normalize:
        signals = scaler.transform(signals.reshape(-1,1)).reshape(-1)
    if facial_video is not None:
        return facial_video, signals, emotional_ratings
    else:
        return None, None, None