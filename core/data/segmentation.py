# face2eeg/core/data/segmentation.py

"""
Segment video and EEG into fixed-length windows.
"""
import numpy as np
from scipy import signal
from sklearn.utils import shuffle
from .participant import load_participant_data, load_participant_physiological_data
from ..config import DEAP_EEG_CHANNELS, FREQ_BANDS

def segment_video_and_signal(video,
                    signals,
                    video_fr,
                    signal_fs,
                    time_window=1,
                    overlap=False,
                    overlap_ratio=0,
                    y_type='RawEEG',
                    freqs_range=(4,45),
                    freqs_bands=None,
                    shuffle_output=False):
    '''
    description:
    ------------
    Create segmented video and signal based on given conditions
    parameters:
    ------------
    -param video: (numpy ndarray) facial video
    -param signals: (numpy ndarray) EEG signal
    -param video_fr: (integer) frame_rate of given video
    -param signal_fs: (integer) sampeling frequency of given signal
    -param time_window: (integer/float)length of segmentation window (in minutes), default is 1
    -param overlap: (boolean) if True: include overlapping in data segmentation, default is False
    -param overlap_ratio: (float) the overlap ratio of windows, default is 0
    -param y_type: (str) y_out type ('RawEEG'/'PSD'/'FBP') default is 'RawEEG'
    -param freqs_range: (tuple) used when y_type='PSD', range of frequency for getting PSD of that range, default is (4,45)
    -param freqs_bands: (dictionary/None) used when y_type='FBP'; if None, uses `config.FREQ_BANDS`
    -param shuffle_output: (boolean) random shuffling the outputs
    
    return:
    ------------
    X : segmented video segments  (n_segments, channels, v_window, H, W)
    y : segmented signal segments (n_segments, s_window)  or  (n_segments, n_freqs)  for PSD/FBP
    '''
    if freqs_bands is None:
        freqs_bands = FREQ_BANDS

    X_data = np.moveaxis(video,-1,0) #new shape->(channels, length, height, width)
    y_data = signals
    v_window = int(time_window * video_fr)
    s_window = int(time_window * signal_fs)
    v_nop = int(v_window * overlap_ratio)
    s_nop = int(s_window * overlap_ratio)

    if overlap:

        X = np.array([X_data[:,i-v_nop:i+v_nop,:,:] for i in range(v_nop,X_data.shape[1],v_nop)])

        if y_type == 'RawEEG':
            y = np.array([y_data[i-s_nop:i+s_nop] for i in range(s_nop,y_data.shape[0],s_nop)])

        elif y_type == 'PSD':
            freqs,psd = signal.welch(y_data,fs=signal_fs, window='hann',nperseg=s_window,nfft=512)
            #select frequency range indices between lower_freq-upper_freq
            indices = np.logical_and(freqs>=freqs_range[0],freqs<=freqs_range[1])
            y = np.array([signal.welch(y_data[i-s_nop:i+s_nop],fs=signal_fs, window='hann',nperseg=s_window,nfft=512)[1][indices] for i in range(s_nop,y_data.shape[0],s_nop)])
        elif y_type =='FBP':
            y = []
            for i in range(s_nop,y_data.shape[0],s_nop):
                freqs,psd = signal.welch(y_data[i-s_nop:i+s_nop],fs=signal_fs, window='hann',nperseg=s_window,nfft=512)
                FBP = []
                for band_name in freqs_bands.keys():
                    #select frequency range indices between lower_freq-upper_freq
                    indices = np.logical_and(freqs>=freqs_bands[band_name][0],freqs<=freqs_bands[band_name][1])
                    FBP.append(np.max(psd[indices]))
                y.append(FBP)
            y = np.array(y)

    else:

        X = np.array([X_data[:,i:i+v_window,:,:] for i in range(0,X_data.shape[1],v_window)])

        if y_type == 'RawEEG':
            y = y_data.reshape(-1,s_window)

        elif y_type == 'PSD':
            freqs,psd = signal.welch(y_data,fs=signal_fs, window='hann',nperseg=s_window,nfft=512)
            #select frequency range indices between lower_freq-upper_freq
            indices = np.logical_and(freqs>=freqs_range[0],freqs<=freqs_range[1])
            y = np.array([signal.welch(y_data[i:i+s_window],fs=signal_fs,window='hann',nperseg=s_window,nfft=512)[1][indices] for i in range(0,y_data.shape[0],s_window)])

        elif y_type =='FBP':
            y = []
            for i in range(0,y_data.shape[0],s_window):
                freqs,psd = signal.welch(y_data[i:i+s_window],fs=signal_fs, window='hann',nperseg=s_window,nfft=512)
                FBP = []
                for band_name in freqs_bands.keys():
                    #select frequency range indices between lower_freq-upper_freq
                    indices = np.logical_and(freqs>=freqs_bands[band_name][0],freqs<=freqs_bands[band_name][1])
                    FBP.append(np.mean(psd[indices]))
                y.append(FBP)
            y = np.array(y)
    if shuffle_output:
      return shuffle(X,y)
    else:
      return X, y

def load_participant_segments(
    participant_id,
    experiments,
    scaler,
    channel_name,
    time_window=1,
    overlap=False,
    overlap_ratio=0.0,
    shuffle_output=False,
    signal_normalize=True,
    frame_size=(32, 32),
    video_fr=50,
    signal_fs=128,
    y_type='RawEEG',
    deap_eeg_channels=None,
    shuffle_pairing=False,     
    shuffle_seed=None,    
    root_dir=".",     
):
    """
    description:
    ------------
    Load and segment video + EEG for a participant across given experiments.

    parameters:
    ------------
    -param participant_id: (int) participant id
    -param experiments: (list/tuple) list of experiment ids
    -param scaler: (object) fitted scaler for EEG normalization
    -param channel_name: (str) key in `deap_eeg_channels` dict
    -param time_window: (int/float) segmentation window in seconds
    -param overlap: (bool) whether to use overlapping windows
    -param overlap_ratio: (float) overlap ratio when overlap=True
    -param shuffle_output: (bool) shuffle X/y before returning
    -param signal_normalize: (bool) apply scaler to signals
    -param frame_size: (tuple) video frame size (h, w)
    -param video_fr: (int) video frame rate
    -param signal_fs: (int) EEG sampling frequency
    -param y_type: (str) 'RawEEG' / 'PSD' / 'FBP'
    -deap_eeg_channels: (dict/None) eeg channel name -> index; if None, uses `config.DEAP_EEG_CHANNELS`
    -param shuffle_pairing: (bool) if True, the EEG list is permuted
        before concatenation, breaking the video-EEG trial correspondence.
        Videos keep their original order; signals are shuffled.
    -param shuffle_seed: (int/None) RNG seed for the permutation.
    -param root_dir: (str) root directory containing physiological data and facial videos
    return:
    ------------
    X : segmented video
    y : segmented signal
    """
    if deap_eeg_channels is None:
        deap_eeg_channels = DEAP_EEG_CHANNELS

    if channel_name not in deap_eeg_channels:
        raise KeyError(f"Channel '{channel_name}' not found.")
    
    # ---- 1. Load all trials (video and EEG paired) ----
    videos, signals = [], []
    for experiment_id in experiments:
        vid, sig, _ = load_participant_data(
            participant_id,
            experiment_id=experiment_id,
            channel=deap_eeg_channels[channel_name],
            scaler=scaler,
            remove_pretrial=True,
            resample=False,
            new_sample_rate=signal_fs,
            resize=True,
            frame_size=frame_size,
            video_normalize=True,
            video_centering=True,
            signal_normalize=signal_normalize,
            root_dir=root_dir
        )
        if vid is not None and sig is not None:
            videos.append(vid)
            signals.append(sig)

    if not videos:
        raise ValueError(
            f"No valid data for participant={participant_id}, "
            f"experiments={experiments}, channel={channel_name}"
        )
    
    # ---- 2. Null model: permute signals (break video-EEG pairing) ----
    if shuffle_pairing:
        rng = np.random.default_rng(shuffle_seed)
        perm = rng.permutation(len(signals))

        # avoid fixed points (trial paired with itself)
        if len(perm) > 1:
            for i in range(len(perm)):
                if perm[i] == i:
                    j = (i + 1) % len(perm)
                    perm[i], perm[j] = perm[j], perm[i]

        signals = [signals[k] for k in perm]

    # ---- 3. Concatenate and segment ----
    videos = np.vstack(videos)
    signals = np.hstack(signals)

    X, y = segment_video_and_signal(
        video=videos,
        signals=signals,
        video_fr=video_fr,
        signal_fs=signal_fs,
        time_window=time_window,
        overlap=overlap,
        overlap_ratio=overlap_ratio,
        y_type=y_type,
        shuffle_output=shuffle_output,
    )
    return X, y


def segment_signal_only(signals,
                    signal_fs,
                    time_window=1,
                    overlap=False,
                    overlap_ratio=0,
                    y_type='RawEEG',
                    freqs_range=(4,45),
                    freqs_bands=None,
                    shuffle_output=False):
    '''
    description:
    ------------
    Segment an EEG signal into fixed-length windows, without any video.

    This mirrors the y-side of `segment_video_and_signal` exactly, so
    the resulting segments are compatible with those produced by the
    full video+signal pipeline.

    parameters:
    ------------
    -param signals: (numpy ndarray) EEG signal
    -param signal_fs: (integer) sampeling frequency of given signal
    -param time_window: (integer/float)length of segmentation window (in minutes), default is 1
    -param overlap: (boolean) if True: include overlapping in data segmentation, default is False
    -param overlap_ratio: (float) the overlap ratio of windows, default is 0
    -param y_type: (str) y_out type ('RawEEG'/'PSD'/'FBP') default is 'RawEEG'
    -param freqs_range: (tuple) used when y_type='PSD', range of frequency for getting PSD of that range, default is (4,45)
    -param freqs_bands: (dictionary/None) used when y_type='FBP'; if None, uses `config.FREQ_BANDS`
    -param shuffle_output: (bool) shuffle the segmented signal before returning.
    
    return:
    ------------
    y : segmented signal segments (n_segments, s_window)  or  (n_segments, n_freqs)  for PSD/FBP
    '''
    if freqs_bands is None:
        freqs_bands = FREQ_BANDS

    y_data = signals
    s_window = int(time_window * signal_fs)
    s_nop = int(s_window * overlap_ratio)

    if overlap:

        if y_type == 'RawEEG':
            y = np.array([y_data[i-s_nop:i+s_nop] for i in range(s_nop,y_data.shape[0],s_nop)])

        elif y_type == 'PSD':
            freqs,psd = signal.welch(y_data,fs=signal_fs, window='hann',nperseg=s_window,nfft=512)
            #select frequency range indices between lower_freq-upper_freq
            indices = np.logical_and(freqs>=freqs_range[0],freqs<=freqs_range[1])
            y = np.array([signal.welch(y_data[i-s_nop:i+s_nop],fs=signal_fs, window='hann',nperseg=s_window,nfft=512)[1][indices] for i in range(s_nop,y_data.shape[0],s_nop)])
        elif y_type =='FBP':
            y = []
            for i in range(s_nop,y_data.shape[0],s_nop):
                freqs,psd = signal.welch(y_data[i-s_nop:i+s_nop],fs=signal_fs, window='hann',nperseg=s_window,nfft=512)
                FBP = []
                for band_name in freqs_bands.keys():
                    #select frequency range indices between lower_freq-upper_freq
                    indices = np.logical_and(freqs>=freqs_bands[band_name][0],freqs<=freqs_bands[band_name][1])
                    FBP.append(np.max(psd[indices]))
                y.append(FBP)
            y = np.array(y)

    else:

        if y_type == 'RawEEG':
            y = y_data.reshape(-1,s_window)

        elif y_type == 'PSD':
            freqs,psd = signal.welch(y_data,fs=signal_fs, window='hann',nperseg=s_window,nfft=512)
            #select frequency range indices between lower_freq-upper_freq
            indices = np.logical_and(freqs>=freqs_range[0],freqs<=freqs_range[1])
            y = np.array([signal.welch(y_data[i:i+s_window],fs=signal_fs,window='hann',nperseg=s_window,nfft=512)[1][indices] for i in range(0,y_data.shape[0],s_window)])

        elif y_type =='FBP':
            y = []
            for i in range(0,y_data.shape[0],s_window):
                freqs,psd = signal.welch(y_data[i:i+s_window],fs=signal_fs, window='hann',nperseg=s_window,nfft=512)
                FBP = []
                for band_name in freqs_bands.keys():
                    #select frequency range indices between lower_freq-upper_freq
                    indices = np.logical_and(freqs>=freqs_bands[band_name][0],freqs<=freqs_bands[band_name][1])
                    FBP.append(np.mean(psd[indices]))
                y.append(FBP)
            y = np.array(y)
    if shuffle_output:
      return shuffle(y)
    else:
      return y


def load_participant_signal_segments(
    participant_id,
    experiments,
    scaler,
    channel_name,
    time_window=1,
    overlap=False,
    overlap_ratio=0.0,
    shuffle_output=False,
    signal_normalize=True,
    signal_fs=128,
    y_type='RawEEG',
    deap_eeg_channels=None,   
    root_dir='.'
):
    """
    description:
    ------------
     Load and segment ONLY the EEG signal of a participant, without
    decoding any video. This is a lightweight alternative to
    `load_participant_segments` for applications that only need the
    EEG side (e.g. trivial null-model baselines).

    The output y matches what `load_participant_segments` produces on
    the same trials, so the segments are directly comparable.

    parameters:
    ------------
    -param participant_id: (int) participant id
    -param experiments: (list/tuple) list of experiment ids
    -param scaler: (object) fitted scaler for EEG normalization
    -param channel_name: (str) key in `deap_eeg_channels` dict
    -param time_window: (int/float) segmentation window in seconds
    -param overlap: (bool) whether to use overlapping windows
    -param overlap_ratio: (float) overlap ratio when overlap=True
    -param shuffle_output: (bool) shuffle the segmented signal before returning.
        Note: in this signal-only path there is no X to keep in sync, so only y is shuffled.
    -param signal_normalize: (bool) apply scaler to signals
    -param signal_fs: (int) EEG sampling frequency
    -param y_type: (str) 'RawEEG' / 'PSD' / 'FBP'
    -deap_eeg_channels: (dict/None) eeg channel name -> index; if None, uses `config.DEAP_EEG_CHANNELS`
    -param root_dir: (str) root directory containing physiological data and facial videos

    return:
    ------------
    y : segmented signal

    """
    if deap_eeg_channels is None:
        deap_eeg_channels = DEAP_EEG_CHANNELS

    if channel_name not in deap_eeg_channels:
        raise KeyError(f"Channel '{channel_name}' not found.")
    
    # ---- 1. Load EEG of all trials ----
    signals =  []
    for experiment_id in experiments:
        sig_raw, _ = load_participant_physiological_data(
            participant_id,
            [experiment_id],
            [deap_eeg_channels[channel_name]],
            remove_pretrial=True,
            resample=False,
            new_sample_rate=signal_fs,
            root_dir=root_dir,
        )
        if sig_raw is None:
            continue
        sig = (scaler.transform(sig_raw.reshape(-1, 1)).reshape(-1)
               if signal_normalize else sig_raw.reshape(-1))
        signals.append(sig)

    if not signals:
        raise ValueError(
            f"No valid EEG for participant={participant_id}, "
            f"experiments={experiments}, channel={channel_name}"
        )
    
    # ---- 2. Concatenate and segment ----
    signals = np.hstack(signals)

    y = segment_signal_only(
        signals=signals,
        signal_fs=signal_fs,
        time_window=time_window,
        overlap=overlap,
        overlap_ratio=overlap_ratio,
        y_type=y_type,
        shuffle_output=shuffle_output,
    )
    return y        