# face2eeg/core/data/video.py
"""
Face detection, cropping and normalization for DEAP facial videos.

Direct port of the original preprocessing pipeline, preserving the
exact behavior (3000-frame detection window,
fixed 256x256 output, DIVX @ 50 fps).

Load a participant's face video for a given experiment_id.
"""
import os
import numpy as np
import pims
import cv2
from .cache import _VIDEO_CACHE
from .mapper import get_exp2trial_mapper

# ============================================================
# Lazy singleton for the face detector
# ============================================================
_FACE_DETECTOR = None


def _get_face_detector():
    import dlib
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        _FACE_DETECTOR = dlib.get_frontal_face_detector()
    return _FACE_DETECTOR

# ============================================================
# 1. Face detection on a subsampled grid
# ============================================================
def detect_video_faces(video, step=25, max_frames=3000):
    """
    Detect the first face every `step` frames, up to `max_frames`.

    Returns
    -------
    np.ndarray, shape (n_detected, 4), columns = (top, bottom, left, right).
    Empty array if no faces are detected.
    """
    detector = _get_face_detector()
    bboxes = []
    n = min(len(video), max_frames)
    for i in range(0, n, step):
        frame = np.copy(video[i])
        faces = detector(frame)
        try:
            bboxes.append([
                faces[0].top(),
                faces[0].bottom(),
                faces[0].left(),
                faces[0].right(),
            ])
        except Exception:
            pass
    return np.array(bboxes)


# ============================================================
# 2. Average the detected boxes into a single box
# ============================================================
def bounding_box_norm(bboxes):
    """Average a stack of (top, bottom, left, right) boxes."""
    new_bboxes = np.round(np.mean(bboxes, axis=0))
    new_bboxes = [int(i) for i in new_bboxes]
    return new_bboxes


# ============================================================
# 3. Crop and resize every frame with the fixed box
# ============================================================
def crop_faces(video, bounding_box, frame_size=(256, 256)):
    """
    Crop every frame with the fixed bounding box and resize to `frame_size`.

    Matches the original behavior:
      - each frame is converted BGR -> RGB
      - cropped with (top:bottom, left:right)
      - resized with INTER_CUBIC
    """
    cropped_faces = []
    for i in range(len(video)):
        frame = video[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame[bounding_box[0]:bounding_box[1],
                      bounding_box[2]:bounding_box[3]]
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_CUBIC)
        cropped_faces.append(frame)
    return cropped_faces


# ============================================================
# 4. Main pipeline
# ============================================================
def preprocess_video(
    video_path,
    out_path,
    *,
    frame_size=(256, 256),
    fps=50.0,
    detect_step=25,
    detect_max_frames=3000,
    skip_if_exists=True,
    verbose=False,
):
    """
    Crop and resize one facial video using a single averaged bounding box.

    Parameters
    ----------
    video_path : str
        Path to the raw .avi (read via pims.Video).
    out_path : str
        Path for the processed .avi. Parent directory is created if missing.
    frame_size : tuple, default (256, 256)
        Output resolution (width, height).
    fps : float, default 50.0
        Output frame rate.
    detect_step : int, default 25
        Face detection runs every `detect_step` frames.
    detect_max_frames : int, default 3000
        Only the first `detect_max_frames` frames are inspected for the box.
    skip_if_exists : bool, default True
        Return early if `out_path` already exists.
    verbose : bool, default False

    Returns
    -------
    dict
        Metadata about the processed (or skipped) video.
    """
    if skip_if_exists and os.path.isfile(out_path):
        if verbose:
            print(f"[skip] {out_path} already exists")
        return {
            "video_path": video_path,
            "out_path":   out_path,
            "n_frames":   -1,
            "bbox":       None,
            "skipped":    True,
        }

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    video = pims.Video(video_path)

    # ---- 1. detect faces on a subsample of frames ----
    bboxes = detect_video_faces(
        video, step=detect_step, max_frames=detect_max_frames
    )
    if bboxes.size == 0:
        raise RuntimeError(f"No faces detected in {video_path}")

    # ---- 2. average -> single box ----
    new_bbox = bounding_box_norm(bboxes)

    # ---- 3. crop all frames with that box ----
    faces = crop_faces(video, new_bbox, frame_size=frame_size)

    # ---- 4. write the processed video ----
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
    if not out.isOpened():
        raise RuntimeError(f"VideoWriter could not open {out_path}")

    try:
        for face in faces:
            out.write(face)
    finally:
        out.release()

    if verbose:
        print(f"[ok] {out_path}  frames={len(faces)}  bbox={new_bbox}")

    return {
        "video_path": video_path,
        "out_path":   out_path,
        "n_frames":   len(faces),
        "bbox":       new_bbox,
        "skipped":    False,
    }

def load_participant_facial_video(participant_id, 
                                  experiment_id, 
                                  resize=True, 
                                  frame_size=(32, 32), 
                                  normalize=True, 
                                  centering=True,
                                  root_dir='.', 
                                  mapper_filename='experiment2trial.csv',
                                  video_subdir='Face_videos',):
    '''
    description:
    ------------
    Load face video of a participant  according to the given experiment_id

    parameters:
    ------------
    -param participant_id: (integer) a number between 1,22
    -param experiment_id: (integer) a number between [0:40)
    -param resize: (boolean) if true: resize frames based on given height and width
    -param frame_size: (tuple) represents the number of pixel rows and columns in the image
    -param normalize: (boolean) escale pixel values from the range of 0-255 to the range 0-1
    -param centering: (boolean) calculate the mean pixel value across the each frame channels, then subtract it from frame
    -param root_dir: (str) root directory containing video and mapper files
    -param mapper_filename: (str) name of the experiment2trial csv file
    -param video_subdir: (str) subdirectory name containing facial videos
    
    return:
    ------------
    facial video/None: (numpy array/None) with shape (frames,height,width,channels):e.g. (3000,256,256,3)/ None: if the video didn't exist
    label/None : (numpy array) with shape (4,) for emotional ratings of (Valence,Arousal,Dominance,Liking)
    '''

    # ---------------- cache lookup ----------------
    cache_key = (participant_id, experiment_id, frame_size,
                resize, normalize, centering,
                root_dir, video_subdir)
    if cache_key in _VIDEO_CACHE:
        video_arr, label = _VIDEO_CACHE[cache_key]
        return video_arr, label
    
    exp_to_trial_mapper_path = os.path.join(root_dir, mapper_filename)
    trials, labels = get_exp2trial_mapper(exp_to_trial_mapper_path) #get experiment_id to trial_id mapper
    trial_id = trials[participant_id-1][experiment_id] #get trial_id(video number or YY) according to given participant_id and experiment_id
    #convert trial_id and participant_id to the YY and XX format respectively
    if trial_id <10:
        trial_id = "0" + str(trial_id)
    else:
        trial_id = str(trial_id)
    if participant_id<10:
        participant_id = "0" + str(participant_id)
    else:
        participant_id = str(participant_id)
    #create video path based on participant_id and trial_id
    video_path = os.path.join(root_dir, video_subdir, f's{participant_id}_trial{trial_id}_cropped.avi')
    if os.path.isfile(video_path): #check out video_path
        video = pims.Video(video_path) #load the video
        if resize or normalize or centering: #resize or normalize to (0:1) or zero-mean centering of each video frame
            facial_video = []
            for n_frame in range(len(video)):
                frame = video[n_frame]
                if resize:
                    frame = cv2.resize(frame,frame_size,interpolation=cv2.INTER_CUBIC)
                if normalize:
                    frame = frame/255
                if centering:
                    frame = frame - frame.mean(axis=(0,1), keepdims=True)
                facial_video.append(frame)
            facial_video = np.array(facial_video,dtype=np.float32)
        else:
            facial_video = np.copy(video)

        label = labels[int(participant_id)-1][experiment_id]
        _VIDEO_CACHE[cache_key] = (facial_video, label)

        return facial_video, label
    else:
        return None,labels[int(participant_id)-1][experiment_id]
