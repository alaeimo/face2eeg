# face2eeg/core/data/preprocess_faces.py
"""
One-off batch preprocessing of raw facial videos.

Run once per subject set:
    python -m face2eeg.core.data.preprocess_faces
"""

import os
from .video import preprocess_video
from ..config import VALID_SUBJECTS

# ---- edit these paths ----
RAW_ROOT = r'D:\Thesis\Codes\Data\raw_face_videos'
OUT_ROOT = r'D:\Thesis\Codes\Data\face_videos'

SUBJECTS = VALID_SUBJECTS


def main():
    total, ok, skipped, failed = 0, 0, 0, 0

    for sid in SUBJECTS:
        for trial in range(1, 41):
            in_path = os.path.join(
                RAW_ROOT, f's{sid:02d}', 
                f's{sid:02d}_trial{trial:02d}.avi',
            )
            out_path = os.path.join(
                OUT_ROOT, f's{sid:02d}_trial{trial:02d}_cropped.avi',
            )

            total += 1
            if not os.path.isfile(in_path):
                continue

            try:
                res = preprocess_video(in_path, out_path, verbose=True)
                if res['skipped']:
                    skipped += 1
                else:
                    ok += 1
            except Exception as e:
                failed += 1
                print(f'[fail] s{sid:02d} t{trial:02d}: {e}')

    print(f'\nDone. total={total}  ok={ok}  skipped={skipped}  failed={failed}')


if __name__ == '__main__':
    main()