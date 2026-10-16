"""A direct pose-only reference using the checkpoint's eight-window guide layout."""
import math

import numpy as np

REFERENCE_FRAMES = 192


def pose_guide_schedule(path, start_frame, time_sec, num_windows):
    """Validate a 30 fps pose clip and place it on the 0.8 s generation grid."""
    chunk = round(time_sec / 0.8) if math.isfinite(time_sec) else -1
    if chunk < 0 or not math.isclose(time_sec, chunk * 0.8, abs_tol=1e-6):
        raise ValueError("Pose guide time must be a non-negative multiple of 0.8 seconds.")
    if chunk + 8 > num_windows:
        raise ValueError("The full 6.4-second pose guide must fit within the generated audio timeline.")
    if start_frame < 0 or int(start_frame) != start_frame:
        raise ValueError("Pose guide start frame must be a non-negative integer.")
    start_frame = int(start_frame)
    with np.load(path, allow_pickle=False) as data:
        if "mocap_frame_rate" in data and float(data["mocap_frame_rate"]) != 30:
            raise ValueError("Pose guide must use 30 fps.")
        for name, joints in (("body_pose", 21), ("left_hand_pose", 15), ("right_hand_pose", 15)):
            key = name if name in data else f"smplh:{name}"
            clip = data[key][start_frame:start_frame + REFERENCE_FRAMES]
            if clip.shape not in ((REFERENCE_FRAMES, joints, 3), (REFERENCE_FRAMES, joints * 3)) or not np.isfinite(clip).all():
                raise ValueError(f"Pose guide {name} must contain 192 finite frames of shape ({joints}, 3) or ({joints * 3},).")
    return {chunk: {
        "action_family": "pose_only", "query_text": "<pose_guide>", "matched_text": "<pose_guide>",
        "reference": {
            "event_uid": "pose_guide", "pose_file": str(path),
            "causal_reference_start_frame": start_frame,
            "causal_reference_length_frames": REFERENCE_FRAMES,
            "reference_anchor_frame": 0,
        },
    }}
