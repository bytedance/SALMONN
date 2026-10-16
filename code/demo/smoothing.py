"""Smooth committed RVQ block boundaries without using subsequent blocks."""
import numpy as np
from scipy.spatial.transform import Rotation

from models.vq.config import UPPER_JOINTS


def smooth_boundary(block, previous, frames):
    """Fade a rotation correction in place, preserving the previous angular velocity.

    Only upper-body and hand rotations change. The correction reaches zero inside
    the current completed block; earlier frames and facial motion stay unchanged.
    """
    if previous is None or not frames:
        return block
    count = min(frames, len(block["body_pose"]))
    phase = np.linspace(0, 1, count)
    weight = (1 - 3 * phase**2 + 2 * phase**3)[:-1]
    for key in ("body_pose", "left_hand_pose", "right_hand_pose"):
        poses = block[key].reshape(len(block[key]), -1, 3)
        history = previous[key][-2:].reshape(2, -1, 3)
        joints = UPPER_JOINTS if key == "body_pose" else np.arange(poses.shape[1])
        before = Rotation.from_rotvec(history[0, joints])
        last = Rotation.from_rotvec(history[1, joints])
        predicted = last * (before.inv() * last)
        first = Rotation.from_rotvec(poses[0, joints])
        correction = (predicted * first.inv()).as_rotvec()
        offset = Rotation.from_rotvec((weight[:, None, None] * correction).reshape(-1, 3))
        raw = Rotation.from_rotvec(poses[:count - 1, joints].reshape(-1, 3))
        poses[:count - 1, joints] = (offset * raw).as_rotvec().reshape(count - 1, len(joints), 3)
    return block
