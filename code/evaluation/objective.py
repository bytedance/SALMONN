"""Speaking/listening motion contrast on the frozen root-fixed skeleton."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def contrast(path, row, joints, parents):
    with np.load(path) as data:
        key = 'body_pose' if 'body_pose' in data else 'smplh:body_pose'
        body = np.asarray(data[key], dtype=np.float64).reshape(-1, 21, 3).copy()
    assert len(body) == 600 and np.isfinite(body).all()
    body[:, [0, 1, 3, 4, 6, 7, 9, 10]] = 0
    # The caller supplies the same zero-shape SMPL-X skeleton for every system.
    rotations = Rotation.from_rotvec(
        np.concatenate([np.zeros((600, 1, 3)), body], axis=1).reshape(-1, 3)
    ).as_matrix().reshape(600, 22, 3, 3)
    world = np.empty_like(rotations)
    xyz = np.zeros((600, 22, 3))
    world[:, 0] = rotations[:, 0]
    for joint in range(1, 22):
        parent = parents[joint]
        world[:, joint] = world[:, parent] @ rotations[:, joint]
        xyz[:, joint] = xyz[:, parent] + np.einsum(
            'tij,j->ti', world[:, parent], joints[joint] - joints[parent]
        )
    speed = np.linalg.norm(np.diff(xyz[:, JOINTS], axis=0), axis=-1).mean(axis=1)
    times = np.arange(1, 600) / 30

    def speech_mask(metadata_path):
        intervals = json.loads(Path(metadata_path).read_text())['metadata:vad']
        return np.array([any(v['start'] <= t < v['end'] for v in intervals) for t in times])

    valid = (times >= row['rating_start_sec']) & (times < row['rating_end_sec'])
    spoken = speech_mask(row['metadata20_target'])
    partner = speech_mask(row['metadata20_partner'])
    # Overlap counts as speaking; mutual silence is excluded from listening.
    speaking = valid & spoken
    listening = valid & ~spoken & partner
    energy_s = float(speed[speaking].mean()) if speaking.any() else None
    energy_l = float(speed[listening].mean()) if listening.any() else None
    ratio = None
    if energy_s is not None and energy_l is not None:
        assert energy_l > 0
        ratio = energy_s / energy_l
    return dict(contrast=ratio, E_S=energy_s, E_L=energy_l,
                speaking_frames=int(speaking.sum()), listening_frames=int(listening.sum()))
