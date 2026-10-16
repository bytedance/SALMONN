"""FLAME metrics on the intersection of face and body validity in 4–16 seconds."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from . import face_metrics as official

def interp(x, fps, rotation=False):
    x = np.asarray(x, dtype=np.float64)
    t = np.arange(len(x)) / fps
    q = np.minimum(np.arange(600) / 30, t[-1])
    assert len(x) >= int(20 * fps), x.shape
    if rotation:
        return Slerp(t, Rotation.from_rotvec(x))(q).as_rotvec().astype(np.float32)
    return np.stack([np.interp(q, t, x[:, i]) for i in range(x.shape[1])], axis=1).astype(np.float32)

def body_head(body):
    b = np.asarray(body).reshape(-1, 21, 3)
    rot = np.broadcast_to(np.eye(3), (len(b), 3, 3)).copy()
    for joint in [3, 6, 9, 12, 15]:
        rot = rot @ Rotation.from_rotvec(b[:, joint - 1]).as_matrix()
    return Rotation.from_matrix(rot).as_rotvec()

def relative_head(x):
    rot = Rotation.from_rotvec(x)
    return (rot[0].inv() * rot).as_rotvec().astype(np.float32)

def masks(r, d):
    t = np.arange(600) / 30
    valid = np.asarray(d['valid_mask']).astype(bool)
    with np.load(r['pose20_gt']) as pose:
        valid &= np.asarray(pose['smplh:is_valid']).astype(bool)
    # Face metrics count mutual silence as listening, unlike body contrast.
    valid &= (t >= r['rating_start_sec']) & (t < r['rating_end_sec'])
    vad = json.loads(Path(r['metadata20_target']).read_text())['metadata:vad']
    assert all((0 <= v['start'] < v['end'] <= 20.000001 for v in vad))
    sp = np.array([any((v['start'] <= q < v['end'] for v in vad)) for q in t])
    return (valid & sp, valid & ~sp)

def vertices(model, e, j, device):
    out = []
    for i in range(0, len(e), 48):
        out.append(model(expression=torch.tensor(e[i:i + 48], device=device), jaw_pose=torch.tensor(j[i:i + 48], device=device)))
    return torch.cat(out)[None]

def score(v, gt, jaw, gj, ms, ml):
    result = {}
    for (prefix, mask) in [('S', ms), ('L', ml)]:
        result[prefix + '_frames'] = int(mask.sum())
        if mask.sum() < 2:
            for k in ['LVE', 'MHD', 'FDD', 'JDD']:
                result[prefix + '_' + k] = None
            continue
        vm = v[:, mask]
        gm = gt[:, mask]
        (lve, mhd, fdd) = official.calc_val_metrics(vm, gm)
        result.update({prefix + '_LVE': float(lve), prefix + '_MHD': float(mhd),
                       prefix + '_FDD': float(fdd),
                       prefix + '_JDD': float(abs(jaw[mask, 0].std(ddof=1)
                                                 - gj[mask, 0].std(ddof=1)) * 100)})
    return result
