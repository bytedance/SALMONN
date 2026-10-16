"""Frozen v10 judge parsing: low energy is evidence, never a speech label."""
import json, subprocess
import numpy as np
from pathlib import Path
TASKS = {'duplex': 'duplex_judge_base.txt', 'semantic': 'semantic_judge_base.txt'}
PROTOCOLS = Path(__file__).resolve().parent / 'protocols'


def task_prompt(task):
    # Three newlines reproduce the saved combined prompt byte for byte.
    base = (PROTOCOLS / TASKS[task]).read_text().rstrip()
    return base + '\n\n\n' + (PROTOCOLS / 'activity_addendum.txt').read_text().lstrip()

def quiet(path):
    raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path), '-vn', '-ac', '1', '-ar', '16000', '-f', 'f32le', 'pipe:1'])
    x = np.frombuffer(raw, dtype='<f4')
    frames = x[:len(x) // 320 * 320].reshape(-1, 320)
    low = np.sqrt(np.mean(frames ** 2, axis=1)) < 0.01
    edges = np.diff(np.r_[False, low, False].astype(int))
    out = []
    for (a, b) in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        start = a * 0.02 + 0.3
        end = min(b * 0.02, 20) - 0.3
        if end - start >= 1.0:
            out.append([round(start, 3), round(end, 3)])
    return out

def parse(response, entry):
    d = json.loads(response)
    if isinstance(d, list) and len(d) == 1:
        d = d[0]
    assert isinstance(d, dict)
    keys = ['Timing', 'Appropriateness', 'Gesture backchannel visibility'] if entry['task'] == 'duplex' else ['Visual clarity', 'Text-motion semantic match', 'Concreteness', 'Beat/fidget penalty']
    for key in keys + ['Silence motion penalty', 'Overactivity penalty']:
        assert type(d[key]) in (int, float) and np.isfinite(d[key]) and (0 <= d[key] <= 1), key
    S = d['Silence motion penalty']
    O = d['Overactivity penalty']
    windows = {}
    for (field, value) in [('Silence evidence', S), ('Overactivity evidence', O)]:
        assert isinstance(d[field], list)
        windows[field] = []
        if value > 0:
            assert d[field], field
        for ev in d[field]:
            (start, end) = (ev['start_sec'], ev['end_sec'])
            assert type(start) in (int, float) and type(end) in (int, float)
            assert 0 <= start < end <= 20
            assert all((isinstance(ev[k], str) and ev[k].strip() for k in ['candidate_motion', 'reference_motion', 'why_unjustified']))
            if field == 'Silence evidence':
                assert end - start >= 1.0 - 1e-05
                assert ev['target_speech_absent'] is True and any((start >= a - 1e-05 and end <= b + 1e-05 for (a, b) in entry['quiet_windows']))
            windows[field].append((start, end))
    if not entry['quiet_windows']:
        assert S == 0
    for (a, b) in windows['Silence evidence']:
        for (c, dend) in windows['Overactivity evidence']:
            assert min(b, dend) <= max(a, c), 'duplicate interval penalty'
    if entry['task'] == 'duplex':
        base = 0.45 * d['Timing'] + 0.45 * d['Appropriateness'] + 0.10 * d['Gesture backchannel visibility']
    else:
        base = float(np.clip(0.35 * d['Visual clarity'] + 0.45 * d['Text-motion semantic match']
                             + 0.20 * d['Concreteness'] - 0.30 * d['Beat/fidget penalty'], 0, 1))
    return {'base': base, 'silence': S, 'overactivity': O,
            'deduction': base * (0.45 * S + 0.35 * O),
            'final': base * (1 - 0.45 * S - 0.35 * O), 'parsed': d}
