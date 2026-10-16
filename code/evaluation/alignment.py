"""Matched/mismatched motion-audio alignment with a frozen development scale."""
import hashlib

import librosa
import soundfile as sf
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit

def audio_cache_path(output_dir, audio_path):
    # A replaced WAV must not reuse features cached for an older file at the same path.
    digest = hashlib.sha256(b'onset-16k-hop320-v1' + audio_path.read_bytes()).hexdigest()
    return output_dir / 'audio_cache' / f'{digest}.npz'

def extract_audio(args):
    (audio_path, cache_path) = args
    if cache_path.is_file():
        return str(audio_path)
    times, onset = audio_onset(audio_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, times=times, onset=onset)
    return str(audio_path)

def audio_onset(path):
    """The frozen scorer uses only onset strength; pitch and VAD never enter it."""
    audio, sample_rate = sf.read(path, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    assert audio.size and np.isfinite(audio).all(), path
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    onset = librosa.onset.onset_strength(y=audio, sr=16000, hop_length=320)
    # Preserve the original centered 40 ms RMS/onset common timeline exactly.
    frames = min(len(onset), len(librosa.feature.rms(
        y=audio, frame_length=640, hop_length=320, center=True)[0]))
    times = librosa.frames_to_time(np.arange(frames), sr=16000, hop_length=320)
    return times, onset[:frames]

def load_body_motion(path):
    data = np.load(path)
    body_key = 'body_pose' if 'body_pose' in data else 'smplh:body_pose'
    body = np.asarray(data[body_key], dtype=np.float32).reshape(len(data[body_key]), -1)
    fps = float(np.asarray(data['mocap_frame_rate']).reshape(-1)[0]) if 'mocap_frame_rate' in data else 30.0
    upper_body = body[:, min(27, body.shape[1]):]
    difference = np.diff(upper_body, axis=0, prepend=upper_body[:1]) * fps
    energy = np.sqrt(np.mean(difference * difference, axis=1) + 1e-08)
    times = np.arange(len(body), dtype=np.float32) / fps
    return (times, energy, fps)

def standardize(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float64))
    median = np.median(values)
    scale = np.quantile(np.abs(values - median), 0.75) + 1e-06
    return np.clip((values - median) / scale, -8, 8)

def correlation(left, right):
    (left, right) = (standardize(left), standardize(right))
    denominator = np.sqrt(np.sum(left * left) * np.sum(right * right))
    return float(np.sum(left * right) / denominator) if denominator > 1e-08 else 0.0

def plausible_correlation(audio_cache, motion_times, motion_energy):
    cached = np.load(audio_cache)
    audio_times = np.asarray(cached['times'])
    duration = min(float(audio_times[-1]), float(motion_times[-1]))
    grid = np.arange(0, duration, 0.05)
    if len(grid) < 20:
        raise RuntimeError(f'Clip too short: {audio_cache}')
    onset = np.interp(grid, audio_times, cached['onset'])
    body = np.interp(grid, motion_times, motion_energy)
    onset = gaussian_filter1d(onset, 0.25 / 0.05)
    body = gaussian_filter1d(body, 0.25 / 0.05)
    values = []
    for lag in range(-8, 9):
        if lag < 0:
            values.append(correlation(onset[-lag:], body[:lag]))
        elif lag > 0:
            values.append(correlation(onset[:-lag], body[lag:]))
        else:
            values.append(correlation(onset, body))
    return float(np.max(values))

def score_one(args):
    (system, group_id, motion_path, matched_audio, mismatched_audio, output_dir, pair_scale, metadata) = args
    (motion_times, motion_energy, fps) = load_body_motion(motion_path)
    matched = plausible_correlation(audio_cache_path(output_dir, matched_audio), motion_times, motion_energy)
    mismatched = plausible_correlation(audio_cache_path(output_dir, mismatched_audio), motion_times, motion_energy)
    difference = matched - mismatched
    probability = float(expit(np.clip(difference / pair_scale, -8, 8)))
    return {'system': system, 'group_id': group_id, 'source_kind': metadata['source_kind'], 'scene_label': metadata['scene_label'], 'motion_path': str(motion_path), 'motion_fps': fps, 'matched_audio': str(matched_audio), 'mismatched_group_id': metadata['mismatched_group_id'], 'mismatched_audio': str(mismatched_audio), 'matched_correlation': matched, 'mismatched_correlation': mismatched, 'feature_difference': difference, 'matched_probability': probability}
