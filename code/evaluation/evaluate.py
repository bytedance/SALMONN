"""Evaluate motion, face, MLLM gesture judgments, or all three metric groups."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_paths import DATA_ROOT
from evaluation.statistics import BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES, summarize


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', choices=['motion', 'face', 'mllm', 'all'], default='motion')
    parser.add_argument('--prediction-dir', help='Directory of generated group_id.npz files; required for motion/face')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--smplx-model', help='SMPLX_NEUTRAL_2020.npz for motion skeleton metrics')
    parser.add_argument('--flame-asset-dir', help='Directory with flame_2020.pt; required for face metrics')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--test-id', action='append', help='Selected frozen clip IDs (repeatable)')
    parser.add_argument('--pose-diagnostics', action='store_true',
                        help='Also report historical motion PDD/L_PFID in motion/all mode')
    parser.add_argument('--video-dir', help='Rendered group_id.mp4 files for MLLM; defaults to --prediction-dir')
    parser.add_argument('--mllm-score-dir', help='Aggregate existing four-judge records offline instead of making API requests')
    parser.add_argument('--max-pending', type=int, default=4, help='Default submitted-job limit per provider; individual provider limits override it')
    parser.add_argument('--mllm-retries', type=int, default=5)
    for provider in ['gemini', 'seed']:
        env = provider.upper()
        parser.add_argument(f'--{provider}-api-key', help=f'Explicit API key; otherwise read --{provider}-api-key-env')
        parser.add_argument(f'--{provider}-api-key-env', default=f'{env}_API_KEY')
        parser.add_argument(f'--{provider}-max-pending', type=int,
                            help='Submitted-job limit for this provider; defaults to --max-pending')
        parser.add_argument(f'--{provider}-model', default=os.environ.get(f'{env}_MODEL', 'gemini-2.5-pro' if provider == 'gemini' else None),
                            help='Model/deployment identifier used by the service')
        parser.add_argument(f'--{provider}-base-url', default=os.environ.get(f'{env}_BASE_URL'))
        parser.add_argument(f'--{provider}-azure-endpoint', default=os.environ.get(f'{env}_AZURE_ENDPOINT'))
        parser.add_argument(f'--{provider}-api-version', default=os.environ.get(f'{env}_API_VERSION', '2024-03-01-preview'))
        parser.add_argument(f'--{provider}-video-format', choices=['video_url', 'azure_image'],
                            default=os.environ.get(f'{env}_VIDEO_FORMAT', 'azure_image' if provider == 'gemini' else 'video_url'))
    return parser.parse_args(argv)


def metric_groups(args):
    groups = ['motion', 'face', 'mllm'] if args.metrics == 'all' else [args.metrics]
    assert args.limit >= 0
    if 'motion' in groups or 'face' in groups:
        assert args.prediction_dir, 'Provide --prediction-dir for motion/face metrics'
    if 'motion' in groups:
        assert args.smplx_model, 'Provide --smplx-model for motion metrics'
    if 'face' in groups:
        assert args.flame_asset_dir, 'Provide --flame-asset-dir for face metrics'
    assert not args.pose_diagnostics or 'motion' in groups, '--pose-diagnostics requires motion or all'
    return groups


def provider_options(args, provider, out, ids):
    def value(name):
        return getattr(args, provider + '_' + name)
    assert value('model'), f'Provide --{provider}-model or set {provider.upper()}_MODEL'
    assert bool(value('base_url')) != bool(value('azure_endpoint')), f'Configure exactly one {provider} endpoint'
    assert value('api_key') or os.environ.get(value('api_key_env')), f'Provide --{provider}-api-key or set {value("api_key_env")}'
    max_pending = value('max_pending') if value('max_pending') is not None else args.max_pending
    assert max_pending >= 1, f'{provider} pending-job limit must be positive'
    if provider == 'gemini':
        assert max_pending <= 4, 'At most four submitted Gemini jobs at a time'
    assert 1 <= args.mllm_retries <= 5
    video_dir = args.video_dir or args.prediction_dir
    assert video_dir, 'Provide --video-dir for online MLLM scoring'
    return SimpleNamespace(
        provider=provider, model=value('model'), api_key=value('api_key'), api_key_env=value('api_key_env'),
        base_url=value('base_url'), azure_endpoint=value('azure_endpoint'), api_version=value('api_version'),
        video_format=value('video_format'), video_dir=video_dir, output_dir=out / 'mllm/judges' / provider,
        rounds=2, max_pending=max_pending, retries=args.mllm_retries, limit=0, test_id=ids,
    )


def objective_metrics(rows, args, out, groups):
    import numpy as np
    import torch
    motion, face = 'motion' in groups, 'face' in groups
    if motion:
        from evaluation import alignment, objective
        with np.load(args.smplx_model) as skeleton:
            joints = np.asarray(skeleton['J_regressor'] @ skeleton['v_template'])[:22]
            parents = np.asarray(skeleton['kintree_table'][0, :22], dtype=np.int64)
        mismatch = {r['group_id']: r for r in map(json.loads, (DATA_ROOT / 'testset/alignment_mismatches.jsonl').read_text().splitlines())}
        scale = json.loads((ROOT / 'evaluation/alignment_protocol.json').read_text())['pair_scale']
        cache = out / 'motion'
        cache.mkdir(parents=True, exist_ok=True)
        paths = {DATA_ROOT / mismatch[row['group_id']][k] for row in rows for k in ['matched_audio', 'mismatched_audio']}
        for path in sorted(paths):
            alignment.extract_audio((path, alignment.audio_cache_path(cache, path)))
    torch.set_num_threads(2)
    if face or args.pose_diagnostics:
        from evaluation import expression
    if face:
        from evaluation.flame import FLAMEModel
        os.environ['FLAME_ASSET_DIR'] = str(Path(args.flame_asset_dir).resolve())
        model = FLAMEModel(n_shape=300, n_exp=100).to(args.device).eval()

    results, pose_statistics = [], []
    for row in rows:
        gid = row['group_id']
        pred = Path(args.prediction_dir).resolve() / (gid + '.npz')
        assert pred.is_file(), pred
        score = {'group_id': gid, 'task': row['human_approved_task']}
        if motion:
            pair = mismatch[gid]
            aligned = alignment.score_one(('ELLSA2', gid, pred, DATA_ROOT / pair['matched_audio'],
                                           DATA_ROOT / pair['mismatched_audio'], cache, scale, pair))
            score['motion'] = {**objective.contrast(pred, row, joints, parents),
                               'alignment': aligned['matched_probability'],
                               'matched_correlation': aligned['matched_correlation'],
                               'mismatched_correlation': aligned['mismatched_correlation']}
        if face or args.pose_diagnostics:
            with np.load(row['flame_gt']) as ground_truth, np.load(pred) as prediction:
                gt = {key: ground_truth[key] for key in ['pose', 'expression', 'valid_mask']}
                data = {key: prediction[key] for key in prediction.files
                        if key in {'body_pose', 'smplh:body_pose', 'motioncode', 'mocap_frame_rate'}}
            fps = float(data.get('mocap_frame_rate', 30))
            speaking, listening = expression.masks(row, gt)
            if args.pose_diagnostics:
                from evaluation.pose_fid import moments
                body_key = 'body_pose' if 'body_pose' in data else 'smplh:body_pose'
                head = expression.relative_head(expression.interp(expression.body_head(data[body_key]), fps, True))
                gt_head = expression.relative_head(gt['pose'][:, :3])
                for prefix, mask in [('S', speaking), ('L', listening)]:
                    score['motion'][prefix + '_frames'] = int(mask.sum())
                    score['motion'][prefix + '_PDD'] = (float(np.abs(head[mask].std(0, ddof=1)
                                                                  - gt_head[mask].std(0, ddof=1)).mean() * 100)
                                                       if mask.sum() >= 2 else None)
                if listening.sum() >= 2:
                    pose_statistics.append({'group_id': gid, 'pred': moments(head[listening]),
                                            'gt': moments(gt_head[listening])})
            if face:
                with torch.inference_mode():
                    motioncode = data['motioncode']
                    predicted_expression = expression.interp(motioncode[:, :100], fps)
                    jaw = np.zeros((len(motioncode), 3))
                    jaw[:, 0] = motioncode[:, 103]
                    jaw = expression.interp(jaw, fps, True)
                    gt_expression = gt['expression'].astype(np.float32)
                    gt_jaw = gt['pose'][:, 3:6].astype(np.float32)
                    vertices = expression.vertices(model, predicted_expression, jaw, args.device)
                    gt_vertices = expression.vertices(model, gt_expression, gt_jaw, args.device)
                    score['face'] = expression.score(vertices, gt_vertices, jaw, gt_jaw, speaking, listening)
        results.append(score)
        print(gid, flush=True)
    return results, pose_statistics


def mllm_metrics(args, out, ids, options):
    from evaluation.aggregate_mllm import aggregate
    from evaluation.score_mllm import run_many
    if args.mllm_score_dir:
        scores = Path(args.mllm_score_dir)
        assert scores.is_dir(), scores
    else:
        run_many([options[provider] for provider in ['gemini', 'seed']])
        scores = out / 'mllm/judges'
    entries = json.loads((DATA_ROOT / 'testset/judge_context.json').read_text())
    selected = set(ids)
    entries = [row for row in entries if row['group_id'] in selected]
    records = [record for path in scores.rglob('*.json')
               if (record := json.loads(path.read_text())).get('group_id') in selected]
    return aggregate(records, entries)


def main(argv=None):
    args = parse_args(argv)
    groups = metric_groups(args)
    out = Path(args.output_dir).resolve()
    rows = [json.loads(line) for line in (DATA_ROOT / 'testset/manifest.jsonl').read_text().splitlines() if line]
    if args.test_id:
        assert set(args.test_id) <= {row['group_id'] for row in rows}, 'Unknown test ID'
        rows = [row for row in rows if row['group_id'] in args.test_id]
    rows = [{k: str(DATA_ROOT / v) if isinstance(v, str) and v.startswith('testset/') else v
             for k, v in row.items()} for row in rows[:args.limit or None]]
    assert rows, 'No selected test clips'
    ids = [row['group_id'] for row in rows]
    options = {}
    if 'mllm' in groups and not args.mllm_score_dir:
        options = {provider: provider_options(args, provider, out, ids) for provider in ['gemini', 'seed']}
    out.mkdir(parents=True, exist_ok=True)
    offline = [group for group in groups if group != 'mllm']
    if offline:
        results, pose_statistics = objective_metrics(rows, args, out, offline)
    else:
        results = [{'group_id': row['group_id'], 'task': row['human_approved_task']} for row in rows]
        pose_statistics = []
    results.sort(key=lambda row: row['group_id'])
    summary = {}
    for group in offline:
        summary[group] = {key: summarize([row[group][key] for row in results]) for key in results[0][group]}
    if pose_statistics:
        from evaluation.pose_fid import frechet_distance
        pose_statistics.sort(key=lambda row: row['group_id'])
        summary['motion']['L_PFID'] = {
            'value': frechet_distance([r['pred'] for r in pose_statistics], [r['gt'] for r in pose_statistics]),
            'n_clips': len(pose_statistics), 'n_frames': sum(r['pred']['n'] for r in pose_statistics),
        }
        (out / 'motion/pose_statistics.json').write_text(json.dumps(pose_statistics, indent=2))
    protocol = dict(bootstrap_replicates=BOOTSTRAP_REPLICATES, bootstrap_seed=BOOTSTRAP_SEED,
                    bootstrap_unit='clip; equal weight; finite applicable clips only',
                    bootstrap_rng='fresh seeded generator per objective metric; sorted clip IDs',
                    metrics=args.metrics, metric_groups=groups, test_ids=sorted(ids), pose_diagnostics=args.pose_diagnostics)
    if 'mllm' in groups:
        result = mllm_metrics(args, out, ids, options)
        summary['mllm'] = result['summary']
        by_id = {row['group_id']: row for row in result['per_clip']}
        for row in results:
            row['mllm'] = {key: value for key, value in by_id[row['group_id']].items() if key not in ['group_id', 'task']}
        protocol['mllm_bootstrap'] = result['bootstrap']
    results.sort(key=lambda row: row['group_id'])
    for group in groups:
        directory = out / group
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'per_clip.jsonl').write_text(''.join(json.dumps(
            dict(group_id=row['group_id'], task=row['task'], **row[group])) + '\n' for row in results))
        (directory / 'summary.json').write_text(json.dumps(summary[group], indent=2) + '\n')
    (out / 'per_clip.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in results))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
