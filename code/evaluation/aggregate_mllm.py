"""Average four individually penalized judgments, then equally weight clips."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_paths import DATA_ROOT
from evaluation.activity import parse
from evaluation.statistics import BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, summarize


def aggregate(records, entries):
    scores = {}
    for record in records:
        if record.get('valid') is not True:
            continue
        key = (record['group_id'], record['provider'], record['round'])
        assert key not in scores, f'Duplicate score: {key}'
        score = record['scores']['final']
        assert np.isfinite(score) and 0 <= score <= 1, key
        scores[key] = record
    per_clip = []
    for entry in sorted(entries, key=lambda row: row['group_id']):
        gid = entry['group_id']
        keys = [(gid, provider, repeat) for provider in ['gemini', 'seed'] for repeat in [1, 2]]
        assert all(key in scores for key in keys), f'Incomplete four-judge coverage: {gid}'
        selected = [scores[key] for key in keys]
        assert all(row['task'] == entry['task'] for row in selected), f'Wrong task: {gid}'
        # New records must agree on the actual videos and complete request text.
        signatures = [row.get('signature') for row in selected]
        if any(signatures):
            assert all(signatures), f'Mixed versioned and legacy records: {gid}'
            identities = [(s['hashes'], s['prompt_sha256']) for s in signatures]
            assert all(identity == identities[0] for identity in identities), f'Mixed judge inputs: {gid}'
        values = []
        for row in selected:
            context = dict(entry, quiet_windows=row.get('quiet_windows', entry.get('quiet_windows', [])))
            value = parse(row['response'], context)['final']
            assert abs(value - row['scores']['final']) < 1e-12, f'Changed stored score: {gid}'
            values.append(value)
        per_clip.append(dict(group_id=gid, task=entry['task'], score=float(np.mean(values)),
                             gemini_mean2=float(np.mean(values[:2])), seed_mean2=float(np.mean(values[2:])),
                             gemini_repeat_difference=abs(values[0] - values[1]),
                             seed_repeat_difference=abs(values[2] - values[3])))
    summary = {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for task in ['duplex', 'semantic']:
        rows = [row for row in per_clip if row['task'] == task]
        if not rows:
            summary[task] = dict(n=0, mean=None, ci95=None)
            continue
        # Share clip draws across judges; resampling individual requests changes the estimand.
        indices = rng.integers(len(rows), size=(BOOTSTRAP_REPLICATES, len(rows)))
        value = summarize([row['score'] for row in rows], indices=indices)
        value['judges'] = {provider: summarize([row[provider + '_mean2'] for row in rows], indices=indices)
                           for provider in ['gemini', 'seed']}
        value['mean_absolute_repeat_difference'] = {
            provider: float(np.mean([row[provider + '_repeat_difference'] for row in rows]))
            for provider in ['gemini', 'seed']}
        summary[task] = value
    return dict(summary=summary, per_clip=per_clip,
                bootstrap=dict(replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                               unit='clip ID; conditional on the recorded two requests per judge'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--test-id', action='append', help='Selected clips; default requires all 138')
    args = parser.parse_args()
    entries = json.loads((DATA_ROOT / 'testset/judge_context.json').read_text())
    if args.test_id:
        assert set(args.test_id) <= {row['group_id'] for row in entries}, 'Unknown test ID'
        entries = [row for row in entries if row['group_id'] in args.test_id]
    records = (json.loads(path.read_text()) for path in Path(args.score_dir).rglob('*.json'))
    result = aggregate(records, entries)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
