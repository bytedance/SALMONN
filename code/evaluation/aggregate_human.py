"""Reproduce the paper's two-annotator / two-judge descriptive correlations.

--trials maps blinded trial IDs to system, task, group_id and four MLLM scores.
--ratings supplies one final interface CSV export per annotator. Raw names and
free-text rating reasons are not copied into the output.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def final_ratings(rows):
    final = {}
    for row in rows:
        trial = row['trial_id']
        if trial not in final or row['updated_at'] >= final[trial]['updated_at']:
            final[trial] = row
    for row in final.values():
        assert row['status'] == 'submitted' and row['invalid'].lower() in {'false', '0', ''}
        score = float(row['score'])
        assert score.is_integer() and 1 <= score <= 5
    return {key: float(row['score']) for key, row in final.items()}


def correlations(values):
    # Columns: each judge, each annotator, then the corresponding two-person means.
    values = np.column_stack([values, values[:, :2].mean(1), values[:, 2:].mean(1)])
    labels = ['gemini', 'seed', 'human_1', 'human_2', 'mllm_mean', 'human_mean']
    result = {a + '__' + b: float(pearsonr(values[:, i], values[:, j]).statistic)
              for i, a in enumerate(labels) for j, b in enumerate(labels) if i < j}
    assert all(np.isfinite(value) for value in result.values()), 'Correlation needs varying ratings'
    result['mean_pairwise_mllm_human'] = float(np.mean([
        result[judge + '__' + human] for judge in labels[:2] for human in labels[2:4]]))
    return result


def aggregate(trials, ratings):
    assert len(ratings) == 2, 'The paper protocol has two annotators'
    ids = {row['trial_id'] for row in trials}
    assert len(ids) == len(trials) and all(set(rater) == ids for rater in ratings)
    identities = {(row['system'], row['group_id']) for row in trials}
    assert len(identities) == len(trials), 'Duplicate system/clip trial'
    output = {}
    for task in ['duplex', 'semantic']:
        rows = sorted([row for row in trials if row['task'] == task],
                      key=lambda row: (row['system'], row['group_id']))
        systems = sorted({row['system'] for row in rows})
        assert len(systems) >= 3, 'System correlation requires at least three systems'
        clip_sets = [{row['group_id'] for row in rows if row['system'] == system} for system in systems]
        assert all(clips == clip_sets[0] for clips in clip_sets), 'Systems must share the same clips'
        values = []
        for row in rows:
            machine = [float(row[f'{provider}_r{repeat}']) for provider in ['gemini', 'seed'] for repeat in [1, 2]]
            assert all(np.isfinite(v) and 0 <= v <= 1 for v in machine)
            values.append([np.mean(machine[:2]), np.mean(machine[2:]),
                           *[rater[row['trial_id']] for rater in ratings]])
        values = np.asarray(values)
        means = np.asarray([values[[r['system'] == system for r in rows]].mean(0) for system in systems])
        output[task] = dict(clips=len(clip_sets[0]), system_clip_pairs=len(rows),
                            individual_human_ratings=2 * len(rows),
                            sample_pearson=correlations(values), system_pearson=correlations(means),
                            systems={system: dict(zip(['gemini', 'seed', 'human_1', 'human_2',
                                                     'mllm_mean', 'human_mean'],
                                                      map(float, [*mean, mean[:2].mean(), mean[2:].mean()])))
                                     for system, mean in zip(systems, means)})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', required=True, help='CSV: trial_id,system,task,group_id,gemini_r1/r2,seed_r1/r2')
    parser.add_argument('--ratings', action='append', required=True, help='Final interface CSV; specify twice')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    exports = [read_csv(path) for path in args.ratings]
    raters = [{row['rater'] for row in rows} for rows in exports]
    assert all(len(rater) == 1 for rater in raters) and len(set.union(*raters)) == 2, 'Use two distinct annotators'
    result = aggregate(read_csv(args.trials), [final_ratings(rows) for rows in exports])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('Aggregated two annotators and two judges for both tasks')


if __name__ == '__main__':
    main()
