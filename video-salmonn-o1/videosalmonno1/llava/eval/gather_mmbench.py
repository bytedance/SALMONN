import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
import math


all_options = ['A', 'B', 'C', 'D']

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

def eval_results(args):
    questions = pd.read_table(os.path.expanduser(args.question_file))
    all_rounds_map = defaultdict(list)
    for _, row in questions.iterrows():
        base_index = row['index'] % 1000000
        all_rounds_map[base_index].append(row['index'])

    results = {}
    for line in open(os.path.expanduser(args.results_file)):
        row = json.loads(line)
        results[row['question_id']] = row

    stats = Counter()
    all_rounds_stats = defaultdict(list)
    correct_category = Counter()
    total_category = Counter()
    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        idx = row['index']
        base_index = row['index'] % 1000000
        gt_answer = row['answer']
        if idx in results:
            result = results[idx]
            all_rounds_stats[base_index].append(result['parsed_answer'] == gt_answer)
        else:
            all_rounds_stats[base_index].append(0)

    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        idx = row['index']
        base_index = row['index'] % 1000000
        if idx != base_index:
            continue
        all_rounds_results = all_rounds_stats[base_index]
        total_category[row['l2-category']] += 1
        stats['total'] += 1
        if len(all_rounds_map[base_index]) != len(all_rounds_results):
            stats['missing'] += 1
        else:
            if all(all_rounds_results):
                stats['correct'] += 1
                correct_category[row['l2-category']] += 1
            if any(all_rounds_results):
                stats['anycorrect'] += 1


    print(f'Correct: {stats["correct"]}, Incorrect: {stats["incorrect"]}, Missing: {stats["missing"]}, Total: {stats["total"]}')
    print(f'Accuracy: {stats["correct"] / stats["total"] * 100:.2f}%')
    print(f'Accuracy (Any): {stats["anycorrect"] / stats["total"] * 100:.2f}%')

    for category in total_category:
        print(f'{category}: {correct_category[category]}/{total_category[category]} = {correct_category[category] / total_category[category] * 100:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--results-file", type=str, default="results.jsonl")
    args = parser.parse_args()

    eval_results(args)
