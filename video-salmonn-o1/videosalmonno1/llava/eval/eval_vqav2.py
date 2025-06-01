import os
import argparse
import json
import re

from m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    results = []
    result_files = [os.path.join(result_file, sub_file) for sub_file in os.listdir(result_file)] if os.path.isdir(result_file) else [result_file]
    for cur_result_file in result_files:
        for line in open(cur_result_file):
            results.append(json.loads(line))
    annotations = json.load(open(annotation_file))['annotations']
    annotations = {annotation['question_id']: annotation for annotation in annotations}

    pred_list = []
    for result in results:
        annotation = annotations[result['question_id']]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": [x['answer'] for x in annotation['answers']],
        })

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, os.path.join(args.result_dir, args.result_file))
    else:
        assert args.result_dir is not None
        for result_file in sorted(os.listdir(args.result_dir)):
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))
