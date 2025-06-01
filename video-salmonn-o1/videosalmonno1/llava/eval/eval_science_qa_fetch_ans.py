import argparse
import os
import json
import pandas as pd
from tqdm import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time


all_options = ['A', 'B', 'C', 'D', 'E']


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_row(df, colname, value):
    assert (df[colname] == value).sum() == 1
    return df[df[colname] == value].iloc[0]


def encode_query(question, options, answer):
    query = ""
    query += "Question: " + question + "\n"
    query += "Options: " + "\n".join([f"{option_char}. {option}" for option_char, option in zip(all_options[:len(options)], options)]) + "\n"
    query += "Answer: " + answer + "\n"
    return query


def get_openai_api():
    api_type = os.environ.get('API_TYPE', 'azure')

    if api_type == 'azure':
        api_key = os.environ.get('API_KEY', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        engine = os.environ.get('ENGINE', 'chatgpt-turbo')
        api_host = os.environ.get('API_BASE')
        return {
            'api_type': 'azure',
            'api_version': '2023-06-01-preview',
            'engine': engine,
            'api_key': api_key,
            'api_base': f'https://{api_host}.openai.azure.com',
        }
    else:
        api_key = os.environ.get('API_KEY', 'sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        model = os.environ.get('MODEL', 'gpt-3.5-turbo-0301')

        return {
            'model': model,
            'api_key': api_key,
        }


def chatgpt_extract_answer(
    question, options, answer, max_tokens=64, temperature=0.2, top_p=0.9, frequency_penalty=0, presence_penalty=0,
    request_timeout=None, num_retry=1):
    api_kwargs = get_openai_api()

    system_message = """You are an AI assistant to help me matching an answer with several options of a multiple choice question.
You are provided with a question, several options, and an answer, and you need to find which option is most similar to the answer.
If the meaning of all options are significantly different from the answer, output X.
You should output a single uppercase character in A, B, C, D, E, if they are valid options, and X otherwise."""
    exemplers = [
        {
            "question": "What is the main object in image?",
            "options": ["teddy bear", "rabbit", "cat", "dog"],
            "answer": "a cute teddy bear",
            "output": "A",
        },
        {
            "question": "What is the main object in image?",
            "options": ["teddy bear", "rabbit", "cat", "dog"],
            "answer": "Spider",
            "output": "X",
        },
    ]

    messages = [
        {"role": "system", "content": system_message},
    ]
    for exempler in exemplers:
        messages.append({"role": "user", "content": encode_query(exempler['question'], exempler['options'], exempler['answer'])})
        messages.append({"role": "assistant", "content": exempler['output']})
    messages.append({"role": "user", "content": encode_query(question, options, answer)})

    response = None
    attempts = []
    for i in range(num_retry):
        try:
            response = openai.ChatCompletion.create(
                messages = messages,
                max_tokens = max_tokens,
                temperature = temperature,
                top_p = top_p,
                frequency_penalty = frequency_penalty,
                presence_penalty = presence_penalty,
                request_timeout = request_timeout,
                **api_kwargs
            )
        except Exception as e:
            if type(e) in [openai.error.RateLimitError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout]:
                pass
            elif type(e) in [openai.error.AuthenticationError, openai.error.InvalidRequestError]:
                print(e)
                return None
            else:
                print(type(e), e)
            attempts.append(e.__class__.__name__)
            time.sleep(1)
        else:
            time.sleep(1)
            break

    if response is None:
        print(f'All {num_retry} attempts failed: {attempts}. Returning None.')
        return None

    try:
        content = response['choices'][0]['message']['content']
    except KeyError as e:
        print(f'KeyError: {e}. Returning None.')
        return None
    content = content.strip()
    return content

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
    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))

    answers = [json.loads(line) for line in open(os.path.expanduser(args.answers_file))]
    answers = {row['question_id']: row for row in answers}
    results_file = os.path.expanduser(args.results_file)
    if os.path.exists(results_file):
        results = [json.loads(line) for line in open(results_file)]
        results = {row['question_id']: row for row in results}
    else:
        results = {}
    results_writer = open(results_file, 'a')

    def process_answer(question_id, answer):
        if question_id in results:
            return None
        question_data = problems[question_id]
        options = question_data['choices']
        parsed_answer = chatgpt_extract_answer(
            question_data['question'], options, answer['text'],
            request_timeout=args.request_timeout, num_retry=args.num_retry)
        if parsed_answer is None:
            return None
        valid_ops = all_options[:len(options)] + ['X']
        if parsed_answer not in valid_ops:
            print(f'Invalid parsed answer: {parsed_answer}')
            return None
        answer['original_answer'] = answer['text']
        answer['text'] = f'The answer is {parsed_answer}.'
        return answer

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(process_answer, key, value): key for key, value in answers.items()}

        # Process results as they become available
        for future in tqdm(as_completed(futures), total=len(answers)):
            answer = future.result()
            if answer is not None:
                results_writer.write(json.dumps(answer) + '\n')
                results_writer.flush()

    results_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--results-file", type=str, default="results.jsonl")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--num-retry", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=None)
    args = parser.parse_args()

    eval_results(args)
