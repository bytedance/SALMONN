"""Score frozen ELLSA2 videos through a configurable compatible API."""
import argparse
import base64
import concurrent.futures as futures
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release_paths import DATA_ROOT
from evaluation.activity import parse, quiet, task_prompt
from evaluation.judge_helpers import format_video_context, normalize_seed_response


def make_prompt(entry):
    return (
        task_prompt(entry['task']) + '\n' + format_video_context(entry['metadata'])
        + '\nAcoustic low-energy candidate windows, clip-relative seconds: '
        + json.dumps(entry['quiet_windows'])
        + '\nThese windows are measured from the actual target-only audio. Low energy alone is not proof of no speech: '
          'corroborate with listening. Outside these windows set Silence motion penalty evidence to none; '
          'Overactivity remains assessable. Partner words in the context can identify legitimate listening feedback; '
          'absence of target speech does not imply absence of partner speech.'
    )


def score_one(entry, repeat, client, args):
    destination = Path(args.output_dir) / f"{entry['group_id']}_{args.provider}_r{repeat}.json"
    videos = {'reference': entry['metadata']['reference_video'], 'candidate': entry['video']}
    prompt = make_prompt(entry)
    signature = dict(
        hashes={key: hashlib.sha256(Path(path).read_bytes()).hexdigest() for key, path in videos.items()},
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), model=args.model, provider=args.provider,
        base_url=args.base_url, azure_endpoint=args.azure_endpoint, video_format=args.video_format,
        api_version=args.api_version if args.azure_endpoint else None,
    )
    attempts = []
    if destination.exists():
        previous = json.loads(destination.read_text())
        # Older records used this exact Azure API default before it was recorded.
        previous['signature'].setdefault(
            'api_version', '2024-03-01-preview' if previous['signature'].get('azure_endpoint') else None)
        assert previous['signature'] == signature, f'Changed inputs for cached score: {destination}'
        if previous.get('valid'):
            assert previous['scores'] == parse(previous['response'], entry), 'Cached score differs from its response'
            return previous
        attempts = previous.get('attempts', [])

    content = [{'type': 'text', 'text': prompt}]
    for key, label in [('reference', 'VIDEO 1: HUMAN-VERIFIED GT REFERENCE'),
                       ('candidate', 'VIDEO 2: CANDIDATE TO SCORE')]:
        encoded = base64.b64encode(Path(videos[key]).read_bytes()).decode()
        if args.video_format == 'azure_image':
            media = {'type': 'image_url', 'image_url': {'url': encoded, 'fps': 5}}
        else:
            media = {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,' + encoded, 'fps': 5}}
        content.extend([{'type': 'text', 'text': label}, media])

    for attempt in range(args.retries):
        response = ''
        try:
            sampling = {'temperature': 0} if args.provider == 'gemini' else {}
            result = client.chat.completions.create(
                model=args.model, messages=[{'role': 'user', 'content': content}],
                max_tokens=30000, **sampling)
            response = result.choices[0].message.content.strip()
            if response.startswith('```'):
                response = response.split('\n', 1)[1].rsplit('```', 1)[0].strip()
            response = normalize_seed_response(response)
            record = dict(valid=True, group_id=entry['group_id'], task=entry['task'],
                          provider=args.provider, round=repeat, response=response,
                          scores=parse(response, entry), signature=signature,
                          quiet_windows=entry['quiet_windows'], returned_model=getattr(result, 'model', None),
                          attempts=attempts)
            destination.write_text(json.dumps(record, indent=2))
            return record
        except Exception as error:
            attempts.append({'error': repr(error), 'response': response})
            destination.write_text(json.dumps(dict(valid=False, signature=signature, attempts=attempts), indent=2))
            if attempt + 1 < args.retries:
                time.sleep(min(30, 5 * (attempt + 1)))
    raise RuntimeError(f'Failed judge request: {destination}')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--provider', choices=['gemini', 'seed'], required=True)
    parser.add_argument('--model', required=True, help='Gemini model name or Seed model/deployment ID')
    parser.add_argument('--base-url', default=os.environ.get('JUDGE_BASE_URL'))
    parser.add_argument('--azure-endpoint', default=os.environ.get('JUDGE_AZURE_ENDPOINT'))
    parser.add_argument('--api-version', default='2024-03-01-preview')
    parser.add_argument('--video-format', choices=['video_url', 'azure_image'],
                        help='Default: azure_image for Gemini, video_url for Seed')
    parser.add_argument('--api-key', help='Explicit API key; otherwise read --api-key-env')
    parser.add_argument('--api-key-env', default='JUDGE_API_KEY')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--max-pending', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--test-id', action='append', help='Judge these frozen IDs only (repeatable)')
    parser.add_argument('--retries', type=int, default=5)
    args = parser.parse_args(argv)
    if args.video_format is None:
        args.video_format = 'azure_image' if args.provider == 'gemini' else 'video_url'
    return args


def prepare_provider(args):
    assert args.rounds > 0 and 1 <= args.retries <= 5 and args.limit >= 0
    assert args.max_pending >= 1, 'Pending-job limit must be positive'
    if args.provider == 'gemini':
        assert args.max_pending <= 4, 'At most four submitted Gemini jobs at a time'
    assert bool(args.base_url) != bool(args.azure_endpoint), 'Configure exactly one API endpoint'

    import openai
    api_key = args.api_key or os.environ.get(args.api_key_env)
    assert api_key, f'Provide --api-key or set {args.api_key_env}'
    options = dict(api_key=api_key, timeout=180, max_retries=0)
    # Only the explicit loop above retries; hidden SDK retries would exceed five attempts.
    if args.azure_endpoint:
        client = openai.AzureOpenAI(azure_endpoint=args.azure_endpoint, api_version=args.api_version, **options)
    else:
        client = openai.OpenAI(base_url=args.base_url, **options)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entries = json.loads((DATA_ROOT / 'testset/judge_context.json').read_text())
    if args.test_id:
        assert set(args.test_id) <= {entry['group_id'] for entry in entries}, 'Unknown test ID'
        entries = [entry for entry in entries if entry['group_id'] in args.test_id]
    entries = entries[:args.limit or None]
    for entry in entries:
        entry['video'] = str(Path(args.video_dir).resolve() / (entry['group_id'] + '.mp4'))
        entry['metadata']['reference_video'] = str(DATA_ROOT / entry['metadata']['reference_video'])
        entry['quiet_windows'] = quiet(entry['video'])

    return client, entries


def run(args):
    client, entries = prepare_provider(args)
    # Bound this provider's submitted jobs, including queued futures.
    jobs = iter((entry, repeat) for entry in entries for repeat in range(1, args.rounds + 1))
    completed = 0
    with futures.ThreadPoolExecutor(max_workers=args.max_pending) as pool:
        pending = set()
        for _ in range(args.max_pending):
            job = next(jobs, None)
            if job is not None:
                pending.add(pool.submit(score_one, *job, client, args))
        while pending:
            done, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for future in done:
                future.result()
                completed += 1
                print(args.provider, completed, '/', len(entries) * args.rounds, flush=True)
                job = next(jobs, None)
                if job is not None:
                    pending.add(pool.submit(score_one, *job, client, args))
    (Path(args.output_dir) / f'{args.provider}_summary.json').write_text(json.dumps(dict(
        valid=completed, model=args.model, rounds=args.rounds, protocol='v10_activity'), indent=2))


def run_many(provider_args):
    """Run provider pools concurrently, each with its own submitted-job limit."""
    assert provider_args
    assert len({args.provider for args in provider_args}) == len(provider_args), 'Duplicate provider'
    with futures.ThreadPoolExecutor(max_workers=len(provider_args)) as pool:
        pending = [pool.submit(run, args) for args in provider_args]
        for future in futures.as_completed(pending):
            future.result()


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
