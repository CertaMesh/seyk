"""Freeze scored raw search results locally; never commit the output manifest.

python3 scripts/bench/mcp_payload_manifest.py /path/to/runs/main /private/manifest.json
Use --synthetic OUTPUT for a public, invented ten-hit invocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def text_content(body):
    if isinstance(body, str):
        return body
    if isinstance(body, list):
        return ''.join(x.get('text', '') for x in body if isinstance(x, dict))
    return ''


def claude_calls(events):
    pending, seen = {}, set()
    for event in events:
        for block in event.get('message', {}).get('content', []) if isinstance(event.get('message', {}).get('content'), list) else []:
            if block.get('type') == 'tool_use' and block.get('name') == 'mcp__seyk__search':
                pending[block['id']] = block
            if block.get('type') == 'tool_result' and block.get('tool_use_id') in pending:
                call = block['tool_use_id']
                if call not in seen:
                    seen.add(call)
                    yield call, text_content(block.get('content')), bool(block.get('is_error'))


def codex_calls(events):
    seen = set()
    for event in events:
        item = event.get('item', {})
        if event.get('type') != 'item.completed' or item.get('type') != 'mcp_tool_call' or item.get('server') != 'seyk' or item.get('tool') != 'search':
            continue
        if item['id'] in seen:
            continue
        seen.add(item['id'])
        result = item.get('result') or {}
        yield item['id'], text_content(result.get('content')), bool(item.get('status') == 'failed' or item.get('error') or result.get('isError'))


def synthetic():
    from urllib.parse import quote
    calls = []
    locators = [
        ('text', {'kind': 'text', 'byte_start': 0, 'byte_end': 128, 'line_start': 1, 'line_end': 4}),
        ('page_text', {'kind': 'page_text', 'page': 1, 'byte_start': 0, 'byte_end': 128, 'text_version': 'pdf-extract-0.12.1;plain-v1', 'page_text_hash': '11' * 32}),
        ('page', {'kind': 'page', 'page': 1}),
        ('time', {'kind': 'time', 'start_ms': 0, 'end_ms': 1000}),
        ('image_region', {'kind': 'image_region', 'x': 0, 'y': 0, 'width': 128, 'height': 128}),
        ('whole_resource', {'kind': 'whole_resource'}),
    ]
    for kind, locator in locators:
        for inline in [False, True]:
            hits = []
            for i in range(10):
                path = f'/fixture/{"long-" * 20 if inline else ""}文-{i}.txt'
                uri = 'file://' + quote(path) if not inline else 'file:///fixture/%FF-' + str(i)
                hit = {'resource': uri, 'locator': locator,
                       'metadata': {'modality': {'page': 'document', 'page_text': 'document', 'time': 'audio', 'image_region': 'image', 'whole_resource': 'image'}.get(kind, 'text'),
                                    'media_type': 'application/pdf' if kind in ['page', 'page_text'] else 'text/plain',
                                    'size_bytes': 4096, 'modified_ns': 1720000000000000000, 'content_hash': '22' * 32},
                       'path': path, 'heading': 'Invented fixture', 'score': .8,
                       'preview': 'Public invented evidence. ' * 4, 'matched': ['meaning'],
                       'match': {'semantic_rank': 1, 'keyword_rank': None, 'filename_rank': None, 'matched_terms': []}, 'matched_query': 'invented fixture'}
                if kind == 'text': hit['lines'] = '1-4'
                if kind in ['page', 'page_text']: hit['page'] = 1
                hits.append(hit)
            response = {'schema_version': 10, 'hits': hits, 'index_progress': [], 'segments_searched': 10, 'files': 10, 'skipped': [], 'page_diagnostics': [], 'page_diagnostics_total': 0, 'index_diagnostics': {'skipped': [], 'page_diagnostics': [], 'pages_total': 0}, 'segments_embedded': 0, 'took_ms': 1, 'index': {'state': 'ready'}}
            raw = json.dumps(response, ensure_ascii=False, separators=(',', ':'))
            calls.append({'task': f'synthetic-{kind}-{inline}', 'agent': 'synthetic', 'arm': kind, 'cond': 'uri' if inline else 'path', 'call_id': '1', 'raw': raw, 'raw_sha256': digest(raw.encode()), 'is_error': False})
    warning = dict(calls[0])
    warning.update(task='synthetic-query-warnings', cond='warnings')
    warning['query_sync'] = {'resources_discovered': 2, 'resources_unchanged': 0,
                             'resources_updated': 0, 'segments_embedded': 0, 'embed_ms': 0,
                             'skipped_counts': [],
                             'skipped': [{'resource': 'file:///fixture/omitted-image.png', 'reason': 'invented missing image model'}] * 2,
                             'page_diagnostics': [{'resource': 'file:///fixture/omitted-page.pdf', 'page': 1, 'reason': 'invented page error'}] * 2}
    calls.append(warning)
    return {'format': 'seyk-payload-manifest-v1', 'synthetic': True, 'sources': [], 'calls': calls, 'tasks': len(calls), 'task_records': [{k: call[k] for k in ['task', 'agent', 'arm', 'cond']} for call in calls], 'reconciliation': {}}


def freeze(root):
    sources, calls, tasks, reconciliation, task_records = [], [], set(), {}, []
    def source(path):
        raw = path.read_bytes()
        sources.append({'path': str(path.resolve()), 'sha256': digest(raw)})
        return raw
    records = [json.loads(line) for line in source(root / 'results.jsonl').decode().splitlines()]
    labels = json.loads(source(root / 'labels.json'))
    seen = set()
    for record in records:
        task = '-'.join(str(record[key]) for key in ['agent', 'arm', 'cond', 'id'])
        if task not in labels or record['cond'] not in ['base', 'seyk'] or record['arm'] not in ['code', 'notes']:
            raise ValueError('unscored condition requires explicit reconciliation')
        if task in tasks:
            raise ValueError('duplicate scored task requires explicit reconciliation')
        tasks.add(task)
        task_records.append({'task': task, **{k: record[k] for k in ['agent', 'arm', 'cond']}})
        transcript = source(root / (task + '.jsonl'))
        events = [json.loads(line) for line in transcript.decode().splitlines()]
        parser = claude_calls if record['agent'] == 'claude' else codex_calls
        selected = list(parser(events))
        reconciliation[task] = {'recorded_summary_calls': len(record['seyk_calls']), 'raw_calls': len(selected)}
        for call_id, raw, is_error in selected:
            key = (task, call_id)
            if key in seen: raise ValueError('duplicate accepted call')
            seen.add(key)
            calls.append({'task': task, 'agent': record['agent'], 'arm': record['arm'], 'cond': record['cond'], 'call_id': call_id, 'raw': raw, 'raw_sha256': digest(raw.encode()), 'is_error': is_error})
    return {'format': 'seyk-payload-manifest-v1', 'synthetic': False, 'sources': sources, 'calls': calls, 'tasks': len(tasks), 'task_records': task_records, 'reconciliation': reconciliation,
            'historical_claude_calls_expected': 34, 'inclusion': 'Exactly tasks in supplied results.jsonl and labels.json; no other directories; unique completed tool-result IDs. Review reconciliation before interpreting size results.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('paths', nargs='+', type=Path)
    args = parser.parse_args()
    if args.synthetic:
        if len(args.paths) != 1: parser.error('--synthetic takes OUTPUT only')
        manifest, output = synthetic(), args.paths[0]
    else:
        if len(args.paths) != 2: parser.error('provide scored directory and OUTPUT')
        manifest, output = freeze(args.paths[0]), args.paths[1]
    raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as dest: dest.write(raw)
    print('manifest_sha256=' + digest(raw))
    print('tasks=' + str(manifest['tasks']) + ' calls=' + str(len(manifest['calls'])))
    print('Pass the manifest path and its SHA-256 to the mcp_payload_sizes example. Keep manifests private.')


if __name__ == '__main__':
    main()
