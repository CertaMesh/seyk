"""Independent stdlib implementation of the documented binary layout (synthetic only)."""
import base64
import json
from pathlib import Path
from urllib.parse import quote


def var(n):
    out = bytearray()
    while n >= 128:
        out.append((n & 127) | 128)
        n >>= 7
    return bytes(out + bytes([n]))


def string(s):
    raw = s.encode('utf-8')
    return var(len(raw)) + raw


def optional(n):
    return b'\x00' if n is None else b'\x01' + var(n)


def encode(s, omit):
    loc, meta = s['locator'], s['metadata']
    raw = b'SR\x01' + bytes([omit])
    if not omit:
        raw += string(s['resource'])
    kind = loc['kind']
    if kind == 'text':
        raw += b'\x00' + var(loc['byte_start']) + var(loc['byte_end']) + optional(loc['line_start']) + optional(loc['line_end'])
    elif kind == 'page_text':
        raw += b'\x01' + var(loc['page']) + var(loc['byte_start']) + var(loc['byte_end']) + string(loc['text_version']) + bytes.fromhex(loc['page_text_hash'])
    elif kind == 'page':
        raw += b'\x02' + var(loc['page'])
    elif kind == 'time':
        raw += b'\x03' + var(loc['start_ms']) + var(loc['end_ms'])
    elif kind == 'image_region':
        raw += b'\x04' + b''.join(var(loc[k]) for k in ['x', 'y', 'width', 'height'])
    else:
        assert kind == 'whole_resource'
        raw += b'\x05'
    raw += bytes([['document', 'text', 'image', 'audio', 'video'].index(meta['modality'])])
    raw += var(meta['size_bytes'])
    mtime = meta['modified_ns']
    raw += optional(None if mtime is None else (mtime << 1) ^ (mtime >> 63))
    media = meta.get('media_type')
    raw += b'\x00' if media is None else b'\x01' if media == 'application/pdf' else b'\x02' + string(media)
    raw += bytes.fromhex(meta['content_hash'])
    return raw


def vectors():
    locators = [
        {'kind': 'whole_resource'},
        {'kind': 'text', 'byte_start': 0, 'byte_end': 128, 'line_start': 1, 'line_end': 127},
        {'kind': 'page_text', 'page': 128, 'byte_start': 127, 'byte_end': 16384, 'text_version': 'unknown-文-v1', 'page_text_hash': 'ff' * 32},
        {'kind': 'page', 'page': 4294967295},
        {'kind': 'time', 'start_ms': 0, 'end_ms': 18446744073709551615},
        {'kind': 'image_region', 'x': 0, 'y': 127, 'width': 128, 'height': 4294967295},
        {'kind': 'text', 'byte_start': 127, 'byte_end': 128, 'line_start': None, 'line_end': None},
        {'kind': 'text', 'byte_start': 0, 'byte_end': 0, 'line_start': 0, 'line_end': 0},
    ]
    rows = []
    for i, loc in enumerate(locators):
        meta = {'modality': ['text', 'document', 'image', 'audio', 'video'][i % 5],
                'size_bytes': [0, 127, 128, 16383, 16384, 18446744073709551615][i % 6],
                'modified_ns': [None, 0, -1, -9223372036854775808, 9223372036854775807][i % 5],
                'content_hash': ('00' * 32 if i == 0 else bytes(range(32)).hex())}
        media = [None, 'application/pdf', '', 'audio/custom-文'][i % 4]
        if media is not None:
            meta['media_type'] = media
        for omit in [True, False]:
            # Explicit URI vectors exercise non-file and non-UTF8 lossless identity too.
            uri = 'file:///fixture/a' if omit else ['file:///fixture/%FF', 'https://example.org/fixture'][i % 2]
            s = {'resource': uri, 'locator': loc, 'metadata': meta}
            raw = encode(s, omit)
            rows.append({'name': f'{i}-{loc["kind"]}-{"path" if omit else "uri"}',
                         'selection': s, 'path': '/fixture/a' if omit else '/edited/ignored',
                         'omit_resource': omit, 'valid_selection': i != 7,
                         'hex': raw.hex(), 'ref': 'seyk-ref-v1:' + base64.urlsafe_b64encode(raw).decode().rstrip('=')})
    path = '/fixture/文 café/😀.txt'
    selection = {'resource': 'file://' + quote(path), 'locator': locators[0],
                 'metadata': rows[0]['selection']['metadata']}
    raw = encode(selection, True)
    rows.append({'name': 'unicode-omitted-uri', 'selection': selection,
                 'path': path, 'omit_resource': True, 'valid_selection': True,
                 'hex': raw.hex(), 'ref': reference(raw)})
    return rows


def reference(raw):
    return 'seyk-ref-v1:' + base64.urlsafe_b64encode(raw).decode().rstrip('=')


def negative_vectors():
    seed = bytes.fromhex(vectors()[0]['hex'])
    cases = []
    def add(name, raw, ref=None, path='/fixture/a'):
        cases.append({'name': name, 'hex': raw.hex(),
                      'ref': reference(raw) if ref is None else ref, 'path': path})
    for n in range(len(seed)):
        add(f'truncated-{n}', seed[:n])
    for name, index, value in [('domain', 0, 0), ('version', 2, 2),
                                ('flags', 3, 2), ('locator', 4, 255),
                                ('modality', 5, 255), ('option', 7, 2),
                                ('media', 8, 255)]:
        raw = bytearray(seed)
        raw[index] = value
        add(name, raw)
    add('trailing-byte', seed + b'\x00')
    add('overlong-size', seed[:6] + b'\x80\x00' + seed[7:])
    add('overflow-size', seed[:6] + b'\xff' * 10 + seed[7:])
    add('noncanonical-pdf-media', seed[:8] + b'\x02' + string('application/pdf') + seed[9:])
    add('invalid-utf8-media', seed[:8] + b'\x02\x01\xff' + seed[9:])
    add('missing-path', seed, path=None)
    add('relative-path', seed, path='relative')
    good = reference(seed)
    for name, ref in [('padding', good + '='), ('whitespace', good + '\n'),
                      ('outer-version', good.replace('-v1:', '-v2:')),
                      ('invalid-base64', good + '!'),
                      ('unused-base64-bits', good[:-1] + 'B')]:
        add(name, seed, ref)
    return cases


if __name__ == '__main__':
    dest = Path(__file__).resolve().parents[2] / 'tests/fixtures/mcp-refs.json'
    dest.write_text(json.dumps(vectors(), indent=2, ensure_ascii=False) + '\n')
    dest.with_name('mcp-refs-invalid.json').write_text(json.dumps(negative_vectors(), indent=2, ensure_ascii=False) + '\n')
