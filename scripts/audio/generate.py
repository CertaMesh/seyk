#!/usr/bin/env python3
"""Frozen synthetic audio retrieval sets. Requires macOS say, ffmpeg, numpy, soundfile.
Usage: python scripts/audio/generate.py heldout /tmp/seyk-audio-heldout
No recognizer is used. Timing annotations come only from synthesis placement.
"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import soundfile as sf

DATASET, DEST = sys.argv[1:]
assert DATASET in ('spike', 'heldout')
ROOT = Path(DEST).resolve(); ROOT.mkdir(parents=True, exist_ok=True)
SOURCE = Path(__file__).resolve().parents[2]/'tests/fixtures/audio'/f'{DATASET}.json'
ITEMS = json.loads(SOURCE.read_text())
recordings, queries = [], []
for i, (a, qa, b, qb) in enumerate(ITEMS):
    voice = ['Samantha', 'Daniel'][i % 2] if DATASET == 'spike' else 'Karen'
    rate = [155, 175, 195][i % 3]
    pcm = np.zeros(21*16000, dtype=np.float32)
    for j, (text, query, start) in enumerate([(a,qa,2), (b,qb,9), ('Please check the calendar after lunch.' if DATASET == 'spike' else 'Remember to bring your notebook tomorrow.',None,16)]):
        spoken = ROOT/f'say-{i}-{j}.wav'
        subprocess.run(['say','-v',voice,'-r',str(rate),'-o',str(spoken),'--file-format=WAVE','--data-format=LEI16@16000',text],check=True)
        samples, sr = sf.read(spoken,dtype='float32'); assert sr==16000 and samples.ndim==1
        active = np.flatnonzero(np.abs(samples)>0.003)
        if len(active): samples = samples[:min(len(samples),active[-1]+800)]
        offset = start*16000; assert offset+len(samples)<len(pcm)
        pcm[offset:offset+len(samples)] = samples
        if query: queries.append(dict(query=query,recording=i,start_ms=start*1000,end_ms=(offset+len(samples))/16,boundary=j==1))
        spoken.unlink()
    master=ROOT/f'{i:02}-source.wav'; sf.write(master,pcm,16000,subtype='PCM_16')
    formats = ['wav','flac','m4a','mp3'] if DATASET=='spike' else ['wav','flac','m4a','mp3','alac']
    fmt = formats[i%len(formats)]
    dest = ROOT/f'{i:02}.{"m4a" if fmt=="alac" else fmt}'
    opts = {'wav':['-c:a','pcm_s16le','-ar','16000','-ac','1'], 'flac':['-c:a','flac','-ar','44100','-ac','2'], 'm4a':['-c:a','aac','-b:a','96k','-ar','48000','-ac','2'], 'mp3':['-c:a','libmp3lame','-q:a','3','-ar','22050','-ac','1'], 'alac':['-c:a','alac','-ar','48000','-ac','2']}[fmt]
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(master),*opts,str(dest)],check=True)
    master.unlink()
    recordings.append(dict(id=i,path=str(dest),format=fmt,voice=voice,rate=rate,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),duration_ms=21000))
manifest=dict(dataset=DATASET,scripts_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),recordings=recordings,queries=queries)
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f'{DATASET}: {len(recordings)} files, {len(queries)} queries; scripts {manifest["scripts_sha256"]}')
