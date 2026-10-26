#!/usr/bin/env python3
"""Regenerate local-only STT gates (never redistribute System Voice output) (macOS say + FFmpeg; no automatic downloads).
uv run --no-project --with numpy==2.5.3 --with torch==2.14.1 python scripts/audio/stt_fixture.py ~/.cache/seyk-spike/whisper-tiny-xenova/reference
Uses OpenAI Whisper 86098128c0b4f24f0e2aa2994de830614b474227.
"""
import gzip, hashlib, json, subprocess, sys, tempfile, types, wave
from pathlib import Path
import numpy as np
import torch
reference=Path(sys.argv[1]);audio_file=reference/'whisper/audio.py'
assert hashlib.sha256(audio_file.read_bytes()).hexdigest()=='36ed287b2de3eaf40207cd2a44a8d45d57ea5c212a39df07401d4fcb4d5a5a1a'
assert hashlib.sha256((reference/'whisper/assets/mel_filters.npz').read_bytes()).hexdigest()=='7450ae70723a5ef9d341e3cee628c7cb0177f36ce42c44b7ed2bf3325f0f6d4c'
manifest=json.loads(Path('crates/seyk-stt/assets/manifest.json').read_text())
mel=np.load(reference/'whisper/assets/mel_filters.npz')['mel_80'].astype('<f4').tobytes()
assert hashlib.sha256(mel).hexdigest()==manifest['mel']['sha256']
Path('crates/seyk-stt/assets/mel-80.f32le').write_bytes(mel)
ref=types.ModuleType('reference_audio');ref.__file__=str(audio_file)
exec(compile(audio_file.read_text().replace('from .utils import exact_div','def exact_div(x,y):\n    assert x%y==0\n    return x//y'),str(audio_file),'exec'),ref.__dict__)
torch.set_num_threads(1)
out=Path('crates/seyk-stt/tests/fixtures');out.mkdir(parents=True,exist_ok=True)
t=np.arange(16000)/16000
samples=(0.2*np.sin(2*np.pi*(170*t+600*t*t))+0.1*np.cos(2*np.pi*3500*t)).astype('<f4')
samples[0]=0.75;samples[-1]=-0.65
features=ref.log_mel_spectrogram(np.pad(samples,(0,480000-len(samples)))).numpy().astype('<f4')
(out/'mel-input.f32le').write_bytes(samples.tobytes())
(out/'mel-reference.f32le.gz').write_bytes(gzip.compress(features.tobytes(),mtime=0))
text='The morning train arrives at the station. Please bring a blue notebook and a pencil. The library opens at nine. Today we will walk through the garden and look at the flowers. A small bird is singing near the window. After lunch we will read a story together.'
# Locally generated speech is for personal evaluation only: never commit or redistribute it.
Path('tmp').mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir='tmp',prefix='stt-fixture-') as tmp:
 aiff=Path(tmp)/'speech.aiff';wav=Path(tmp)/'speech.wav'
 subprocess.run(['say','-v','Karen','-r','160','-o',str(aiff),text],check=True)
 subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(aiff),'-ar','16000','-ac','1','-c:a','pcm_s16le',str(wav)],check=True)
 with wave.open(str(wav)) as w:pcm=w.readframes(w.getnframes())
 with wave.open(str(Path('tmp')/'stt-speech.wav'),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(pcm)
 for seconds in [10,30]:
  base=pcm+b'\0'*(8000*2);data=(base*((seconds*32000+len(base)-1)//len(base)))[:seconds*32000]
  with wave.open(f'tmp/stt-{seconds}.wav','wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(data)
# Attest only the mel outputs written above. Frozen decoder captures have separate provenance.
meta=dict(speech_storage="Generated at test time in ignored tmp/stt-speech.wav; never commit or redistribute System Voice output",text=text,voice='Karen en_AU',rate=160,reference_commit='86098128c0b4f24f0e2aa2994de830614b474227',torch=torch.__version__,numpy=np.__version__,mel_tolerance='max absolute error <= 0.0001 across all 240000 floats',sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('mel-input.f32le','mel-reference.f32le.gz')})
(out/'provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
