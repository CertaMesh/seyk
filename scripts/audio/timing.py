#!/usr/bin/env python3
"""Generate synthetic decoder timing fixtures; ffmpeg and macOS afconvert required.
No media files are committed. Run the ignored core audio_timing test afterwards.
No-edit-list and no-LAME fixtures intentionally use the decoded-sample timeline.
"""
import json, subprocess, sys, wave, struct
from pathlib import Path
import numpy as np
ROOT=Path(sys.argv[1]).resolve(); ROOT.mkdir(parents=True,exist_ok=True)
rate=48000; n=rate*3
rng=np.random.default_rng(5901)
t=np.arange(n)/rate
x=.2*np.sin(2*np.pi*(300*t+700*t*t))+.05*rng.standard_normal(n)
x[:rate//4]=0; x[-rate//4:]=0
x=np.clip(x,-.9,.9)
def write(path,samples):
    with wave.open(str(path),'wb') as f:
        f.setparams((1,2,rate,0,'NONE','not compressed')); f.writeframes((samples*32767).astype('<i2').tobytes())
master=ROOT/'master.wav';write(master,x)
def ff(*args): subprocess.run(['ffmpeg','-v','error','-y',*map(str,args)],check=True)
records=[]
def add(name,args,reference='master.wav'):
    out=ROOT/name; ff('-i',master,*args,out);records.append(dict(file=name,reference=reference))
add('lossless.wav',['-c:a','pcm_s16le'])
add('lossless.flac',['-c:a','flac'])
add('lossless.m4a',['-c:a','alac'])
add('ffmpeg-aac.m4a',['-c:a','aac','-b:a','192k'])
add('lame.mp3',['-c:a','libmp3lame','-b:a','192k'])
add('no-lame.mp3',['-c:a','libmp3lame','-b:a','192k','-write_xing','0'],'no-lame-reference.wav')
ff('-i',ROOT/'no-lame.mp3','-c:a','pcm_s16le',ROOT/'no-lame-reference.wav')
add('no-edts.m4a',['-c:a','aac','-b:a','192k','-use_editlist','0'],'no-edts-reference.wav')
ff('-i',ROOT/'no-edts.m4a','-c:a','pcm_s16le',ROOT/'no-edts-reference.wav')
ff('-ss','0.5','-i',ROOT/'ffmpeg-aac.m4a','-c:a','copy',ROOT/'trimmed.m4a')
write(ROOT/'trimmed-reference.wav',x[rate//2:]);records.append(dict(file='trimmed.m4a',reference='trimmed-reference.wav'))
subprocess.run(['afconvert','-f','m4af','-d','aac','-b','192000',str(master),str(ROOT/'apple.m4a')],check=True)
records.append(dict(file='apple.m4a',reference='master.wav'))
ff('-i',master,'-i',master,'-map','0:a','-map','1:a','-c:a','aac',ROOT/'multi.m4a')
subprocess.run(['afconvert','-f','m4af','-d','aach','-b','32000',str(master),str(ROOT/'he-aac.m4a')],check=True)
ff('-i', master, '-t', '2.99973', '-c:a', 'pcm_s16le', ROOT/'fractional-reference.wav')
ff('-i', ROOT/'fractional-reference.wav', '-c:a', 'aac', '-b:a', '192k', ROOT/'fractional.m4a')
records.append(dict(file='fractional.m4a',reference='fractional-reference.wav'))
(ROOT/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print('TIMING FIXTURES',len(records))

# Identity/slicing edges; the timing correlation suite remains the ten codecs above.
for name, count, hz in [('one-sample',1,16000), ('short-tail',160001,16000), ('non16-tail',480013,48000)]:
    with wave.open(str(ROOT/(name+'.wav')),'wb') as f:
        f.setparams((1,2,hz,0,'NONE','not compressed'))
        f.writeframes(b''.join(struct.pack('<h',(i*7919)%32767) for i in range(count)))
