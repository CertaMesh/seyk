#!/usr/bin/env python3
"""Generate deterministic changing chirps, with bounded memory, for long-audio cost tests."""
import subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
for label,seconds in [('10min','600'),('old-cap','1048.576'),('2h','7200'),('over-cap','7260')]:
    wav=root/(label+'.wav')
    subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i',f'aevalsrc=0.2*sin(2*PI*(220*t+0.012*t*t)):s=16000:d={seconds}','-c:a','pcm_s16le',str(wav)],check=True)
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(wav),'-c:a','libmp3lame','-b:a','64k',str(root/(label+'.mp3'))],check=True)
