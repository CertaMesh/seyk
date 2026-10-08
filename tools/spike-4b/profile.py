"""macOS-only external RSS sampling. No content, queries, or vectors leave the host."""
import json, math, pathlib, subprocess, sys, time
mode=sys.argv[1]
assert mode in ('render-only','embed')
root=pathlib.Path(__file__).parent
policy='(version 1)(allow default)(deny network*)'
samples=[]
with (root/'evidence'/f'{mode}-sampled.jsonl').open('w') as stdout, (root/'evidence'/f'{mode}-sampled.stderr').open('w') as stderr:
 start=time.monotonic()
 p=subprocess.Popen(['sandbox-exec','-p',policy,'target/release/seyk-4b-spike',mode],cwd=root,stdout=stdout,stderr=stderr)
 while p.poll() is None:
  result=subprocess.run(['ps','-o','rss=','-p',str(p.pid)],capture_output=True,text=True)
  if result.returncode==0 and result.stdout.strip():samples.append(int(result.stdout.strip())*1024)
  time.sleep(.05)
 elapsed=time.monotonic()-start
samples.sort()
result=dict(mode=mode,exit=p.returncode,wall_seconds=elapsed,sample_interval_ms=50,sample_count=len(samples),rss_p50_bytes=samples[math.ceil(len(samples)*.50)-1],rss_p95_bytes=samples[math.ceil(len(samples)*.95)-1],rss_sample_max_bytes=max(samples),note='RSS samples span process startup/model load and all page work; ps reports KiB on macOS; sampled maximum is not OS high-water mark')
(root/'evidence'/f'{mode}-rss-sampled.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
sys.exit(p.returncode)
