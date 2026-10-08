"""Explicit provisioning for this spike only. Never called by load or inference."""
import hashlib, json, pathlib, urllib.request
root = pathlib.Path(__file__).parent
models = root / 'models'
models.mkdir(exist_ok=True)
manifest=[]
for repo, revision in [
 ('PaddlePaddle/PP-OCRv5_mobile_det_onnx','e6f4fa85f00e168c862bc462aebca69eef9b3d3d'),
 ('PaddlePaddle/PP-OCRv5_mobile_rec_onnx','ed152b8b495f84de93cda5709d768548a9127622')]:
 kind='det' if '_det_' in repo else 'rec'
 for file in ['inference.onnx','inference.yml','README.md']:
  url=f'https://huggingface.co/{repo}/resolve/{revision}/{file}'
  data=urllib.request.urlopen(url).read()
  target=models / f'{kind}-{file}'
  target.write_bytes(data)
  manifest.append(dict(repo=repo,revision=revision,url=url,file=target.name,length=len(data),sha256=hashlib.sha256(data).hexdigest(),license='Apache-2.0 (publisher model card)'))
(root/'evidence'/'models-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
