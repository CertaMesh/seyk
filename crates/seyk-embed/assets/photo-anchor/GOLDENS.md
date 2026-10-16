# Reference golden generation

Generated using Python 3.13.13 and NumPy 2.3.4 in an isolated uv environment.
Only deterministic synthetic vectors are used; these are arithmetic checks,
not photo retrieval quality measurements. Each vector has exactly unit norm:
256 entries of ±1/16, or a single entry of 1. Query uses seed 2.

The downloaded reference `scripts/quant4.py` was run unmodified from
pastel-labs/embeddinggemma-2-photo-anchor at
be9e066984345c92424c86bfcce3b17f3e3e8484, CC BY 4.0 by Pastel-Cloud OÜ.
Its SHA-256 is `161b5b9ea97ff6dbe60925483b4ca52d1470023bc72114832634d2b029dd8a52`.
All downloads used explicit pinned-revision resolve URLs and were first saved
under ignored `tmp/photo-anchor/`, preserving upstream paths. Binaries and the
reference script were checked against the pinned manifest's full SHA-256s.
The manifest hash was measured from that pinned revision (not provided by the
original brief). The 768 calibration binary matches the JSON offset and scale
arrays exactly as little-endian f32. The JSON has SHA-256
`5c17e0219137393c817fedc55e85e9cc93d461aa9c1e3b697206661f98e5e89b`.

Run from the repository root after explicitly provisioning those non-weight
files (no downloader is part of the codec):

```sh
uv venv tmp/photo-anchor-venv --python python3
uv pip install --python tmp/photo-anchor-venv/bin/python numpy==2.3.4
tmp/photo-anchor-venv/bin/python3 -I tmp/generate_photo_anchor.py
rm -rf tmp/photo-anchor-venv
```

Contents of `tmp/generate_photo_anchor.py`:

```python
import importlib.util, pathlib, json, os
import numpy as np
root=pathlib.Path.cwd()
source=root/'tmp/photo-anchor'
spec=importlib.util.spec_from_file_location('reference',source/'scripts/quant4.py')
reference=importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
os.chdir(source)
def vector(d, seed):
    v=np.zeros(d,dtype=np.float32)
    if seed==3:
        v[0]=1
    else:
        for i in range(256):
            v[(i*5+seed*11)%d]=np.float32(1/16 if (i*17+seed*13)%7<3 else -1/16)
    return v
rows=[]
for d in (256,768):
    a,o,s=reference.load('half-precision-model',d)
    q=vector(d,2)
    for seed in (0,1,3):
        x=vector(d,seed)[None,:]
        packed=reference.encode(x,a,o,s)
        score=float((reference.decode(packed,a,o,s)@q)[0])
        rows.append(dict(dim=d,seed=seed,codes=packed[0].tolist(),score=score))
output=root/'crates/seyk-embed/assets/photo-anchor/golden.json'
output.write_text(json.dumps(rows,separators=(',',':'))+'\n')
print('Generated',len(rows),'goldens;',output.stat().st_size,'bytes')
```
