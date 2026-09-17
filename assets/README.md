# README images

| File | Used in | Source |
| --- | --- | --- |
| `banner.png` | top of both READMEs | `pics/DataFlow-AgentMM.png`, downscaled and quantized |
| `pipeline.png` | `How it works` / `工作方式` | `pics/overview.png`, downscaled and quantized |

Both are 1600px-wide 256-colour PNGs. The full-resolution sources live in
`pics/` outside the package, together with `pics/PIPELINE_PROMPT.md`, which
records the prompts and the structural rules the overview diagram follows.

Regenerate either one the same way after replacing its source:

```python
from PIL import Image

source = Image.open("pics/overview.png").convert("RGB")          # or DataFlow-AgentMM.png
scaled = source.resize((1600, round(source.height * 1600 / source.width)), Image.LANCZOS)
scaled.quantize(colors=256).save("dataflow-mm-agent/assets/pipeline.png", optimize=True)
```

Keep the alt text on both images: it is what readers get when images fail to
load.
