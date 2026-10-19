---
title: Fabric Defect Detection
emoji: 🧵
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Fabric Defect Detection (SpecTexNet)

Classifies a fabric image into *defect-free, hole, horizontal, lines, stain, vertical* and shows
two explanation maps: Grad-CAM for the predicted class and the spectral-residual saliency that
highlights where the weave's periodicity breaks.

The weights are stored as `model.safetensors`, so loading them cannot execute code
(the v1 Space used a pickle, which the Hub flagged as unsafe).

To update this Space from the research repo:

```bash
python scripts/train.py --exp spectex_convnext_tiny --full
python scripts/export_model.py --run results/spectex_convnext_tiny/full
cd deploy/hf_space && git lfs track "*.safetensors" && git add . && git commit -m "update model" && git push
```
