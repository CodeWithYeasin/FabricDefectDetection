"""Gradio demo: fabric defect classification with Grad-CAM and spectral-residual maps."""
from __future__ import annotations

from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
from PIL import Image

from inference import FabricPredictor

HERE = Path(__file__).parent
predictor = FabricPredictor(HERE)
PRETTY = {"defect_free": "Defect-free", "hole": "Hole", "horizontal": "Horizontal defect",
          "lines": "Lines", "stain": "Stain", "vertical": "Vertical defect"}


def _overlay(img: Image.Image, heat: np.ndarray | None, alpha: float = 0.45):
    base = np.asarray(img.convert("RGB").resize((predictor.size, predictor.size)), dtype=np.float32) / 255
    if heat is None:
        return None
    h = matplotlib.colormaps["jet"](heat)[..., :3]
    return (np.clip((1 - alpha) * base + alpha * h, 0, 1) * 255).astype(np.uint8)


def run(img: Image.Image, tta: bool, threshold: float):
    if img is None:
        return None, "Upload an image.", None, None
    probs = predictor.predict(img, tta=tta)
    p_defect = 1.0 - probs["defect_free"]
    top = max(probs, key=probs.get)
    verdict = "DEFECT" if p_defect >= threshold else "OK"
    if verdict == "DEFECT" and top == "defect_free":  # defect probability spread over several types
        top = max((c for c in probs if c != "defect_free"), key=probs.get)
    text = (f"### {verdict}\n"
            f"P(defect) = **{p_defect:.1%}** (threshold {threshold:.0%})\n\n"
            f"Most likely type: **{PRETTY.get(top, top)}** ({probs[top]:.1%})")
    cam = _overlay(img, predictor.heatmap(img, predictor.classes.index(top)))
    spec = _overlay(img, predictor.spectral_map(img))
    return {PRETTY.get(k, k): v for k, v in probs.items()}, text, cam, spec


examples = sorted(str(p) for p in (HERE / "examples").glob("*") if p.suffix.lower() in {".jpg", ".png", ".jpeg"})

with gr.Blocks(title="Fabric Defect Detection") as demo:
    gr.Markdown(
        "# Fabric Defect Detection\n"
        "SpecTexNet: CNN appearance features fused with spectral-residual *periodicity-break* cues. "
        "Upload a close-up photo of woven fabric. Research demo, not a certified QC system."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Fabric image")
            tta = gr.Checkbox(value=True, label="Orientation-aware test-time augmentation")
            thr = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Defect decision threshold")
            btn = gr.Button("Analyse", variant="primary")
            if examples:
                gr.Examples(examples, inp)
        with gr.Column():
            verdict = gr.Markdown()
            label = gr.Label(num_top_classes=6, label="Class probabilities")
            with gr.Row():
                cam = gr.Image(label="Grad-CAM (predicted type)")
                spec = gr.Image(label="Spectral-residual saliency")
    btn.click(run, [inp, tta, thr], [label, verdict, cam, spec])
    inp.change(run, [inp, tta, thr], [label, verdict, cam, spec])

if __name__ == "__main__":
    demo.launch()
