"""FabricNet: reproducible fabric defect classification research code.

Main entry points live in ``scripts/``. The package is split into:

- ``fabricnet.data``     dataset indexing, near-duplicate removal, leakage-free splits, transforms
- ``fabricnet.models``   SpecTexNet (proposed) and timm baselines
- ``fabricnet.engine``   training / inference loops
- ``fabricnet.metrics``  classification + calibration metrics
- ``fabricnet.stats``    bootstrap CIs and significance tests
- ``fabricnet.explain``  Grad-CAM and spectral saliency visualisation
- ``fabricnet.legacy``   the original (v1) frozen-ResNet50 + from-scratch GBDT pipeline, kept as a baseline
"""

__version__ = "2.0.0"

CLASSES = ["defect_free", "hole", "horizontal", "lines", "stain", "vertical"]
