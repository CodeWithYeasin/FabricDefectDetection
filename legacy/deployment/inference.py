
import pickle
import numpy as np
import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import sys

# --- 1. MODEL CLASSES (Must match training exactly) ---
def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

class TreeNode:
    def __init__(self, depth=0):
        self.depth = depth; self.left = None; self.right = None; self.feature_idx = None; self.threshold = None; self.value = None; self.is_leaf = False

class RegularizedTree:
    def __init__(self, max_depth=3, lambda_reg=1.0, gamma=0.0, min_child_weight=1.0, feature_subsample=1.0):
        self.max_depth = max_depth; self.lambda_reg = lambda_reg; self.gamma = gamma; self.root = None
    def predict(self, X): return np.array([self._traverse(x, self.root) for x in X])
    def _traverse(self, x, node):
        if node.is_leaf: return node.value
        if x[node.feature_idx] <= node.threshold: return self._traverse(x, node.left)
        else: return self._traverse(x, node.right)

class CustomModelBase:
    def predict_proba(self, X):
        if not hasattr(self, 'n_classes'): self.n_classes = len(self.trees[0])
        scores = np.zeros((X.shape[0], self.n_classes))
        for iter_trees in self.trees:
            for k, tree in enumerate(iter_trees):
                scores[:, k] += self.lr * tree.predict(X)
        return softmax(scores)

class CustomXGBoost(CustomModelBase):
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0):
        self.K = n_estimators; self.lr = learning_rate; self.trees = []

class CustomLightGBM(CustomModelBase):
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, goss_alpha=0.2, goss_beta=0.2):
        self.K = n_estimators; self.lr = learning_rate; self.trees = []

class CustomCatBoost(CustomModelBase):
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.K = n_estimators; self.lr = learning_rate; self.trees = []

class WeightStacker:
    def __init__(self, n_trials=2000):
        self.n_trials = n_trials; self.best_weights = None

# --- 2. CUSTOM UNPICKLER (THE FIX) ---
class ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect __main__ class lookups to the current script's global namespace
        if module == "__main__":
            if name in globals():
                return globals()[name]
        return super().find_class(module, name)

# --- 3. PREDICTOR CLASS ---
class FabricPredictor:
    def __init__(self, model_path='deployment_model.pkl'):
        print(f"Loading model from {model_path}...")

        # Use Custom Unpickler to load safely
        with open(model_path, 'rb') as f:
            data = ModelUnpickler(f).load()

        self.models = data['models']
        self.idx_to_class = data['idx_to_class']

        # Load ResNet50
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet50(weights="IMAGENET1K_V1")
        self.resnet.fc = nn.Identity()
        self.resnet.to(self.device)
        self.resnet.eval()

    def predict(self, img_pil):
        # 1. Preprocess
        img = img_pil.resize((224, 224)).convert('RGB')
        img_np = np.array(img, dtype=np.float32) / 255.0

        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std

        img_tensor = torch.tensor(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # 2. Extract Features
        with torch.no_grad():
            features = self.resnet(img_tensor).cpu().numpy()

        # 3. Base Predictions
        p1 = self.models['final_xgb'].predict_proba(features)
        p2 = self.models['final_lgb'].predict_proba(features)
        p3 = self.models['final_cat'].predict_proba(features)

        # 4. Ensemble Weighting
        w = self.models['final_stacker'].best_weights
        final_prob = (p1 * w[0]) + (p2 * w[1]) + (p3 * w[2])

        # 5. Result
        idx = np.argmax(final_prob)
        # Handle dict keys being strings or ints
        try:
            label = self.idx_to_class[idx]
        except KeyError:
            label = self.idx_to_class[str(idx)]

        conf = final_prob[0][idx]

        return label, conf, final_prob[0]
