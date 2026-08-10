"""
Export the trained FaceCNN to ONNX format so it can be run via OpenCV DNN
at inference time — no PyTorch needed during live recognition.

    python export_model.py

Creates:
  checkpoints/face_cnn.onnx   - ONNX model weights
  checkpoints/centroids.npy   - class centroid embeddings (numpy)
  checkpoints/classes.json    - class names (unchanged)
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from model import FaceCNN


def main():
    checkpoint_dir = "checkpoints"
    weights_path = os.path.join(checkpoint_dir, "face_cnn.pt")
    classes_path = os.path.join(checkpoint_dir, "classes.json")
    onnx_path = os.path.join(checkpoint_dir, "face_cnn.onnx")
    centroids_path = os.path.join(checkpoint_dir, "centroids.npy")

    if not os.path.exists(weights_path) or not os.path.exists(classes_path):
        raise RuntimeError("No trained model found. Run train_model.py first.")

    with open(classes_path) as f:
        classes = json.load(f)

    device = torch.device("cpu")
    checkpoint = torch.load(weights_path, map_location=device)

    model = FaceCNN(num_classes=len(classes)).to(device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        centroids = checkpoint["centroids"].numpy()
    else:
        model.load_state_dict(checkpoint)
        centroids = None

    model.eval()

    # Export only the feature extractor (embedding) part
    # Input: (1, 1, 96, 96) float32
    dummy_input = torch.zeros(1, 1, 96, 96, dtype=torch.float32)

    # Wrap extract_features so torch.onnx exports correctly
    class EmbeddingOnly(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):
            return self.base.extract_features(x)

    embed_model = EmbeddingOnly(model)
    embed_model.eval()

    # Use legacy export API to avoid onnxscript/numpy DLL issues on Windows
    import torch.onnx.utils as onnx_utils
    with torch.no_grad():
        onnx_utils._export(
            embed_model,
            (dummy_input,),
            onnx_path,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            opset_version=11,
            do_constant_folding=True,
        )
    print(f"Exported ONNX model -> {onnx_path}")


    # Save centroids as numpy array
    if centroids is not None:
        np.save(centroids_path, centroids)
        print(f"Saved centroids    -> {centroids_path}")
    else:
        print("WARNING: No centroids found in checkpoint. Run train_model.py again.")

    print(f"Classes: {classes}")
    print("Export complete. Run: python recognize_live.py --camera 1")


if __name__ == "__main__":
    main()
