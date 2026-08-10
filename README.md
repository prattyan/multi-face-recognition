# Multi-Face Recognition System (Trained From Scratch)

A real-time, multi-face recognition pipeline built without any pretrained
identity models. Both detection and recognition use **OpenCV DNN** at
inference time — no PyTorch runtime needed during live recognition.

---

## 1. Architecture overview

```mermaid
flowchart TD
    A["📷 Webcam Frame"] --> B

    B["🔍 OpenCV DNN Face Detector\nSSD ResNet-10  —  face_detector.py\nmodels/deploy.prototxt\nmodels/res10_300x300_ssd_iter_140000_fp16.caffemodel"]
    B --> |"bounding boxes\nx, y, w, h"| C

    C["✂️ Crop + Resize Each Face\n96×96 grayscale · histogram equalized"]
    C --> D

    D["🧠 FaceCNN  —  ONNX via cv2.dnn\ncheckpoints/face_cnn.onnx\ncheckpoints/centroids.npy\ntrained from scratch on YOUR captured images"]
    D --> |"128-D embedding"| E

    E["📐 Cosine Similarity\nvs. per-class centroids"]
    E --> F{{"best_sim ≥ threshold?"}}

    F --> |"YES"| G["✅ Person Name\nconfidence %"]
    F --> |"NO"| H["❌ Unknown"]

    style B fill:#1a3a5c,stroke:#4a9eff,color:#fff
    style C fill:#1a3a2c,stroke:#4aff9e,color:#fff
    style D fill:#3a1a5c,stroke:#c44aff,color:#fff
    style E fill:#3a2a1a,stroke:#ffaa4a,color:#fff
    style G fill:#1a4a1a,stroke:#4aff4a,color:#fff
    style H fill:#4a1a1a,stroke:#ff4a4a,color:#fff
```

**Two independent stages:**

| Stage | Component | File | Nature |
|:---|:---|:---|:---|
| Detection ("where is a face?") | OpenCV DNN SSD ResNet-10 | `face_detector.py` | Pre-trained face localizer — NOT trained on identities |
| Recognition ("whose face is it?") | FaceCNN → ONNX → cv2.dnn | `model.py`, `recognize_live.py` | Trained 100% from scratch on your captured data |

---

## 2. End-to-end workflow (4 steps)

```
Step 1         Step 2         Step 3          Step 4
capture        train          export          recognize
_face.py  →  train_model  →  export_model  →  recognize
  ↓            .py              .py             _live.py
dataset/      checkpoints/    checkpoints/    Live camera
<name>/       face_cnn.pt     face_cnn.onnx   with labels
*.png         centroids       centroids.npy
              classes.json    classes.json
```

> ⚠️ **Always run `export_model.py` after every `train_model.py` run.**
> The ONNX + centroids must stay in sync with `classes.json`.

---

## 3. Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

The `models/` directory already contains the DNN face detector files:
- `models/deploy.prototxt`
- `models/res10_300x300_ssd_iter_140000_fp16.caffemodel`

---

## 4. Step-by-step usage

### Step 1 — Capture faces (repeat per person)

```powershell
# Webcam (auto-detects index 0, 1, 2...)
python capture_face.py --name alice
python capture_face.py --name bob

# DroidCam (phone as wireless camera)
python capture_face.py --name alice --droidcam 192.168.1.5
```

**Tips for best accuracy:**
- Move your head (left/right/up/down), vary expressions and distance.
- Capture under **different lighting conditions** (indoor, near window, etc).
- Aim for **250 images per person** (default `--num_images 250`).
- Minimum **2 people** required — the model needs multiple classes.
- Only saves when **exactly one face** is detected — prevents mislabeling.
- Re-running for the same person **appends** images, not overwrites.

### Step 2 — Train the CNN

```powershell
python train_model.py --epochs 30
```

- Training output shows `train_loss`, `train_acc`, `val_acc` per epoch.
- Checkpoint saves **only when `val_acc` improves** — always the best model.
- Typical time: **3–15 min on CPU**, under a minute on GPU.
- `val_acc 1.000` by epoch ~5 is normal for a small, well-captured dataset.

### Step 3 — Export to ONNX

```powershell
python export_model.py
```

Converts the trained PyTorch model to ONNX format and saves:
- `checkpoints/face_cnn.onnx` — model for `cv2.dnn` inference
- `checkpoints/centroids.npy` — per-class centroid embeddings
- `checkpoints/classes.json` — class index → name mapping

**Run this every time after training.** If you forget, `recognize_live.py`
will detect the mismatch and print a clear error.

### Step 4 — Run live recognition

```powershell
# Webcam
python recognize_live.py --camera 1

# DroidCam
python recognize_live.py --droidcam 192.168.1.5

# With custom threshold
python recognize_live.py --camera 1 --threshold 0.55

# Debug mode: prints cosine scores per face to calibrate threshold
python recognize_live.py --camera 1 --debug
```

Press `q` to quit.

---

## 5. Calibrating the threshold

The `--threshold` is a **cosine similarity** value (0.0 – 1.0):

| Behavior | Action |
|:---|:---|
| Known person showing as "Unknown" | **Lower** threshold (e.g. `0.25`) |
| Unknown stranger being matched | **Raise** threshold (e.g. `0.60`) |

**Use `--debug` to see actual scores:**
```
[DEBUG] alice:0.741, bob:0.123  | threshold=0.30
```

Default threshold: `0.30`. A good starting range is `0.25 – 0.55`.

---

## 6. Adding a new person later

```powershell
python capture_face.py --name charlie
python train_model.py --epochs 30
python export_model.py
python recognize_live.py --camera 1
```

Full retraining on all captured data takes only a few minutes — simpler
and more reliable than incremental updates.

---

## 7. Dataset layout

```
dataset/
├── alice/
│   ├── 0000.png   ┐
│   ├── 0001.png   │  96×96 grayscale PNG (lossless)
│   ├── ...        │  histogram-equalized face crop
│   └── 0249.png   ┘  folder name = label
├── bob/
│   ├── 0000.png
│   └── ...
└── charlie/
    └── ...
```

- **PNG** — lossless, no compression artifacts.
- **Grayscale** — removes color as a variable; matches `FaceCNN`'s `Conv2d(1, ...)` input.
- **96×96** — fixed size every crop is resized to.
- **Label by folder** — `train_model.py` treats each subfolder as a class, same convention as `torchvision.datasets.ImageFolder`.

---

## 8. File structure

```
Multi_Face/
├── requirements.txt            # pip dependencies
├── face_detector.py            # OpenCV DNN SSD face detector wrapper
├── model.py                    # FaceCNN architecture (PyTorch)
├── capture_face.py             # Step 1: Build your dataset
├── train_model.py              # Step 2: Train from scratch (PyTorch)
├── export_model.py             # Step 3: Export to ONNX for cv2.dnn
├── recognize_live.py           # Step 4: Live inference (pure OpenCV DNN)
├── models/
│   ├── deploy.prototxt                              # DNN face detector config
│   └── res10_300x300_ssd_iter_140000_fp16.caffemodel  # DNN face detector weights
├── dataset/                    # Created by capture_face.py
│   └── <name>/*.png
└── checkpoints/                # Created by train_model.py + export_model.py
    ├── face_cnn.pt             # PyTorch weights (training)
    ├── face_cnn.onnx           # ONNX model (inference via cv2.dnn)
    ├── centroids.npy           # Per-class centroid embeddings
    └── classes.json            # Index → person name mapping
```

---

## 9. Design notes

- **Why OpenCV DNN for detection**: Haar Cascade (the 2001 Viola-Jones method) misses angled faces, struggles in low light, and produces many false positives. The SSD ResNet-10 model is a small, pre-trained face localizer — it detects "is there a face here?" without ever knowing *whose* face it is.

- **Why cosine centroids instead of softmax at inference**: Softmax always picks the most likely known class even for a complete stranger. Cosine similarity against class centroids allows genuine "Unknown" rejection when the face embedding doesn't cluster near any trained identity.

- **Why centroids use clean (non-augmented) images**: Augmentations (random flips, rotations, jitter) shift embeddings away from what a real live camera frame produces. Centroids computed from clean crops match the inference distribution and produce higher cosine similarity scores for known faces.

- **Why ONNX + cv2.dnn at inference**: PyTorch is needed for training (autograd, backprop) but is a large runtime dependency (~700MB). Exporting to ONNX and running via `cv2.dnn` keeps live inference lightweight — no PyTorch import during recognition.

- **Why heavy augmentation during training**: With ~250 images per person, a CNN can memorize backgrounds and lighting rather than faces. `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomAffine`, and `RandomErasing` in `train_model.py` fight this memorization.