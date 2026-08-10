"""
Step 3: Real-time multi-face recognition using OpenCV DNN for both:
  - Face Detection  (SSD ResNet-10 DNN model)
  - Face Recognition (FaceCNN exported to ONNX, run via cv2.dnn)

No PyTorch needed at inference time — pure OpenCV pipeline.

    python recognize_live.py --camera 1
    python recognize_live.py --droidcam 192.168.1.5

For every frame:
  1. OpenCV DNN SSD detects ALL faces (handles angles, lighting, distance).
  2. Each face crop is preprocessed and fed through the ONNX FaceCNN via cv2.dnn.
  3. Cosine similarity against trained class centroids determines identity.
  4. If the best match is below --threshold, the face is labeled "Unknown".

Press 'q' to quit.
"""

import argparse
import json
import os

import cv2
import numpy as np

from face_detector import FaceDetector


def preprocess_face(face_gray_96):
    """Normalize face crop to the same format used during training."""
    img = face_gray_96.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize: mean=0.5, std=0.5
    img = img[np.newaxis, np.newaxis, :, :]  # (1, 1, 96, 96)
    return img


def get_camera_source(args):
    if args.droidcam:
        ip_or_url = args.droidcam.strip()
        if not ip_or_url.startswith("http://") and not ip_or_url.startswith("https://"):
            if ":" not in ip_or_url:
                ip_or_url = f"http://{ip_or_url}:4747/video"
            else:
                ip_or_url = f"http://{ip_or_url}/video"
        return ip_or_url

    cam_str = str(args.camera).strip()
    if cam_str.isdigit():
        return int(cam_str)
    return cam_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Min cosine similarity (0.0-1.0) to accept an identity; below this -> Unknown. "
                             "Lower = more permissive. Start at 0.30, tune with --debug.")
    parser.add_argument("--camera", "--camera_index", default="0",
                        help="Camera index (0, 1, 2) or IP camera URL")
    parser.add_argument("--droidcam",
                        help="DroidCam IP (e.g., 192.168.1.5) or URL (http://192.168.1.5:4747/video)")
    parser.add_argument("--debug", action="store_true",
                        help="Print cosine similarity scores per frame for threshold calibration")
    args = parser.parse_args()

    # ── Load class names & centroids ──────────────────────────────────────────
    classes_path = os.path.join(args.checkpoint_dir, "classes.json")
    onnx_path = os.path.join(args.checkpoint_dir, "face_cnn.onnx")
    centroids_path = os.path.join(args.checkpoint_dir, "centroids.npy")

    if not os.path.exists(classes_path):
        raise RuntimeError("classes.json not found. Run train_model.py then export_model.py first.")
    if not os.path.exists(onnx_path):
        raise RuntimeError("face_cnn.onnx not found. Run export_model.py first.")
    if not os.path.exists(centroids_path):
        raise RuntimeError("centroids.npy not found. Run export_model.py first.")

    with open(classes_path) as f:
        classes = json.load(f)

    centroids = np.load(centroids_path)          # shape: (num_classes, 128)

    # ── Stale model check ─────────────────────────────────────────────────────
    if centroids.shape[0] != len(classes):
        raise RuntimeError(
            f"STALE EXPORT: centroids.npy has {centroids.shape[0]} classes but "
            f"classes.json has {len(classes)} classes.\n"
            f"Run:  python export_model.py   to re-export after training."
        )

    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    # ── Load recognition model via OpenCV DNN ─────────────────────────────────
    recognition_net = cv2.dnn.readNetFromONNX(onnx_path)
    print(f"Loaded ONNX model: {onnx_path}")
    print(f"Classes: {classes}")

    # ── Open camera ───────────────────────────────────────────────────────────
    detector = FaceDetector()
    source = get_camera_source(args)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened() and isinstance(source, int) and source == 0:
        for idx in [1, 2, 3]:
            temp_cap = cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                cap = temp_cap
                source = idx
                print(f"Connected to camera index {idx}.")
                break

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source '{source}'. "
            "If using DroidCam, ensure DroidCam Client is running or provide --droidcam <IP>."
        )

    print("Recognizing live. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)

        for box in boxes:
            x, y, w, h = box

            # Crop & preprocess face
            face_gray = detector.crop_face(frame, box)
            blob = preprocess_face(face_gray)

            # Run recognition through OpenCV DNN ONNX model
            recognition_net.setInput(blob)
            embedding = recognition_net.forward()           # (1, 128)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)  # L2 normalize

            # Cosine similarity against all class centroids
            cosine_sims = np.dot(centroids, embedding[0])  # (num_classes,)
            best_idx = int(np.argmax(cosine_sims))
            best_sim = float(cosine_sims[best_idx])

            if args.debug:
                scores = ", ".join(f"{classes[i]}:{cosine_sims[i]:.3f}" for i in range(len(classes)))
                print(f"[DEBUG] {scores}  | threshold={args.threshold:.2f}")

            # Decide label & color (guard against any index mismatch)
            if best_sim >= args.threshold and 0 <= best_idx < len(classes):
                label = f"{classes[best_idx]} ({best_sim * 100:.0f}%)"
                color = (0, 220, 0)
            else:
                label = f"Unknown ({best_sim * 100:.0f}%)"
                color = (0, 0, 220)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label background for readability
            label_y = y - 10 if y - 10 > 10 else y + h + 20
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, label_y - text_h - 4), (x + text_w, label_y + 2), color, -1)
            cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # HUD
        cv2.putText(frame, f"Faces: {len(boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Live Face Recognition (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()