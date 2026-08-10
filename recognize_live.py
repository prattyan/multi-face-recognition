"""
Step 4: Real-time multi-face recognition using OpenCV DNN for both:
  - Face Detection     (SSD ResNet-10 DNN model)
  - Face Recognition   (FaceCNN exported to ONNX, run via cv2.dnn)
  - Liveness Detection (texture + blink — anti-spoofing, pure OpenCV)
  - Attendance Logging (CSV per day, per-person cooldown)

No PyTorch needed at inference time — pure OpenCV pipeline.

    python recognize_live.py --camera 1
    python recognize_live.py --droidcam 192.168.1.5
    python recognize_live.py --camera 1 --no_spoof       # disable liveness check
    python recognize_live.py --camera 1 --no_attendance  # disable attendance log
    python recognize_live.py --camera 1 --debug          # print similarity scores

Press 'q' to quit.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np

from face_detector import FaceDetector
from anti_spoof import LivenessDetector
from attendance_logger import AttendanceLogger


# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess_face(face_gray_96: np.ndarray) -> np.ndarray:
    """Normalize face crop to the same format used during training."""
    img = face_gray_96.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5                         # mean=0.5, std=0.5
    return img[np.newaxis, np.newaxis, :, :]         # (1, 1, 96, 96)


def get_camera_source(args):
    if args.droidcam:
        url = args.droidcam.strip()
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}:4747/video" if ":" not in url else f"http://{url}/video"
        return url
    cam = str(args.camera).strip()
    return int(cam) if cam.isdigit() else cam


def face_id(box) -> str:
    """Stable string key for a bounding box (used to track liveness per face)."""
    x, y, w, h = box
    return f"{x // 20}_{y // 20}_{w // 20}_{h // 20}"


def draw_label(frame, text, x, y, color, font_scale=0.55, thickness=2):
    """Draw text with a solid background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    cv2.rectangle(frame, (x, y - th - pad), (x + tw + pad, y + baseline), color, -1)
    cv2.putText(frame, text, (x + 2, y), font, font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA)


def draw_banner(frame, lines: list[str], color=(30, 120, 30)):
    """Draw a centered multi-line notification banner at the bottom of frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    line_h = 30
    banner_h = len(lines) * line_h + 20
    y0 = h - banner_h - 10

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (w // 6, y0), (5 * w // 6, y0 + banner_h), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    for i, line in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        tx = (w - tw) // 2
        ty = y0 + 20 + i * line_h
        cv2.putText(frame, line, (tx, ty), font, font_scale, (255, 255, 255), thickness,
                    cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time multi-face recognition with liveness detection and attendance logging."
    )
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Min cosine similarity to accept identity (default: 0.30). "
                             "Tune with --debug.")
    parser.add_argument("--camera", "--camera_index", default="0",
                        help="Camera index (0, 1, 2) or IP camera URL")
    parser.add_argument("--droidcam",
                        help="DroidCam IP or URL (e.g. 192.168.1.5)")
    parser.add_argument("--debug", action="store_true",
                        help="Print cosine scores + texture scores per frame")

    # Liveness
    parser.add_argument("--no_spoof", action="store_true",
                        help="Disable liveness / anti-spoofing check")
    parser.add_argument("--spoof_texture_threshold", type=float, default=80.0,
                        help="Laplacian variance threshold for liveness (default: 80). "
                             "Lower = more permissive. Use --debug to tune.")

    # Attendance
    parser.add_argument("--no_attendance", action="store_true",
                        help="Disable attendance CSV logging")
    parser.add_argument("--attendance_dir", default="attendance",
                        help="Directory for attendance CSV files (default: attendance/)")
    parser.add_argument("--attendance_cooldown", type=float, default=60.0,
                        help="Seconds before the same person can be logged again (default: 60)")

    args = parser.parse_args()

    # ── Load class names & centroids ─────────────────────────────────────────
    classes_path  = os.path.join(args.checkpoint_dir, "classes.json")
    onnx_path     = os.path.join(args.checkpoint_dir, "face_cnn.onnx")
    centroids_path = os.path.join(args.checkpoint_dir, "centroids.npy")

    for path, hint in [
        (classes_path,   "Run train_model.py then export_model.py first."),
        (onnx_path,      "Run export_model.py first."),
        (centroids_path, "Run export_model.py first."),
    ]:
        if not os.path.exists(path):
            raise RuntimeError(f"Missing: {path}. {hint}")

    with open(classes_path) as f:
        classes = json.load(f)

    centroids = np.load(centroids_path)
    if centroids.shape[0] != len(classes):
        raise RuntimeError(
            f"STALE EXPORT: centroids.npy has {centroids.shape[0]} classes but "
            f"classes.json has {len(classes)} classes.\n"
            f"Run:  python export_model.py   to re-export after training."
        )
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    # ── Load recognition model (ONNX via cv2.dnn) ────────────────────────────
    recognition_net = cv2.dnn.readNetFromONNX(onnx_path)
    print(f"Loaded ONNX model : {onnx_path}")
    print(f"Classes           : {classes}")

    # ── Init liveness detector ────────────────────────────────────────────────
    liveness = None
    if not args.no_spoof:
        liveness = LivenessDetector(texture_threshold=args.spoof_texture_threshold)
        print(f"Liveness check    : ENABLED  (texture_threshold={args.spoof_texture_threshold})")
    else:
        print("Liveness check    : DISABLED")

    # ── Init attendance logger ────────────────────────────────────────────────
    attendance = None
    if not args.no_attendance:
        attendance = AttendanceLogger(
            log_dir=args.attendance_dir,
            cooldown_seconds=args.attendance_cooldown,
        )
        print(f"Attendance log    : {attendance.today_log_path}")
    else:
        print("Attendance log    : DISABLED")

    # ── Open camera ───────────────────────────────────────────────────────────
    detector = FaceDetector()
    source = get_camera_source(args)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened() and isinstance(source, int) and source == 0:
        for idx in [1, 2, 3]:
            tmp = cv2.VideoCapture(idx)
            if tmp.isOpened():
                cap = tmp
                source = idx
                print(f"Connected to camera index {idx}.")
                break

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source '{source}'. "
            "If using DroidCam, ensure the DroidCam app is running and provide --droidcam <IP>."
        )

    print("Recognizing live. Press 'q' to quit.\n")

    # ── Banner state ──────────────────────────────────────────────────────────
    banner_until = 0.0          # monotonic timestamp until which to show banner
    banner_lines: list[str] = []
    BANNER_DURATION = 2.5       # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)
        active_ids = set()

        for box in boxes:
            x, y, w, h = box
            fid = face_id(box)
            active_ids.add(fid)

            # ── Crop & recognize ─────────────────────────────────────────────
            face_gray = detector.crop_face(frame, box)
            blob = preprocess_face(face_gray)

            recognition_net.setInput(blob)
            embedding = recognition_net.forward()                    # (1, 128)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

            cosine_sims = np.dot(centroids, embedding[0])
            best_idx    = int(np.argmax(cosine_sims))
            best_sim    = float(cosine_sims[best_idx])

            recognized = best_sim >= args.threshold and 0 <= best_idx < len(classes)
            name       = classes[best_idx] if recognized else "Unknown"

            if args.debug:
                scores = ", ".join(f"{classes[i]}:{cosine_sims[i]:.3f}"
                                   for i in range(len(classes)))
                print(f"[DEBUG] {scores}  | threshold={args.threshold:.2f}")

            # ── Liveness check ───────────────────────────────────────────────
            is_live = True
            texture_score = 0.0
            if liveness is not None:
                is_live, texture_score = liveness.check(face_gray, fid)
                if args.debug:
                    print(f"[LIVENESS] fid={fid}  texture={texture_score:.1f}  "
                          f"live={is_live}")

            # ── Draw bounding box ─────────────────────────────────────────────
            if not is_live:
                box_color = (0, 0, 200)          # red  → spoof
            elif recognized:
                box_color = (0, 200, 0)          # green → known
            else:
                box_color = (200, 100, 0)        # blue-ish → unknown

            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # ── Draw labels ───────────────────────────────────────────────────
            label_y = y - 12 if y - 12 > 14 else y + h + 22

            if not is_live:
                draw_label(frame, "SPOOF", x, label_y, (0, 0, 180))
            else:
                if recognized:
                    draw_label(frame, f"{name}  {best_sim * 100:.0f}%", x, label_y,
                               (0, 160, 0))
                    # Small live badge above the name
                    draw_label(frame, "LIVE", x, label_y - 22, (0, 120, 0),
                               font_scale=0.4, thickness=1)
                else:
                    draw_label(frame, f"Unknown  {best_sim * 100:.0f}%", x, label_y,
                               (140, 60, 0))

            # ── Attendance logging ────────────────────────────────────────────
            if attendance is not None and recognized and is_live:
                logged = attendance.try_log(name, best_sim)
                if logged:
                    now_str = __import__("datetime").datetime.now().strftime("%H:%M:%S")
                    banner_lines = [f"  ATTENDANCE LOGGED  ", f"{name}   {now_str}  "]
                    banner_until = time.monotonic() + BANNER_DURATION

        # ── Stale liveness history cleanup ────────────────────────────────────
        if liveness is not None:
            liveness.clear_stale(active_ids)

        # ── HUD ───────────────────────────────────────────────────────────────
        hud = f"Faces: {len(boxes)}"
        if attendance is not None:
            hud += f"   Log: {attendance.today_log_path}"
        cv2.putText(frame, hud, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 1, cv2.LINE_AA)

        # Mode badges (top-right corner)
        badge_x = frame.shape[1] - 160
        if liveness is not None:
            cv2.putText(frame, "ANTISPOOF ON", (badge_x, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 200), 1, cv2.LINE_AA)
        if attendance is not None:
            cv2.putText(frame, "ATTENDANCE ON", (badge_x, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 220), 1, cv2.LINE_AA)

        # ── Attendance banner ─────────────────────────────────────────────────
        if time.monotonic() < banner_until:
            draw_banner(frame, banner_lines)

        cv2.imshow("Live Face Recognition  |  q = quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    if attendance is not None:
        print(f"\nAttendance saved to: {attendance.today_log_path}")


if __name__ == "__main__":
    main()