"""
Liveness / Anti-Spoofing Detector — pure OpenCV, no external model downloads.

Combines two complementary signals:

  1. Texture Richness (Laplacian variance)
     Real 3-D faces have complex micro-texture (skin pores, fine hair).
     Printed photos / phone screens are texturally flat → low variance.

  2. Temporal Blink Tracking (haarcascade_eye)
     Uses OpenCV's built-in eye cascade.
     Real people blink → eye-detection toggles over time.
     Static photos/screens → eyes always visible or always absent (no change).
     A state change in a rolling window is required to pass as LIVE.

Decision:
    LIVE  = texture_score > threshold  AND  blink_event_in_window
    SPOOF = either condition fails

Per-face history is keyed by a face_id (e.g. bounding-box hash) so each
tracked face has its own independent blink window.

Usage:
    from anti_spoof import LivenessDetector
    detector = LivenessDetector()
    is_live, score = detector.check(face_gray_96, face_id="face_0")
"""

from collections import defaultdict, deque

import cv2
import numpy as np


class LivenessDetector:
    """
    Pure-OpenCV liveness detector — no pretrained identity models.

    Args:
        texture_threshold:  Laplacian variance below this → SPOOF.
                            Lower value = more permissive (camera-dependent).
                            Tune with LivenessDetector.texture_score().
        blink_window:       Number of recent frames to keep per face.
                            State change in this window = blink detected.
        eye_scale:          scaleFactor for haarcascade_eye detectMultiScale.
        eye_neighbors:      minNeighbors for eye detection (higher = stricter).
    """

    def __init__(
        self,
        texture_threshold: float = 80.0,
        blink_window: int = 90,       # ~3 s at 30 fps
        eye_scale: float = 1.1,
        eye_neighbors: int = 4,
    ):
        self.texture_threshold = texture_threshold
        self.blink_window = blink_window
        self.eye_scale = eye_scale
        self.eye_neighbors = eye_neighbors

        # Load eye cascade — shipped with every OpenCV install
        cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self._eye_cascade = cv2.CascadeClassifier(cascade_path)
        if self._eye_cascade.empty():
            raise RuntimeError(f"Could not load eye cascade from {cascade_path}")

        # Rolling eye-state history per face_id  (True = eyes detected)
        self._history: dict = defaultdict(lambda: deque(maxlen=blink_window))

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, face_gray_96: np.ndarray, face_id: str) -> tuple[bool, float]:
        """
        Run liveness check on a 96×96 grayscale face crop.

        Args:
            face_gray_96:  Grayscale face crop (any size; internally resized).
            face_id:       Unique identifier for this face track (e.g. box hash).
                           Used to maintain blink history across frames.

        Returns:
            (is_live, texture_score)
            is_live       — True if both texture and blink tests pass.
            texture_score — Raw Laplacian variance (for debugging / tuning).
        """
        # ── 1. Texture test ──────────────────────────────────────────────────
        score = self.texture_score(face_gray_96)
        texture_ok = score > self.texture_threshold

        # ── 2. Blink / temporal test ─────────────────────────────────────────
        eyes_now = self._eyes_detected(face_gray_96)
        history = self._history[face_id]
        history.append(eyes_now)

        blink_ok = self._has_state_change(history)

        return (texture_ok and blink_ok), score

    def texture_score(self, face_gray: np.ndarray) -> float:
        """
        Compute Laplacian variance of a grayscale face image.
        Higher = richer texture = more likely a real face.
        """
        resized = cv2.resize(face_gray, (96, 96)) if face_gray.shape != (96, 96) else face_gray
        lap = cv2.Laplacian(resized, cv2.CV_64F)
        return float(lap.var())

    def reset(self, face_id: str) -> None:
        """Clear blink history for a face_id (call when a face disappears)."""
        self._history.pop(face_id, None)

    def clear_stale(self, active_ids: set) -> None:
        """Remove history for face_ids that are no longer tracked."""
        stale = [fid for fid in self._history if fid not in active_ids]
        for fid in stale:
            del self._history[fid]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _eyes_detected(self, face_gray: np.ndarray) -> bool:
        """Return True if at least one eye is found in the face crop."""
        # Equalize for consistent detection across lighting
        eq = cv2.equalizeHist(face_gray)
        eyes = self._eye_cascade.detectMultiScale(
            eq,
            scaleFactor=self.eye_scale,
            minNeighbors=self.eye_neighbors,
            minSize=(10, 10),
        )
        return len(eyes) >= 1

    @staticmethod
    def _has_state_change(history: deque) -> bool:
        """
        Returns True if the eye-state has changed at least once in the window.
        A brand-new face (< 10 frames) gets the benefit of the doubt → True.
        """
        if len(history) < 10:
            return True   # Not enough data yet — don't penalize a new face
        return len(set(history)) > 1
