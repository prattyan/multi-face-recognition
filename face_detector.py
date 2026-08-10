import os
import cv2
import numpy as np


class FaceDetector:
    """
    Face detector using OpenCV's DNN SSD ResNet-10 model.
    Far more accurate than Haar Cascade — handles:
      - Varied lighting & shadows
      - Angled and tilted faces
      - Multiple faces simultaneously
      - Smaller and further away faces

    Downloads required (already placed in models/):
      - models/deploy.prototxt
      - models/res10_300x300_ssd_iter_140000_fp16.caffemodel
    """

    def __init__(self, confidence_threshold=0.5, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

        prototxt = os.path.join(model_dir, "deploy.prototxt")
        caffemodel = os.path.join(model_dir, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

        if not os.path.exists(prototxt) or not os.path.exists(caffemodel):
            raise FileNotFoundError(
                f"DNN model files not found in '{model_dir}'.\n"
                "Download them with:\n"
                "  deploy.prototxt from https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt\n"
                "  res10_300x300_ssd_iter_140000_fp16.caffemodel from https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20180205_fp16"
            )

        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame_bgr):
        """
        Returns a list of (x, y, w, h) bounding boxes for every face found.
        Uses DNN SSD which handles lighting, angles, and multiple faces reliably.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w = frame_bgr.shape[:2]

        # Prepare blob: resize to 300x300, mean subtraction for normalization
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue

            # Scale bounding box back to original frame size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)

            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            bw = x2 - x1
            bh = y2 - y1
            if bw > 0 and bh > 0:
                boxes.append((x1, y1, bw, bh))

        return boxes

    @staticmethod
    def crop_face(frame_bgr, box, out_size=96, margin=0.15):
        """
        Crops and preprocesses a face region from the frame.
        Adds a margin around the detection box and applies histogram equalization.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros((out_size, out_size), dtype=np.uint8)

        h_img, w_img = frame_bgr.shape[:2]
        x, y, w, h = box

        # Add margin around bounding box
        dw = int(w * margin)
        dh = int(h * margin)

        x1 = max(0, x - dw)
        y1 = max(0, y - dh)
        x2 = min(w_img, x + w + dw)
        y2 = min(h_img, y + h + dh)

        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            face = frame_bgr[max(0, y):min(h_img, y + h), max(0, x):min(w_img, x + w)]

        if face.size == 0:
            return np.zeros((out_size, out_size), dtype=np.uint8)

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_gray = cv2.equalizeHist(face_gray)
        face_resized = cv2.resize(face_gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return face_resized