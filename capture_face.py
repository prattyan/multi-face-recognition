import argparse
import os
import time
import cv2


from face_detector import FaceDetector

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
    parser.add_argument("--name", required=True, help="Person's name / label")
    parser.add_argument("--num_images", type=int, default=250)
    parser.add_argument("--out_dir", default="dataset")
    parser.add_argument("--camera", "--camera_index", default="0", help="Camera index (0, 1, 2) or IP camera URL")
    parser.add_argument("--droidcam", help="DroidCam IP (e.g., 192.168.1.5) or URL (http://192.168.1.5:4747/video)")
    args = parser.parse_args()
 
    person_dir = os.path.join(args.out_dir, args.name)
    os.makedirs(person_dir, exist_ok=True)
    existing = len([f for f in os.listdir(person_dir) if f.endswith(".png")])
 
    detector = FaceDetector()
    source = get_camera_source(args)
    cap = cv2.VideoCapture(source)

    # Auto-try index 1 and 2 if default index 0 failed
    if not cap.isOpened() and isinstance(source, int) and source == 0:
        for idx in [1, 2, 3]:
            temp_cap = cv2.VideoCapture(idx)
            if temp_cap.isOpened():
                cap = temp_cap
                source = idx
                print(f"Connected to camera index {idx}.")
                break
 
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera source '{source}'. If using DroidCam, ensure DroidCam Client is running or provide --droidcam <IP>.")
 
    print(f"Capturing for '{args.name}'. Press 'q' to stop early.")
    count = 0
    last_save = 0
 
    while count < args.num_images:
        ret, frame = cap.read()
        if not ret:
            break
 
        boxes = detector.detect(frame)
        display = frame.copy()
 
        # Only save when exactly one face is visible, to avoid mislabeling
        # someone walking through the background as the target person.
        if len(boxes) == 1:
            box = boxes[0]
            x, y, w, h = box
            now = time.time()
            if now - last_save > 0.15:  # throttle so frames aren't near-duplicates
                face_img = detector.crop_face(frame, box)
                fname = os.path.join(person_dir, f"{existing + count:04d}.png")
                cv2.imwrite(fname, face_img)
                count += 1
                last_save = now
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif len(boxes) > 1:
            cv2.putText(display, "Multiple faces - only one person at a time",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
 
        cv2.putText(display, f"{args.name}: {count}/{args.num_images}",
                    (10, display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Capture (q to quit)", display)
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} new images to {person_dir}")
 
if __name__ == "__main__":
    main()