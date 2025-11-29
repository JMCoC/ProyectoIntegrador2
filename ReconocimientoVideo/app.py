import time

import cv2
import torch
from facenet_pytorch import MTCNN
import argparse
import os

VIDEO_SOURCE = "video_rostro.mp4"
MIN_CONFIDENCE = 0.9

device = "cuda:0" if torch.cuda.is_available() else "cpu"
detector = MTCNN(keep_all=True, device=device)

def resize_max(frame, max_w=1280, max_h=720):
    """Redimensiona manteniendo aspecto si excede max_w/max_h."""
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame

def annotate_faces(frame):
    """Dibuja los rostros detectados usando MTCNN."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = detector.detect(rgb)
    if boxes is None or probs is None:
        return frame

    for (x1, y1, x2, y2), prob in zip(boxes, probs):
        if prob is None or prob < MIN_CONFIDENCE:
            continue
        start = (int(x1), int(y1))
        end = (int(x2), int(y2))
        cv2.rectangle(frame, start, end, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{prob:.2f}",
            (start[0], max(15, start[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return frame

def open_capture(source):
    """Abre fuente de video con comprobaciones y fallback para Windows."""
    print(f"[INFO] Abriendo fuente: {source}")
    if isinstance(source, int):
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        if isinstance(source, str):
            abs_path = os.path.abspath(source)
            if not os.path.exists(abs_path):
                print(f"[ERROR] Archivo no existe: {abs_path}")
            else:
                print(f"[INFO] Archivo localizado: {abs_path}")
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la fuente: {source}")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5):
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        time.sleep(0.1)
    print(f"[ERROR] La fuente {source} no entrega frames.")
    cap.release()
    return None

def stream_faces(capture, window_name=None, process_every=3, display_size=(960, 540)):
    """Procesa un VideoCapture y dibuja FPS + detecciones con salvaguardas."""
    if capture is None:
        return
    if window_name:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_size[0], display_size[1])
    prev = time.time()
    frames = 0
    detections_total = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            print("[INFO] Fin del stream o sin frames.")
            break
        frames += 1

        if frames % process_every == 0:
            before = frame.copy()
            frame = annotate_faces(frame)
            if not (before is frame):
                detections_total += 1

        if frames % 10 == 0:
            now = time.time()
            fps = 10 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

        if window_name:
            disp = cv2.resize(frame, display_size, interpolation=cv2.INTER_AREA)
            cv2.imshow(window_name, disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            _, _ = cv2.imencode(".jpg", frame)

    capture.release()
    print(f"[INFO] Frames procesados: {frames}, anotaciones realizadas: {detections_total}")

def parse_args():
    parser = argparse.ArgumentParser(description="Detección de rostros con MTCNN (video/webcam)")
    parser.add_argument("--source", type=str, default=None, help="Ruta de video o 'cam' para webcam")
    parser.add_argument("--cam-index", type=int, default=0, help="Índice de webcam (0,1,2)")
    parser.add_argument("--process-every", type=int, default=3, help="Procesar 1 de cada N frames")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.source is None:
        src_file = VIDEO_SOURCE
    else:
        src_file = None if args.source.lower() == "cam" else args.source

    if src_file is not None:
        cap_file = open_capture(src_file)
        stream_faces(cap_file, window_name="MTCNN Video", process_every=args.process_every)

    cam_indices = [args.cam_index, 1, 2] if args.cam_index == 0 else [args.cam_index, 0, 1, 2]
    cap_cam = None
    for idx in cam_indices:
        cap_cam = open_capture(idx)
        if cap_cam is not None:
            break
    stream_faces(cap_cam, window_name="MTCNN Live", process_every=args.process_every)
    cv2.destroyAllWindows()