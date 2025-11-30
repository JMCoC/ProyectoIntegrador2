import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo personalizado para detección en vivo con YOLOv8 nano."
    )
    parser.add_argument(
        "--source",
        default=0,
        help="Índice de cámara (0, 1, ...) o ruta a un video/stream.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Ruta al modelo YOLOv8 a utilizar.",
    )
    parser.add_argument(
        "--window",
        default="Detección YOLOv8 Lite",
        help="Nombre de la ventana de visualización.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {args.source}")

    prev_time = time.time()
    screenshot_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin del stream o error de lectura.")
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        now = time.time()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now

        info_text = f"FPS: {fps:.1f} | Fuente: {args.source}"
        cv2.putText(
            annotated,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "Presiona S para capturar / Q para salir",
            (10, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(args.window, annotated)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            filename = f"captura_{screenshot_id}.png"
            cv2.imwrite(filename, annotated)
            print(f"Captura guardada en {filename}")
            screenshot_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()