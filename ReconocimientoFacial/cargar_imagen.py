import argparse
from pathlib import Path
from time import time

import cv2
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN


def iou(a, b):
    """Calcula la superposición entre dos cajas [x, y, w, h]."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Laboratorio: detección de rostros con MTCNN personalizable"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="grupo.jpg",
        help="Ruta de la imagen a procesar (ignorada si usas --camara)",
    )
    parser.add_argument(
        "--camara",
        action="store_true",
        help="Usa un frame de la webcam en lugar de un archivo",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="Umbral mínimo de confianza para reportar rostros",
    )
    parser.add_argument(
        "--min-face",
        type=int,
        default=40,
        help="Tamaño mínimo en pixeles para considerar un rostro",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Ruta donde guardar la visualización (deja vacío para no guardar)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Evita abrir la ventana de Matplotlib (útil en ejecuciones remotas)",
    )
    return parser.parse_args()


def load_frame(args):
    if args.camara:
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("No se pudo capturar frame de la cámara")
        return frame, "frame_webcam"
    path = Path(args.source)
    if not path.exists():
        raise FileNotFoundError(f"No encuentro la imagen: {path}")
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError("cv2.imread retornó None, revisa el formato de la imagen")
    return frame, path.stem


def draw_detections(image_rgb, detections, thr):
    vis = image_rgb.copy()
    for det in detections:
        if det["confidence"] < thr:
            continue
        x, y, w, h = det["box"]
        color = (0, 255, 0)
        label = f"{det['confidence']*100:.1f}%"
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            vis,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        for (px, py) in det["keypoints"].values():
            cv2.circle(vis, (px, py), 2, (255, 0, 0), -1)
    return vis


def main():
    args = parse_args()
    frame_bgr, descriptor = load_frame(args)
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    detector = MTCNN(min_face_size=args.min_face)

    t0 = time()
    detections = detector.detect_faces(img_rgb)
    t1 = time()
    elapsed_ms = (t1 - t0) * 1000

    seleccionados = [
        det for det in detections if det["confidence"] >= args.confidence
    ]
    print(
        f"[{descriptor}] Detecciones totales: {len(detections)} | "
        f"con thr={args.confidence:.2f}: {len(seleccionados)} | "
        f"tiempo: {elapsed_ms:.1f} ms"
    )

    for idx, det in enumerate(seleccionados, start=1):
        box = det["box"]
        print(
            f"- Rostro {idx}: conf={det['confidence']:.3f} "
            f"bbox={box} keypoints={list(det['keypoints'].keys())}"
        )

    vis = draw_detections(img_rgb, detections, args.confidence)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"Visualización guardada en {save_path}")

    if not args.no_show:
        plt.imshow(vis)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if len(seleccionados) > 1:
        base = seleccionados[0]["box"]
        overlaps = [iou(base, det["box"]) for det in seleccionados[1:]]
        if overlaps:
            print(
                f"IoU respecto a la primera detección: "
                f"{', '.join(f'{ov:.2f}' for ov in overlaps)}"
            )


if __name__ == "__main__":
    TF_ENABLE_ONEDNN_OPTS = 0  # desactivar optimizaciones que rompen MTCNN en CPU
    main()