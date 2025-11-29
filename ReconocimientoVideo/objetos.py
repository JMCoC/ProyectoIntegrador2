import argparse
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import SSD300_VGG16_Weights


COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detección de objetos con SSD300 + VGG16 preentrenado."
    )
    parser.add_argument(
        "-i",
        "--image",
        type=Path,
        default=Path("perros.jpg"),
        help="Ruta a la imagen de entrada (por defecto perros.jpg).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("perros_detectados.jpg"),
        help="Ruta del archivo anotado a guardar.",
    )
    parser.add_argument(
        "-c",
        "--min-confidence",
        type=float,
        default=0.4,
        help="Confianza mínima para dibujar una detección.",
    )
    parser.add_argument(
        "-k",
        "--max-detections",
        type=int,
        default=10,
        help="Máximo de cajas a mostrar (más altas primero).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.image.exists():
        raise FileNotFoundError(f"No encontré la imagen {args.image}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weights = SSD300_VGG16_Weights.DEFAULT
    preprocess = weights.transforms()
    model = torchvision.models.detection.ssd300_vgg16(weights=weights).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(tensor)[0]

    scores = prediction["scores"].cpu()
    boxes = prediction["boxes"].cpu()
    labels = prediction["labels"].cpu()

    keep = scores >= args.min_confidence
    scores = scores[keep][: args.max_detections]
    boxes = boxes[keep][: args.max_detections]
    labels = labels[keep][: args.max_detections]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores), start=1):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        name = COCO_INSTANCE_CATEGORY_NAMES[label] if label < len(
            COCO_INSTANCE_CATEGORY_NAMES
        ) else f"id_{label}"
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1 + 4, y1 + 4), f"{name} {score:.2f}", fill="black")
        draw.text((x1 + 5, y1 + 5), f"{name} {score:.2f}", fill="white")
        print(f"[{idx}] {name} -> {score:.2f} ({x1},{y1},{x2},{y2})")

    annotated.save(args.output)
    print(f"Guardé las detecciones en {args.output.resolve()}")


if __name__ == "__main__":
    main()