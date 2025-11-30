from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models.detection import SSD300_VGG16_Weights, ssd300_vgg16


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta detección de objetos con SSD300-VGG16 sobre una imagen "
            "o sobre todas las imágenes de una carpeta."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="imagenes",
        help="Ruta de una imagen o carpeta con imágenes (por defecto: ./imagenes).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Umbral mínimo de confianza para mostrar una detección (0-1).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="resultados",
        help=(
            "Carpeta donde guardar las imágenes anotadas. "
            "Se crea si no existe (por defecto: ./resultados)."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No mostrar las imágenes en pantalla, solo guardarlas.",
    )
    return parser.parse_args()


def gather_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    raise FileNotFoundError(
        f"La ruta {path} no existe. Proporciona una imagen o carpeta válida."
    )


def annotate_image(
    img: Image.Image,
    detections: dict,
    categories: list[str],
    confidence: float,
    title: str,
) -> plt.Figure:
    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]

    fig, ax = plt.subplots(1, figsize=(9, 6))
    ax.imshow(img)
    ax.set_title(title)
    kept = 0
    for box, lab, sc in zip(boxes, labels, scores):
        if sc < confidence:
            continue
        kept += 1
        x1, y1, x2, y2 = box.tolist()
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor="lime",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 5, 5),
            f"{categories[int(lab)]}: {sc:.2f}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6, pad=1),
        )

    if kept == 0:
        ax.text(
            0.5,
            0.05,
            "Sin detecciones sobre el umbral configurado",
            transform=ax.transAxes,
            color="yellow",
            fontsize=12,
            ha="center",
        )
    ax.axis("off")
    return fig


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando pesos preentrenados SSD300-VGG16...")
    weights = SSD300_VGG16_Weights.DEFAULT
    model = ssd300_vgg16(weights=weights).eval()
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    images = gather_images(input_path)
    if not images:
        raise RuntimeError(
            f"No se encontraron imágenes en {input_path}. "
            "Extensiones soportadas: .jpg, .jpeg, .png, .bmp"
        )

    print(f"Se procesarán {len(images)} imagen(es) con umbral {args.confidence:.2f}")

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        batch = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            t0 = time.time()
            detections = model(batch)[0]
            elapsed = time.time() - t0

        fig = annotate_image(
            img,
            detections,
            categories,
            args.confidence,
            title=f"{img_path.name} · {elapsed:.3f}s",
        )

        output_path = save_dir / f"{img_path.stem}_ssd.png"
        fig.savefig(output_path, bbox_inches="tight", dpi=150)

        if args.no_show:
            plt.close(fig)
        else:
            plt.show()

        filtered = [
            (categories[int(lab)], float(score))
            for lab, score in zip(detections["labels"], detections["scores"])
            if score >= args.confidence
        ]
        print(
            f"{img_path.name}: {len(filtered)} detecciones "
            f"(tiempo {elapsed:.3f} s). Resultado: {output_path}"
        )
        for name, score in filtered:
            print(f"  - {name:<15} {score:.2f}")


if __name__ == "__main__":
    main()