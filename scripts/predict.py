"""Prédiction sur une photographie de maïs avec le modèle Keras du mémoire."""

import argparse
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

BASE = Path(__file__).resolve().parents[1]
MODEL = BASE / "models" / "best_simple_cnn_model.keras"
LABELS = BASE / "models" / "labels.txt"
IMAGE_SIZE = (150, 150)


def main():
    parser = argparse.ArgumentParser(description="Classifier une photographie de feuille de maïs.")
    parser.add_argument("image", type=Path, help="Chemin de l'image à analyser")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image introuvable : {args.image}")
    if not MODEL.is_file():
        parser.error(f"Modèle introuvable : {MODEL} (vérifier Git LFS)")
    if not LABELS.is_file():
        parser.error(f"Étiquettes introuvables : {LABELS}")

    classes = [line.strip() for line in LABELS.read_text(encoding="utf-8").splitlines() if line.strip()]
    model = tf.keras.models.load_model(MODEL, compile=False)
    img = tf.keras.utils.load_img(args.image, target_size=IMAGE_SIZE, color_mode="rgb")
    x = tf.keras.utils.img_to_array(img).astype("float32") / 255.0
    scores = model.predict(np.expand_dims(x, axis=0), verbose=0)[0]

    if len(classes) != len(scores):
        raise ValueError(f"{len(classes)} classes dans labels.txt, mais {len(scores)} sorties du modèle")

    top = int(np.argmax(scores))
    print(f"Classe prédite : {classes[top]} ({scores[top] * 100:.1f} %)\n")
    for name, score in sorted(zip(classes, scores), key=lambda pair: -pair[1]):
        print(f"  {name:<26} {score * 100:6.2f} %")


if __name__ == "__main__":
    sys.exit(main())
