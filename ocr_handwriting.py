import argparse
import json
import os

import cv2
import h5py
import numpy as np
import tensorflow as tf

# ── constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 32        # dimensions the model expects
# map integer label index → alphanumeric character
# NOTE: this ordering must exactly match the class indices used during training.
#       Do NOT reorder or remove characters — doing so will break model predictions.
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# minimum bounding-box dimensions (in pixels) used to discard noise contours
MIN_CONTOUR_W = 5
MIN_CONTOUR_H = 15
MAX_CONTOUR_W = 150
MAX_CONTOUR_H = 120


# ── helpers ───────────────────────────────────────────────────────────────────
def load_model(model_path: str):
    """Load a Keras .h5/.keras model.

    Returns a callable that accepts a preprocessed ROI array and returns
    class-probability predictions as a NumPy array.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Treat the provided path as a single-file Keras model (.h5/.keras).
    try:
        keras_model = tf.keras.models.load_model(model_path)
    except TypeError as exc:
        # Compatibility fallback for older/newer Keras mismatches in H5 configs.
        if "quantization_config" not in str(exc):
            raise

        with h5py.File(model_path, "r") as h5_file:
            model_config = h5_file.attrs.get("model_config")
            if model_config is None:
                raise RuntimeError("H5 model is missing model_config metadata")

            if isinstance(model_config, (bytes, bytearray)):
                model_config = model_config.decode("utf-8")

        def strip_key(obj, key_to_remove: str):
            if isinstance(obj, dict):
                return {
                    k: strip_key(v, key_to_remove)
                    for k, v in obj.items()
                    if k != key_to_remove
                }
            if isinstance(obj, list):
                return [strip_key(v, key_to_remove) for v in obj]
            return obj

        cleaned_config = strip_key(json.loads(model_config), "quantization_config")
        keras_model = tf.keras.models.Model.from_config(cleaned_config["config"])
        keras_model.load_weights(model_path)

    def infer(roi_input: np.ndarray) -> np.ndarray:
        return keras_model.predict(roi_input, verbose=0)

    return infer


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """Threshold, aspect-preserving resize, pad, and normalise a character ROI."""
    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    tH, tW = thresh.shape

    if tW > tH:
        new_w = IMG_SIZE
        new_h = max(1, int(round((tH / float(tW)) * IMG_SIZE)))
    else:
        new_h = IMG_SIZE
        new_w = max(1, int(round((tW / float(tH)) * IMG_SIZE)))

    interp = cv2.INTER_AREA if (new_w < tW or new_h < tH) else cv2.INTER_LINEAR
    resized = cv2.resize(thresh, (new_w, new_h), interpolation=interp)

    dX = max(0, IMG_SIZE - new_w)
    dY = max(0, IMG_SIZE - new_h)
    left = dX // 2
    right = dX - left
    top = dY // 2
    bottom = dY - top

    padded = cv2.copyMakeBorder(
        resized,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    padded = cv2.resize(padded, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    padded = padded.astype("float32") / 255.0
    padded = np.expand_dims(padded, axis=-1)   # (32, 32, 1)
    return padded


def sort_contours(contours, method: str = "left-to-right"):
    """Sort contours spatially so characters are read in order."""
    reverse = method in ("right-to-left", "bottom-to-top")
    i = 1 if method in ("top-to-bottom", "bottom-to-top") else 0
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, bounding_boxes = zip(
        *sorted(zip(contours, bounding_boxes),
                key=lambda b: b[1][i],
                reverse=reverse)
    )
    return contours, bounding_boxes


# ── argument parsing ──────────────────────────────────────────────────────────
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OCR handwriting samples with a ResNet model.")
    ap.add_argument("-m", "--model",
                    default="ocr_model.h5",
                    help="path to the trained handwriting OCR model")
    ap.add_argument("-i", "--image",
                    required=True,
                    help="path to the input image")
    ap.add_argument("-o", "--output",
                    default=None,
                    help="(optional) path to save the annotated output image")
    return ap


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = build_arg_parser().parse_args()

    # load model
    print(f"[INFO] loading model from '{args.model}' …")
    model = load_model(args.model)

    # load image and convert to grayscale
    print(f"[INFO] processing image '{args.image}' …")
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # detect external contours on edges to better separate touching strokes
    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged.copy(),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[WARN] No characters found in the image.")
        return

    # sort left-to-right so we build the string in reading order
    contours, bounding_boxes = sort_contours(contours, method="left-to-right")

    chars_result = []
    char_images = []
    boxes = []

    for (x, y, w, h) in bounding_boxes:
        # filter out very small blobs (noise)
        if (
            w < MIN_CONTOUR_W
            or h < MIN_CONTOUR_H
            or w > MAX_CONTOUR_W
            or h > MAX_CONTOUR_H
        ):
            continue

        # clip ROI coordinates to image bounds to avoid index errors
        img_h, img_w = gray.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)

        # extract the character ROI from grayscale image and preprocess
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        char_images.append(preprocess_roi(roi))
        boxes.append((x, y, w, h))

    if not char_images:
        print("[WARN] No valid character regions after contour filtering.")
        return

    # run inference on all character crops at once
    chars_array = np.array(char_images, dtype="float32")
    predictions = model(chars_array)

    for pred, (x, y, w, h) in zip(predictions, boxes):
        label_idx = int(np.argmax(pred))
        label = CHARS[label_idx]
        confidence = float(pred[label_idx])
        chars_result.append(label)

        # Match reference-style console output per detected character.
        print(f"[INFO] {label} - {confidence * 100:.2f}%")

        # draw bounding box and label on the original colour image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image,
                    f"{label} ({confidence * 100:.1f}%)",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1)

    ocr_output = "".join(chars_result)

    # optionally save annotated image
    if args.output:
        cv2.imwrite(args.output, image)
        print(f"[INFO] annotated image saved to '{args.output}'")


if __name__ == "__main__":
    main()
