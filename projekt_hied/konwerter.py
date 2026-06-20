import os
import json
import numpy as np
import cv2
import glob

# Definicja klas
CLASS_MAP = {
    "background": 0,
    "pupil": 1,
    "iris": 2,
    "sclera": 3,
    "skin": 4
}

DRAW_ORDER = {
    "skin": 1,
    "sclera": 2,
    "iris": 3,
    "pupil": 4
}


def convert_labelme_json_to_mask(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob.glob(os.path.join(json_dir, "*.json"))

    print(f"Znaleziono {len(json_files)} plików JSON do konwersji")

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        h = data.get("imageHeight")
        w = data.get("imageWidth")

        if not h or not w:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        shapes = data.get("shapes", [])

        # Sortowanie kształtów według z-index (DRAW_ORDER)
        shapes_sorted = sorted(
            shapes,
            key=lambda x: DRAW_ORDER.get(x["label"].lower(), 0)
        )

        for shape in shapes_sorted:
            label = shape["label"].lower()
            points = shape["points"]

            if label in CLASS_MAP:
                class_id = CLASS_MAP[label]
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], color=class_id)

        base_name = os.path.basename(json_path).replace(".json", ".png")
        output_path = os.path.join(output_dir, base_name)
        cv2.imwrite(output_path, mask)

    print(f"Konwersja zakończona. Pliki zapisano w: {output_dir}")


if __name__ == "__main__":
    INPUT_JSON_DIR = "./dataset/json"
    OUTPUT_MASK_DIR = "./dataset/masks"
    convert_labelme_json_to_mask(INPUT_JSON_DIR, OUTPUT_MASK_DIR)