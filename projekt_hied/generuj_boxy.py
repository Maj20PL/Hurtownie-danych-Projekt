import os
import cv2
import numpy as np
import glob

# KONFIGURACJA ŚCIEŻEK
MASKS_DIR = "./dataset/masks"  # Folder z maskami
OUTPUT_LABELS_DIR = "./dataset/labels"  # Gotowe pliki .txt

os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# Mapowanie klas:
CLASS_MAPPING = {
    1: 0,  # Pupil -> YOLO class 0
    2: 1,  # Iris  -> YOLO class 1
    3: 2,  # Sclera -> YOLO class 2
    4: 3  # Skin  -> YOLO class 3
}


def generate_yolo_bboxes_from_masks():
    mask_files = glob.glob(os.path.join(MASKS_DIR, "*.png"))
    print(f"Rozpoczęcie analizy {len(mask_files)} masek...\n")

    for mask_path in mask_files:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        img_height, img_width = mask.shape

        base_name = os.path.basename(mask_path).replace(".png", ".txt")
        label_path = os.path.join(OUTPUT_LABELS_DIR, base_name)

        unique_classes = np.unique(mask)
        bboxes = []

        for cls_val in unique_classes:
            if cls_val == 0 or cls_val not in CLASS_MAPPING:
                continue

            yolo_class_id = CLASS_MAPPING[cls_val]

            y_indices, x_indices = np.where(mask == cls_val)

            if len(x_indices) == 0 or len(y_indices) == 0:
                continue

            x_min = np.min(x_indices)
            x_max = np.max(x_indices)
            y_min = np.min(y_indices)
            y_max = np.max(y_indices)

            box_width = x_max - x_min
            box_height = y_max - y_min

            x_center = x_min + (box_width / 2.0)
            y_center = y_min + (box_height / 2.0)

            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = box_width / img_width
            height_norm = box_height / img_height

            bbox_line = f"{yolo_class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            bboxes.append(bbox_line)

        with open(label_path, "w", encoding="utf-8") as f:
            for box in bboxes:
                f.write(box + "\n")

    print(f"Zapisano pliki tekstowe w folderze: {OUTPUT_LABELS_DIR}")


if __name__ == "__main__":
    generate_yolo_bboxes_from_masks()