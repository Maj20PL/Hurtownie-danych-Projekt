import cv2
from ultralytics import YOLO


def draw_custom_boxes():
    print("Wczytywanie modelu...")
    model = YOLO('runs/detect/YOLO_Oko/trening1/weights/best.pt')
    img_path = 'dataset/test/test_oka_Patryk.png'

    results = model(img_path, verbose=False)[0]

    img = cv2.imread(img_path)

    # 0: zrenica, 1: teczowka, 2: twardowka, 3: skora
    COLORS = {
        0: (0, 0, 255),  # Czerwony dla źrenicy
        1: (0, 255, 0),  # Zielony dla tęczówki
        2: (255, 0, 0),  # Niebieski dla twardówki
        3: (0, 255, 255)  # Żółty dla skóry
    }

    print("Rysowanie boxów...")
    for box in results.boxes:
        # Wyciągnięcie współrzędnych, ID klasy i pewności modelu (Confidence)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        name = results.names[cls_id].capitalize()
        color = COLORS.get(cls_id, (255, 255, 255))

        # Rysowanie ramki
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"

        if cls_id == 0:
            y_text = y2 + 20
        else:
            y_text = y1 - 10

        # Zabezpieczenie, by napis nie wyszedł poza górną krawędź zdjęcia
        y_text = max(20, y_text)

        # Rysowanie wypełnionego tła pod tekstem
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y_text - h - 5), (x1 + w, y_text + 5), color, -1)

        # Rysowanie samego tekstu
        text_color = (0, 0, 0) if cls_id in [1, 3] else (255, 255, 255)
        cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Zapisanie i wyświetlenie wyniku
    output_path = "wyniki_wykrywanie/oko_wykryte.png"
    cv2.imwrite(output_path, img)
    print(f"Zapisano jako: {output_path}")

    cv2.imshow("Custom YOLO", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    draw_custom_boxes()