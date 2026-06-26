from ultralytics import YOLO

def train_yolo():
    print("Inicjalizacja modelu YOLOv8 Nano...")
    model = YOLO('yolov8n.pt')

    print("\n--- ROZPOCZECIE TRENINGU YOLO (50 EPOK) ---")
    # Trening
    results = model.train(
        data='trening_oko.yaml',     # Mapa klas
        epochs=50,           # Liczba epok
        imgsz=512,           # Rozmiar obrazów
        device='cpu',        # Wybranie procesora (nie działą karta)
        batch=15,            # Liczba zdjęć naraz
        project='YOLO_Oko',  # Gdzie zapisać wyniki
        name='trening1'      # Nazwa próby
    )
    print("--- TRENING ZAKOŃCZONY ---")
    print("Wytrenowany model znajduje się w: YOLO_Oko/trening1/weights/best.pt")

if __name__ == '__main__':
    train_yolo()