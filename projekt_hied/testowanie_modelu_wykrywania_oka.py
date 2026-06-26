from ultralytics import YOLO

def test_yolo():
    print("Wczytywanie wytrenowanego modelu...")
    model = YOLO('YOLO_Oko/trening1/weights/best.pt')

    print("Analiza nowego zdjęcia...")
    # YOLO automatycznie przeanalizuje zdjęcie, narysuje boxy i zapisze wynik
    results = model('dataset/test/nowe_oko.png', save=True, show=True)

    print("\nWynikowe zdjęcie zostało zapisane w folderze: runs/detect/predict")

if __name__ == '__main__':
    test_yolo()