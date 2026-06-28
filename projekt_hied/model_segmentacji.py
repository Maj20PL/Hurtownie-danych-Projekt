import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# KONFIGURACJA
MODEL_PATH = "najlepszy_model_segmentacji_oko.pth"
TEST_IMAGE_PATH = "dataset/test/test_oka_Patryk.png"

# Mapowanie kolorów (RGB) dla klas:
COLORS = np.array([
    [0, 0, 0],  # 0 - Tło
    [255, 0, 0],  # 1 - Pupil (Źrenica)
    [0, 255, 0],  # 2 - Iris (Tęczówka)
    [0, 0, 255],  # 3 - Sclera (Twardówka)
    [255, 255, 0]  # 4 - Skin (Skóra)
], dtype=np.uint8)


def test_model():
    device = torch.device("cpu")
    print("Ładowanie modelu...")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=5,
    )

    # Wczytanie wag z pliku .pth
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()  # Przełączenie w tryb ewaluacji (testowania)

    # Wczytanie i przygotowanie zdjęcia testowego
    image = cv2.imread(TEST_IMAGE_PATH)
    if image is None:
        print(f"\n[BŁĄD] Nie znaleziono zdjęcia pod adresem: {TEST_IMAGE_PATH}")
        print("Utwórz folder 'dataset/test/' i wrzuć tam zdjęcie 'nowe_oko.png'.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]

    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])

    input_tensor = transform(image=image_rgb)['image'].unsqueeze(0).to(device)

    print("Przetwarzanie obrazu przez sieć...")
    with torch.no_grad():
        output = model(input_tensor)
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Skalowanie wygenerowanej maski z powrotem do rozmiaru oryginalnego zdjęcia
    predicted_mask_resized = cv2.resize(predicted_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # Nakładanie kolorów na przewidziane piksele
    color_mask = COLORS[predicted_mask_resized]

    alpha = 0.5
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, color_mask, alpha, 0)

    # Wyświetlanie wyniku za pomocą Matplotlib
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Oryginalne zdjęcie")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Wygenerowana Maska za pomocą modelu")
    plt.imshow(color_mask)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Nałożenie na zdjęcie")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Instalacja biblioteki matplotlib w razie potrzeby: pip install matplotlib
    test_model()