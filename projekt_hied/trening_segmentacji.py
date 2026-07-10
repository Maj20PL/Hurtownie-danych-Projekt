import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim


# Wczytanie danych
class EyeSegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Brak zdjęcia {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Brak maski {mask_path}")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.to(torch.long)
        return image, mask


if __name__ == "__main__":
    # Transformacje
    train_transform = A.Compose([
        A.Resize(height=512, width=512),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2()
    ])

    train_dataset = EyeSegmentationDataset(
        images_dir="./dataset/images",
        masks_dir="./dataset/masks",
        transform=train_transform
    )

    # Przygotowanie narzędzi
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    device = torch.device("cpu")
    print(f"Trening uruchomiony na urządzeniu: {device}")

    # Inicjalizacja modelu
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Główna pętla treningowa
    EPOCHS = 60
    best_loss = float('inf')
    save_path = "najlepszy_model_segmentacji_oko.pth"

    print(f"\n--- ROZPOCZYNAM TRENING ({EPOCHS} EPOK) ---")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(
                f"Epoka [{epoch + 1}/{EPOCHS}] | Batch [{batch_idx + 1}/{len(train_loader)}] | Strata: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        print(f">>> Podsumowanie epoki {epoch + 1}: Średnia strata = {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f" ---> Wykryto postęp! Zapisano wagi do pliku (Nowy rekord: {best_loss:.4f})\n")
        else:
            print("\n")

    print("--- TRENING ZAKOŃCZONY ---")
    print(f"Model zapisano jako: {save_path}")