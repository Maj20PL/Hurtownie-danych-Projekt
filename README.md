# Projekt HIED - Detekcja i Segmentacja Struktur Anatomicznych Oka

## 👁️ O projekcie
Celem projektu jest automatyczne wykrywanie oraz segmentacja struktur anatomicznych ludzkiego oka na podstawie zdjęć. Narzędzie potrafi lokalizować oraz precyzyjnie wydzielać następujące struktury:
* **Źrenica** (zrenica)
* **Tęczówka** (teczowka)
* **Twardówka** (twardowka)
* **Skóra / Powieki** (skora)

Projekt składa się z dwóch niezależnych i uzupełniających się modułów wykorzystujących głębokie uczenie maszynowe (Deep Learning):
1. **Model detekcji (YOLOv8)** – do tworzenia ramek ograniczających (bounding boxes) wokół poszczególnych struktur.
2. **Model segmentacji (U-Net + ResNet34)** – do dokładnego klasyfikowania każdego piksela przypisując mu odpowiednią strukturę oka.

## 🛠️ Wykorzystane technologie i biblioteki
* **Język programowania:** Python 3
* **Głębokie uczenie:** PyTorch, Ultralytics (YOLOv8), Segmentation Models PyTorch (smp)
* **Przetwarzanie obrazów:** OpenCV (cv2), NumPy
* **Augmentacja danych:** Albumentations
* **Przygotowanie danych (anotacja):** LabelMe

## 📊 Zbiór danych
Zbiór danych składał się ze zdjęć, do których ręcznie przygotowano maski segmentacyjne oraz etykiety przy pomocy programu **LabelMe**.
Podczas treningu wykorzystano zaawansowaną augmentację danych (np. `HorizontalFlip`, `RandomBrightnessContrast`), by zwiększyć odporność modelu i zminimalizować efekt przeuczenia.
Zdjęcia przed trafieniem do sieci segmentującej są normalizowane oraz przeskalowywane do rozdzielczości 512x512 pikseli.

## 🧠 Modele
### 1. Model Segmentacji (U-Net z koderem ResNet34)
- **Architektura:** U-Net
- **Backbone (koder):** ResNet34 z wagami wstępnie wytrenowanymi na zbiorze ImageNet.
- **Funkcja straty:** CrossEntropyLoss
- **Optymalizator:** Adam (lr=0.001)
- Model przewiduje 5 klas (4 struktury anatomiczne + tło).
- Kod treningowy znajduje się w `trening_segmentacji.py`.
- Wytrenowane wagi są zapisywane jako `najlepszy_model_segmentacji_oko.pth`.

### 2. Model Detekcji (YOLO)
- Wykorzystano framework YOLO (prawdopodobnie w wersji YOLOv8).
- Model został nauczony wykrywania 4 klas na podstawie wygenerowanych na bazie masek boxów (skrypt `generuj_boxy.py`).
- Konfiguracja dla treningu YOLO znajduje się w `trening_oko.yaml`.
- Kod związany z wykrywaniem obiektów: `trening_wykrywania.py` i `modelu_wykrywania.py`.

## 📁 Struktura projektu
```text
projekt_hied/
│
├── dataset/                         
    ├── images/
    ├── json/   
    ├── lables/
    ├── masks/      
    └── test/
├── runs/
    ├── detect/
       └── YOLO_Oko/                 # Miejsce zapisu treningow YOLO do detekcji oka
├── wyniki_wykrywania/               # Folder zawiera opisany obraz wynikowy detekcji                    
├── trening_segmentacji.py           # Skrypt trenujący model U-Net
├── model_segmentacji.py             # Kod ewaluacji i inferencji modelu segmentacji
├── trening_oko.yaml                 # Plik konfiguracyjny dla modelu wykrywania (YOLO)
├── trening_wykrywania.py            # Skrypt trenujący model YOLO
├── modelu_wykrywania.py         # Skrypt testujący model YOLO na nowych zdjęciach
│
├── generuj_boxy.py                  # Skrypt konwertujący maski na Bounding Boxy dla YOLO
├── konwerter.py                     # Skrypty pomocnicze przy przetwarzaniu danych
│
├── najlepszy_model_segmentacji_oko.pth # Zapisane wagi wytrenowanego modelu segmentacji
└── requirements.txt                 # Lista zależności i bibliotek Pythona
```

## 🚀 Uruchomienie

### Wymagania
Wszystkie niezbędne pakiety znajdują się w pliku `requirements.txt`. Aby je zainstalować, uruchom komendę:
```bash
pip install -r requirements.txt
```

### Trening segmentacji
Aby uruchomić trening modelu segmentacji na własnych danych z folderu `dataset/`:
```bash
python trening_segmentacji.py
```

### Inferencja i testowanie
Dla wykrywania (YOLO):
```bash
python trening_wykrywania.py
python modelu_wykrywania.py
```
Dla segmentacji:
```bash
python model_segmentacji.py
```

## 🎓 Autorzy
Patryk Majewski 198021
Łukasz Zych 197842
Wiktor Gnaczyński 198387

