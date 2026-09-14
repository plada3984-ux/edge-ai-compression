import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Пути (абсолютные, через __file__)
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "imagenette", "imagenette2-320")
TRAIN_DIR = os.path.join(ROOT, "train")
VAL_DIR = os.path.join(ROOT, "val")
OUTPUT_DIR = os.path.join(BASE, "imagenette", "tensors")

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def process_split(split_dir, split_name):
    print(f"Обработка {split_name}...")
    dataset = ImageFolder(split_dir, transform=transform)
    print(f"  Найдено изображений: {len(dataset)}")
    print(f"  Классы: {dataset.classes}")

    loader = DataLoader(dataset, batch_size=500, shuffle=False, num_workers=0)

    all_images = []
    all_labels = []

    for i, (imgs, labels) in enumerate(loader):
        all_images.append(imgs)
        all_labels.append(labels)
        print(f"  Батч {i+1}: {imgs.shape}")

    images = torch.cat(all_images, dim=0)
    labels = torch.cat(all_labels, dim=0)

    output_path = os.path.join(OUTPUT_DIR, f"{split_name}.pt")
    torch.save({
        "images": images,
        "labels": labels,
        "class_to_idx": dataset.class_to_idx,   # <-- сохраняем маппинг классов
    }, output_path)

    print(f"  Сохранено: {output_path}")
    print(f"  Размер: {images.shape}, метки: {labels.shape}")


if __name__ == "__main__":
    process_split(TRAIN_DIR, "train")
    process_split(VAL_DIR, "val")
    print("Готово!")