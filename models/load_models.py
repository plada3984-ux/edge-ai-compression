import torch
import torchvision.models as tv_models
import timm

# тут все модели которые нам нужны
# ключ - короткое имя, значение - лямбда которая создаёт модель
MODELS = {
    "resnet18": lambda: tv_models.resnet18(weights="IMAGENET1K_V1"),
    "mobilenetv3_large": lambda: tv_models.mobilenet_v3_large(weights="IMAGENET1K_V1"),
    "repvit_m1": lambda: timm.create_model("repvit_m1.dist_in1k", pretrained=True),
    "fastvit_t8": lambda: timm.create_model("fastvit_t8.apple_in1k", pretrained=True),
}


def load_model(name, device="cuda"):
    # загружаем модель по имени
    if name not in MODELS:
        print("нет такой модели:", name)
        print("доступные:", list(MODELS.keys()))
        return None

    model = MODELS[name]()
    model = model.to(device)
    model.eval()  # eval режим - чтобы batchnorm и dropout не мешали
    return model


def list_models():
    return list(MODELS.keys())


def model_info(name):
    # грузим на cpu чтобы не занимать видюху
    model = load_model(name, device="cpu")

    if model is None:
        return

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Модель:", name)
    print("  Всего параметров:", total)
    print("  Обучаемых:", trainable)
    # 4 байта на float32
    print("  Примерный размер:", round(total * 4 / 1024 / 1024, 2), "МБ")
    print()


if __name__ == "__main__":
    for name in list_models():
        model_info(name)