
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader


def ModeliEgit():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 1) Modeli hazırla (ResNet18, 3 sınıf)
    model = models.resnet18(pretrained=True)  # Colab'da genelde bu daha sorunsuz
    num_ftr = model.fc.in_features
    model.fc = nn.Linear(num_ftr, 4)
    model = model.to(device)

    data_root = "/content/drive/MyDrive/resnet_data"

    # İstersen sadece fc eğit (feature extractor)
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False

    # 2) Transformlar
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 1 kanaldan 3 kanala çevir
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # 3) Dataset ve DataLoader'lar
    train_dataset = datasets.ImageFolder(root=f"{data_root}/train", transform=transform)
    val_dataset = datasets.ImageFolder(root=f"{data_root}/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 4) Loss + optimizer (sadece eğitilebilir parametreler)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
    )

    # 5) Eğitim + validation + en iyi modeli kaydetme
    num_epochs = 20
    best_val_acc = 0.0
    best_model_path = "best_resnet18.pth"

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        total_train = 0
        correct_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # ---- Validation ----
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {epoch_loss:.4f}  "
              f"Train Acc: {train_acc * 100:.2f}%  "
              f"Val Acc: {val_acc * 100:.2f}%")

        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Yeni en iyi model kaydedildi: {best_model_path} (Val Acc: {best_val_acc * 100:.2f}%)")

    print(f"Eğitim bitti. En iyi validation accuracy: {best_val_acc * 100:.2f}%")
