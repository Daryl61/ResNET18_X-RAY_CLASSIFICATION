import os
import torch
import torch.nn as nn
from anaconda_navigator.utils.telemetry import ANALYTICS
from torchvision import models, datasets, transforms
from PIL import Image
import streamlit as st


st.set_page_config(page_icon="📊",menu_items=None,
                   initial_sidebar_state="expanded", page_title="X-RAY ANALYTICS" )



# ---------------- AYARLAR -----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# BURALARI KENDİNE GÖRE DÜZENLE
data_root = "./resnet_data"         # train/val klasörlerinin olduğu yer
best_model_path = "./best_resnet18.pth"  # eğittiğin model dosyası

# -------------------------------------------

@st.cache_resource
def load_model_and_classes():
    # Train'de kullandığın sınıf isimlerini al
    train_dataset = datasets.ImageFolder(root=os.path.join(data_root, "train"))
    class_names = train_dataset.classes
    num_classes = len(class_names)

    # Modeli kur
    model = models.resnet18(pretrained=False)
    num_ftr = model.fc.in_features
    model.fc = nn.Linear(num_ftr, num_classes)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Transform (siyah-beyaz -> 3 kanal)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, class_names, transform


def predict_image(file, model, class_names, transform):
    img = Image.open(file).convert("L")
    img = transform(img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    pred_class = class_names[prediction.item()]
    return pred_class, confidence.item()


def main():
    st.title("ResNet18 Image X-RAY Classification")
    st.write("Model:ResNet18-FineTuning")
    st.write("Model covid-19, pneumonıa ve tuberculosıs analizleri yapabilir")
    model, class_names, transform = load_model_and_classes()

    yüklenen =uploaded_file = st.file_uploader("Bir resim yükle", type=["jpg", "jpeg", "png"])

    if yüklenen is not None:
        st.image(yüklenen, caption="Yüklenen resim", use_column_width=True)

        if st.button("Tahmin Et"):
            pred_class, conf = predict_image(yüklenen, model, class_names, transform)
            st.write(f"**Tahmin:** {pred_class}")
            st.write(f"**Güven:** {conf*100:.2f} %")


if __name__ == "__main__":
    main()