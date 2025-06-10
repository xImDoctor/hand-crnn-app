# streamlit_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
import cv2
from PIL import Image
from model import CRNN, VOCAB, idx2char, ctc_greedy_decode
from text_segmentation import segment_text
from torchvision import transforms
import editdistance
import matplotlib.pyplot as plt
import io
import tempfile

# Инициализация модели
@st.cache_resource
def load_model():
    model = CRNN(num_classes=len(VOCAB))
    model.load_state_dict(torch.load("model/resnet-word-trained-ver2.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict_image(model, img_pil):
    img_tensor = transform(img_pil).unsqueeze(0)  # [1, 1, H, W]
    with torch.no_grad():
        logits = model(img_tensor)
        pred_idxs = torch.argmax(logits, dim=-1).permute(1, 0).tolist()
        decoded = ctc_greedy_decode(pred_idxs, idx2char)
    return decoded[0]

def process_image_from_path(image_path, visualize=False):
    original, word_boxes = segment_text(image_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    results = []

    vis = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR) if visualize else None

    for line in word_boxes:
        for (x, y, w, h) in line:
            word_img = image[y:y+h, x:x+w]
            word_pil = Image.fromarray(word_img).convert("L")
            pred = predict_image(model, word_pil)
            results.append(pred)

            if visualize:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if visualize:
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(vis_rgb)
        ax.axis("off")
        st.pyplot(fig)

    return " ".join(results)

st.title("Распознавание рукописного текста")
st.write("*(на русском языке)*")

option = st.radio("Выберите режим ввода:", ["Нарисовать текст", "Загрузить изображение"])

if option == "Нарисовать текст":
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=3,
        stroke_color="black",
        background_color="white",
        height=150,
        width=500,
        drawing_mode="freedraw",
        key="canvas_key",
    )
    if canvas_result.image_data is not None:
        img_data = canvas_result.image_data[:, :, 0].astype(np.uint8)
        img_pil = Image.fromarray(img_data).convert("L")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            img_pil.save(tmp_path)

        show_boxes = st.checkbox("Показать боксы слов на изображении")

        if st.button("Распознать текст"):
            st.session_state.canvas_prediction = process_image_from_path(tmp_path, visualize=show_boxes)

        if "canvas_prediction" in st.session_state:
            prediction = st.session_state.canvas_prediction
            st.success(f"Распознанный текст: {prediction}")
            gt_text = st.text_input("Введите правильный текст (для оценки качества)", key="gt_canvas")
            if gt_text:
                cer = editdistance.eval(prediction, gt_text) / max(len(gt_text), 1)
                st.metric("CER (ошибка по символам)", f"{cer:.4f}")

elif option == "Загрузить изображение":
    uploaded_file = st.file_uploader("Загрузите изображение PNG/JPEG", type=["png", "jpg", "jpeg"])
    show_boxes = st.checkbox("Показать боксы слов на изображении")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Загруженное изображение", use_container_width=True)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        if st.button("Распознать текст на изображении"):
            st.session_state.image_prediction = process_image_from_path(tmp_path, visualize=show_boxes)

        if "image_prediction" in st.session_state:
            final_text = st.session_state.image_prediction
            st.success(f"Распознанный текст: {final_text}")
            gt_text = st.text_input("Введите правильный текст (для оценки качества)", key="gt_image")
            if gt_text:
                cer = editdistance.eval(final_text, gt_text) / max(len(gt_text), 1)
                st.metric("CER (ошибка по символам)", f"{cer:.4f}")
