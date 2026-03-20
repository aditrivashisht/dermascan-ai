
import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
from huggingface_hub import hf_hub_download

# Load model from Hugging Face
print("Loading model...")
model_path = hf_hub_download(
    repo_id="aditrivashishat/dermascan-ai",   # replace with your username
    filename="skin_cancer_resnet50_final.keras"
)
model = tf.keras.models.load_model(model_path)
print("Model loaded!")

class_names = ["Basal Cell Carcinoma", "Melanoma", "Nevus"]
risk_levels = {
    "Melanoma":             "🔴 HIGH RISK",
    "Basal Cell Carcinoma": "🟡 MEDIUM RISK",
    "Nevus":                "🟢 LOW RISK",
}

def predict(image):
    img = Image.fromarray(image).resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds     = model.predict(arr)[0]
    top_class = class_names[np.argmax(preds)]
    risk      = risk_levels[top_class]
    result    = f"## {risk}\n\n**Predicted: {top_class}**\n\n### Confidence Scores:\n"
    for name, score in zip(class_names, preds):
        bar = "█" * int(score * 20)
        result += f"- **{name}**: {score*100:.1f}% {bar}\n"
    result += "\n\n⚠️ *This is an AI tool only. Always consult a dermatologist.*"
    return result

app = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Skin Lesion Image"),
    outputs=gr.Markdown(label="Diagnosis"),
    title="🔬 DermaScan AI",
    description="Upload a dermoscopic image to detect Melanoma, Basal Cell Carcinoma, or Nevus.",
    theme=gr.themes.Soft()
)

app.launch()
