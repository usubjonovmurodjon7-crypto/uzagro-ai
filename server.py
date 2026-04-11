from fastapi import FastAPI, UploadFile, File
import torch
import torch.nn as nn
from PIL import Image
import io
import torchvision.transforms as transforms
from torchvision import models

app = FastAPI()

device = torch.device("cpu")

# 🔥 MODEL
model = models.efficientnet_b0(weights=None)

NUM_CLASSES = 15  # ✅ TO‘G‘RI

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, NUM_CLASSES)

model.load_state_dict(
    torch.load("models/uzagro_ai_final.pth", map_location=device)
)

model.eval()

# 🔥 CLASS NOMLARI
classes = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_healthy",
    "Potato_Late_blight",
    "Tomato_Target_Spot",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# 🔥 TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output[0], dim=0)

    confidence, predicted = torch.max(probs, 0)

    return {
        "class": classes[predicted.item()],
        "confidence": float(confidence.item())
    }