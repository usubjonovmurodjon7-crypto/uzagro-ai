import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0()

num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 15)

model.load_state_dict(torch.load("../models/uzagro_ai_final.pth"))
model.to(device)
model.eval()

# O'zbekcha kasallik nomlari
classes = [
"Qalampir bakterial dog' kasalligi",
"Qalampir sog'lom",
"Kartoshka erta kuyish kasalligi",
"Kartoshka sog'lom",
"Kartoshka kech kuyish kasalligi",
"Pomidor target spot kasalligi",
"Pomidor mozaika virusi",
"Pomidor sariq barg virus kasalligi",
"Pomidor bakterial dog' kasalligi",
"Pomidor erta kuyish kasalligi",
"Pomidor sog'lom",
"Pomidor kech kuyish kasalligi",
"Pomidor barg mog'or kasalligi",
"Pomidor septoria barg dog'i",
"Pomidor o'rgimchak kana zarari"
]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open("test.jpg")
img = transform(img).unsqueeze(0).to(device)

with torch.no_grad():

    outputs = model(img)

    probs = F.softmax(outputs, dim=1)

    confidence, predicted = torch.max(probs, 1)

kasallik = classes[predicted.item()]
ishonch = confidence.item() * 100

print("Aniqlangan kasallik:", kasallik)
print("Ishonchlilik:", round(ishonch,2), "%")