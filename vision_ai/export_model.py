import torch
import torchvision.models as models

# Modelni yuklash
model = models.efficientnet_b0()

num_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_features, 15)

model.load_state_dict(torch.load("../models/uzagro_ai_final.pth", map_location="cpu"))
model.eval()

# Dummy input (224x224 rasm)
example = torch.rand(1, 3, 224, 224)

# TorchScript ga o‘tkazish
traced_script_module = torch.jit.trace(model, example)

# Saqlash
traced_script_module.save("../models/uzagro_ai_mobile.pt")

print("Mobil model saqlandi: uzagro_ai_mobile.pt")