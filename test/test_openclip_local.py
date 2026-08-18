import open_clip
import torch

MODEL_PATH = r"C:\recipe_project\models\ViT-B-32.pt"

print("Loading local OpenCLIP ViT-B/32...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained=MODEL_PATH,
    weights_only=False
)

model.eval()

print("OPENCLIP MODEL LOADED SUCCESSFULLY")
print("Model: ViT-B-32")
print("Device: CPU")