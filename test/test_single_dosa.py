import torch
import clip
from PIL import Image

# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = r"C:\recipe_project\models\ViT-B-32.pt"

IMAGE_PATH = r"C:\recipe_project\test_images\dosa_opencv_resized.jpg"

# --------------------------------------------------
# LOAD CLIP MODEL
# --------------------------------------------------

print("Loading OpenAI CLIP ViT-B/32...")

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load(
    MODEL_PATH,
    device=device,
    jit=False
)

model.eval()

print("CLIP MODEL LOADED SUCCESSFULLY")
print("Device:", device)

# --------------------------------------------------
# DISH NAMES
# --------------------------------------------------

dish_names = [
    "dosa",
    "idli",
    "vada",
    "poha",
    "kachori",
    "bhatura",
    "chicken tikka masala",
    "paneer butter masala",
    "kadai paneer"
]

# Create text descriptions
text = clip.tokenize(
    [f"a photo of {dish}" for dish in dish_names]
).to(device)

# --------------------------------------------------
# LOAD IMAGE
# --------------------------------------------------

print("\nLoading image:")

image = Image.open(IMAGE_PATH).convert("RGB")

print("Image:", IMAGE_PATH)
print("Original image size:", image.size)

image_input = preprocess(image).unsqueeze(0).to(device)

# --------------------------------------------------
# CLIP PREDICTION
# --------------------------------------------------

with torch.no_grad():

    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    probabilities = similarity[0]

# --------------------------------------------------
# TOP PREDICTIONS
# --------------------------------------------------

results = []

for dish, probability in zip(dish_names, probabilities):
    results.append(
        (dish, probability.item() * 100)
    )

results.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\n===================================")
print("CLIP PREDICTIONS")
print("===================================")

for i, (dish, probability) in enumerate(results, start=1):
    print(f"{i}. {dish} ({probability:.2f}%)")

print("\n===================================")
print("FINAL PREDICTION:", results[0][0])
print("CONFIDENCE:", f"{results[0][1]:.2f}%")
print("===================================")