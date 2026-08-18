import os
import torch
import open_clip
from PIL import Image

# ==============================
# SETTINGS
# ==============================

IMAGE_DATASET = r"C:\recipe_project\images"
MODEL_PATH = r"C:\recipe_project\models\ViT-B-32.pt"

# ==============================
# LOAD OPENCLIP
# ==============================

print("Loading OpenCLIP ViT-B/32...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained=MODEL_PATH,
    weights_only=False
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

device = "cpu"
model = model.to(device)
model.eval()

print("OpenCLIP loaded successfully!")
print("Device:", device)

# ==============================
# GET DISH NAMES FROM FOLDERS
# ==============================

dish_names = sorted([
    folder for folder in os.listdir(IMAGE_DATASET)
    if os.path.isdir(os.path.join(IMAGE_DATASET, folder))
])

print("\nNumber of dish classes:", len(dish_names))
print("Dish classes:", dish_names)

# ==============================
# CREATE TEXT FEATURES
# ==============================

print("\nCreating text features...")

texts = [f"a photo of {dish}" for dish in dish_names]

text_tokens = tokenizer(texts)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)

text_features /= text_features.norm(dim=-1, keepdim=True)

# ==============================
# TEST ALL IMAGES
# ==============================

total_images = 0
correct = 0
failed = 0

print("\n======================================")
print("TESTING ALL DATASET IMAGES")
print("======================================")

for dish in dish_names:

    dish_folder = os.path.join(IMAGE_DATASET, dish)

    image_files = [
        f for f in os.listdir(dish_folder)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    print(f"\nTesting {dish}: {len(image_files)} images")

    for image_file in image_files:

        image_path = os.path.join(dish_folder, image_file)

        try:
            image = preprocess(Image.open(image_path).convert("RGB"))
            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)

            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = image_features @ text_features.T

            predicted_index = similarity.argmax(dim=-1).item()
            predicted_dish = dish_names[predicted_index]

            total_images += 1

            if predicted_dish.lower() == dish.lower():
                correct += 1
            else:
                failed += 1

            print(
                f"{image_file} | "
                f"Actual: {dish} | "
                f"Predicted: {predicted_dish}"
            )

        except Exception as e:
            print(f"ERROR: {image_path}")
            print(e)

# ==============================
# FINAL RESULT
# ==============================

print("\n======================================")
print("FINAL OPENCLIP DATASET TEST RESULT")
print("======================================")

print("Total images tested :", total_images)
print("Correct predictions :", correct)
print("Wrong predictions   :", failed)

if total_images > 0:
    accuracy = (correct / total_images) * 100
    print(f"Accuracy            : {accuracy:.2f}%")

print("\nOpenCLIP dataset testing completed.")