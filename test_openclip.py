import os
import torch
import open_clip
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

IMAGE_FOLDER = "images"

# Number of top predictions to display
TOP_K = 5

# ============================================================
# LOAD OPENCLIP
# ============================================================

print("Loading OpenCLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained=r"C:\recipe_project\models\ViT-B-32.pt"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print("Model loaded.")
print("Using device:", device)

# ============================================================
# FIND DISH FOLDERS
# ============================================================

dish_names = []

for name in os.listdir(IMAGE_FOLDER):

    path = os.path.join(IMAGE_FOLDER, name)

    if os.path.isdir(path):
        dish_names.append(name)

dish_names.sort()

print()
print("Dishes found:", len(dish_names))

for dish in dish_names:
    print(" -", dish)

# ============================================================
# CREATE TEXT PROMPTS
# ============================================================

prompts = []

for dish in dish_names:

    clean_name = dish.replace("_", " ")

    prompt = f"a photo of {clean_name}"

    prompts.append(prompt)

text_tokens = tokenizer(prompts).to(device)

with torch.no_grad():

    text_features = model.encode_text(text_tokens)

    text_features /= text_features.norm(dim=-1, keepdim=True)

# ============================================================
# TEST IMAGES
# ============================================================

total_images = 0
correct_images = 0

results = []

print()
print("=" * 60)
print("STARTING OPENCLIP TEST")
print("=" * 60)

for actual_dish in dish_names:

    folder = os.path.join(IMAGE_FOLDER, actual_dish)

    image_files = []

    for file in os.listdir(folder):

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        ):
            image_files.append(file)

    print()
    print("Testing:", actual_dish)
    print("Images:", len(image_files))

    dish_correct = 0

    for image_file in image_files:

        image_path = os.path.join(folder, image_file)

        try:

            image = Image.open(image_path).convert("RGB")

            image_input = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():

                image_features = model.encode_image(image_input)

                image_features /= image_features.norm(
                    dim=-1,
                    keepdim=True
                )

                similarity = (
                    image_features @ text_features.T
                )[0]

                probabilities = similarity.softmax(dim=0)

            top_values, top_indices = probabilities.topk(
                TOP_K
            )

            predicted_dish = dish_names[
                top_indices[0].item()
            ]

            is_correct = (
                predicted_dish.lower()
                == actual_dish.lower()
            )

            total_images += 1

            if is_correct:

                correct_images += 1
                dish_correct += 1

            results.append({
                "actual": actual_dish,
                "image": image_file,
                "predicted": predicted_dish,
                "correct": is_correct,
                "confidence": float(top_values[0])
            })

        except Exception as e:

            print(
                "ERROR:",
                image_file,
                "->",
                e
            )

    if len(image_files) > 0:

        accuracy = (
            dish_correct / len(image_files)
        ) * 100

        print(
            f"Accuracy for {actual_dish}: "
            f"{accuracy:.2f}%"
        )

# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print("FINAL OPENCLIP RESULT")
print("=" * 60)

if total_images > 0:

    overall_accuracy = (
        correct_images / total_images
    ) * 100

    print("Total images:", total_images)
    print("Correct:", correct_images)
    print("Incorrect:", total_images - correct_images)

    print(
        f"Overall accuracy: "
        f"{overall_accuracy:.2f}%"
    )

else:

    print("No images were found.")

print()
print("Testing completed.")