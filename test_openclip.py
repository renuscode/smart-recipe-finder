import os
import csv
import torch
import open_clip
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

IMAGE_FOLDER = "images"

MODEL_PATH = r"C:\recipe_project\models\ViT-B-32.pt"

OUTPUT_FILE = "openclip_predictions.csv"

TOP_K = 5

# ============================================================
# START
# ============================================================

print("=" * 60)
print("SMART RECIPE FINDER - OPENCLIP IMAGE RECOGNITION")
print("=" * 60)

# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print()
print("Using device:", device)

# ============================================================
# LOAD TORCHSCRIPT MODEL
# ============================================================

print()
print("Loading TorchScript OpenCLIP model...")

model = torch.jit.load(
    MODEL_PATH,
    map_location=device
)

model.eval()

print("Model loaded successfully.")

# ============================================================
# OPENCLIP PREPROCESSING
# ============================================================

print()
print("Loading OpenCLIP preprocessing...")

# Create a temporary model configuration
# only to obtain the standard ViT-B-32 preprocessing.
_, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained=None
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")

print("Preprocessing loaded.")

# ============================================================
# FIND DISH FOLDERS
# ============================================================

dish_names = []

for name in os.listdir(IMAGE_FOLDER):

    path = os.path.join(
        IMAGE_FOLDER,
        name
    )

    if os.path.isdir(path):

        dish_names.append(name)

dish_names.sort()

print()
print("Number of dish classes:", len(dish_names))

print()
print("Dish classes:")

for dish in dish_names:
    print(" -", dish)

# ============================================================
# CREATE TEXT PROMPTS
# ============================================================

print()
print("Creating text prompts...")

prompts = []

for dish in dish_names:

    clean_name = dish.replace("_", " ")

    prompt = f"a photo of {clean_name}"

    prompts.append(prompt)

text_tokens = tokenizer(
    prompts
).to(device)

# ============================================================
# CREATE TEXT FEATURES
# ============================================================

print("Creating text features...")

with torch.no_grad():

    text_features = model.encode_text(
        text_tokens
    )

    text_features /= text_features.norm(
        dim=-1,
        keepdim=True
    )

print("Text features created.")

# ============================================================
# PROCESS IMAGES
# ============================================================

results = []

total_images = 0

correct_images = 0

print()
print("=" * 60)
print("STARTING IMAGE RECOGNITION")
print("=" * 60)

for actual_dish in dish_names:

    folder = os.path.join(
        IMAGE_FOLDER,
        actual_dish
    )

    image_files = []

    for file in os.listdir(folder):

        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp"
            )
        ):

            image_files.append(file)

    image_files.sort()

    print()
    print(
        f"Processing {actual_dish} "
        f"({len(image_files)} images)"
    )

    for image_file in image_files:

        image_path = os.path.join(
            folder,
            image_file
        )

        try:

            # ------------------------------------------------
            # LOAD IMAGE
            # ------------------------------------------------

            image = Image.open(
                image_path
            ).convert("RGB")

            image_input = preprocess(
                image
            ).unsqueeze(0).to(device)

            # ------------------------------------------------
            # IMAGE FEATURES
            # ------------------------------------------------

            with torch.no_grad():

                image_features = model.encode_image(
                    image_input
                )

                image_features /= image_features.norm(
                    dim=-1,
                    keepdim=True
                )

                # ------------------------------------------------
                # SIMILARITY
                # ------------------------------------------------

                similarity = (
                    image_features @ text_features.T
                )[0]

                probabilities = similarity.softmax(
                    dim=0
                )

            # ------------------------------------------------
            # TOP PREDICTIONS
            # ------------------------------------------------

            top_values, top_indices = probabilities.topk(
                min(TOP_K, len(dish_names))
            )

            predicted_dish = dish_names[
                top_indices[0].item()
            ]

            confidence = float(
                top_values[0]
            )

            # ------------------------------------------------
            # CHECK
            # ------------------------------------------------

            is_correct = (
                predicted_dish.lower()
                ==
                actual_dish.lower()
            )

            if is_correct:

                correct_images += 1

            total_images += 1

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            results.append({

                "image_path":
                    os.path.join(
                        actual_dish,
                        image_file
                    ),

                "actual_dish":
                    actual_dish,

                "predicted_dish":
                    predicted_dish,

                "confidence":
                    confidence,

                "correct":
                    is_correct
            })

            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            print(
                f"  {image_file} "
                f"-> {predicted_dish} "
                f"({confidence:.4f})"
            )

        except Exception as e:

            print(
                f"  ERROR: {image_file} -> {e}"
            )

# ============================================================
# SAVE CSV
# ============================================================

print()
print("=" * 60)
print("SAVING PREDICTIONS")
print("=" * 60)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image_path",
            "actual_dish",
            "predicted_dish",
            "confidence",
            "correct"
        ]
    )

    writer.writeheader()

    writer.writerows(results)

# ============================================================
# FINAL ACCURACY
# ============================================================

print()
print("=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(
    "Total images:",
    total_images
)

print(
    "Correct:",
    correct_images
)

print(
    "Incorrect:",
    total_images - correct_images
)

if total_images > 0:

    accuracy = (
        correct_images /
        total_images
    ) * 100

    print(
        f"Accuracy: {accuracy:.2f}%"
    )

print()
print(
    "Prediction file:",
    OUTPUT_FILE
)

print()
print("=" * 60)
print("OPENCLIP TEST COMPLETED")
print("=" * 60)