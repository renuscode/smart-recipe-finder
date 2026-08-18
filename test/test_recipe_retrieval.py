import pandas as pd

# Recipe dataset
CSV_PATH = r"C:\recipe_project\strong_dishes_recipe_dataset.csv"

# Load dataset
recipes = pd.read_csv(CSV_PATH)

print("Recipe dataset loaded successfully!")
print("Total records:", len(recipes))

# CLIP prediction
predicted_dish = "dosa"

# Find all recipes for the predicted dish
results = recipes[
    recipes["matched_dish"].astype(str).str.strip().str.lower()
    == predicted_dish.lower()
].copy()

print("\n===================================")
print("RECIPE RETRIEVAL")
print("===================================")
print("Predicted dish:", predicted_dish)
print("Matching recipes found:", len(results))

if len(results) == 0:
    print("\nNo recipe found for:", predicted_dish)

else:

    # Words indicating a special/modified dosa
    unwanted_words = [
        "vazhaipoo",
        "bajra",
        "ragi",
        "oats",
        "mushroom",
        "corn",
        "paneer",
        "schezwan",
        "soya",
        "soy",
        "cheesy",
        "onion",
        "wheat",
        "godhuma",
        "godumai",
        "milagai",
        "paniyaram",
        "kanchipuram",
        "mysore",
        "masala"
    ]

    # Convert recipe names to lowercase
    results["name_lower"] = results["name"].astype(str).str.lower()

    # Remove special/modified dosa recipes
    normal_dosa = results[
        ~results["name_lower"].apply(
            lambda name: any(word in name for word in unwanted_words)
        )
    ]

    # If suitable normal dosa recipes exist, use them
    if len(normal_dosa) > 0:
        selected_recipe = normal_dosa.iloc[0]
    else:
        # Fallback: use the first available dosa recipe
        selected_recipe = results.iloc[0]

    print("\n===================================")
    print("SELECTED RECIPE")
    print("===================================")

    print("Recipe:", selected_recipe["name"])
    print("Description:", selected_recipe["description"])
    print("Cuisine:", selected_recipe["cuisine"])
    print("Course:", selected_recipe["course"])
    print("Diet:", selected_recipe["diet"])
    print("Preparation time:", selected_recipe["prep_time"])
    print("Ingredients:", selected_recipe["ingredients"])
    print("Instructions:", selected_recipe["instructions"])
    print("Image URL:", selected_recipe["image_url"])