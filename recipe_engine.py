"""
recipe_engine.py
Smart Recipe Finder - Recipe Retrieval, Fallback & Search Engine
Uses 'all_varieties_recipe_with_official_source_verification.csv' as the SOLE recipe dataset.
Does NOT modify or regenerate the dataset.
"""

import os
import re
import urllib.parse
import pandas as pd

# Path to the immutable recipe dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "all_varieties_recipe_with_official_source_verification.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Maintainable mapping of related dishes for fallback suggestions
RELATED_DISHES_MAP = {
    "dosa": [
        "plain dosa", "masala dosa", "rava dosa", "set dosa", "onion dosa", "mysore masala dosa"
    ],
    "biryani": [
        "chicken biryani", "hyderabadi biryani", "vegetable biryani", "egg biryani", "mutton biryani"
    ],
    "samosa": [
        "aloo samosa", "vegetable samosa", "paneer samosa", "chicken samosa", "corn samosa"
    ],
    "idli": [
        "plain idli", "rava idli", "fried idli", "thatte idli", "kanchipuram idli"
    ],
    "paratha": [
        "aloo paratha", "plain aloo paratha", "stuffed aloo paratha", "crispy aloo paratha", "buttery aloo paratha"
    ],
    "dhokla": [
        "khaman dhokla", "khatta dhokla", "rava dhokla", "sandwich dhokla", "white dhokla"
    ],
    "paneer": [
        "kadai paneer", "palak paneer", "paneer butter masala", "paneer tikka masala"
    ],
    "kofta": [
        "classic malai kofta", "paneer malai kofta", "vegetable malai kofta"
    ]
}


class RecipeEngine:
    def __init__(self, csv_path=DATASET_PATH):
        self.csv_path = csv_path
        self.df = None
        self.load_dataset()

    def load_dataset(self):
        """Loads all_varieties_recipe_with_official_source_verification.csv without modifying it."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Recipe dataset not found at: {self.csv_path}")

        # Read the CSV as-is
        self.df = pd.read_csv(self.csv_path)

        # Normalize lookup columns in memory for fast, case-insensitive matching
        self.df["matched_dish_clean"] = (
            self.df["matched_dish"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ")
        )
        self.df["predicted_variety_clean"] = (
            self.df["predicted_variety"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace("_", " ")
        )
        self.df["name_clean"] = (
            self.df["name"].astype(str).str.strip().str.lower()
        )

    def parse_ingredients(self, raw_ingredients):
        """
        Parses existing ingredients field into a clean, structured list.
        Returns: [{'amount': '2 cups', 'name': 'Bajra Flour ( Pearl Millet)'}, ...]
        """
        if pd.isna(raw_ingredients) or not str(raw_ingredients).strip():
            return []

        lines = str(raw_ingredients).split("\n")
        ingredients_list = []

        for line in lines:
            # Clean tabs and excessive whitespace
            line = re.sub(r"\s+", " ", line).strip()
            # Clean comma spacing
            line = re.sub(r"\s*,\s*", ", ", line).strip(" ,")

            if not line or len(line) < 2:
                continue

            # Skip section header strings if present
            if line.lower().startswith(
                ("ingredients for", "for the ", "to serve", "main ingredients")
            ):
                continue

            # Regex to cleanly separate leading quantity/measurement from ingredient name
            match = re.match(
                r"^((?:\d+(?:[-/]\d+)?|\d*\.\d+)?\s*(?:cups?|teaspoons?|tsp|tablespoons?|tbsp|sprigs?|grams?|g|kg|pinch|inches?|cloves?|pieces?|bunch|leaves)?)\s*(.*)$",
                line,
                re.IGNORECASE,
            )

            if match and match.group(1).strip() and match.group(2).strip():
                amount = match.group(1).strip()
                name = match.group(2).strip()
            else:
                amount = ""
                name = line

            ingredients_list.append({"amount": amount, "name": name})

        return ingredients_list

    def parse_instructions(self, raw_instructions):
        """Parses existing instructions field into clean, readable numbered steps."""
        if pd.isna(raw_instructions) or not str(raw_instructions).strip():
            return []

        raw = str(raw_instructions)
        # Fix run-on sentences where a dot is directly followed by a capital letter without space
        raw = re.sub(r"\.([A-Z])", r". \1", raw)
        raw = re.sub(r"\s+", " ", raw).strip()

        # Split steps by periods followed by capital letters, numbers, or end of line
        raw_steps = re.split(r"(?<=\.)\s+(?=[A-Z0-9])", raw)
        steps_list = []

        for step in raw_steps:
            step = step.strip()
            # Strip any pre-existing step prefix (e.g. "1.", "Step 1:")
            step = re.sub(r"^(?:\d+\.|\d+\)|\bstep\s*\d+:?)\s*", "", step, flags=re.I)
            if len(step) > 4:
                steps_list.append(step)

        return steps_list

    def resolve_image_url(self, raw_image_url):
        """
        Safely resolves local Windows image paths from CSV for Flask browser display.
        Returns: '/dataset_images/<path>' or None if no valid image exists.
        Does NOT expose raw invalid filesystem paths to browser HTML.
        """
        if pd.isna(raw_image_url) or not str(raw_image_url).strip():
            return None

        paths = [p.strip() for p in str(raw_image_url).split(";") if p.strip()]

        for path in paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path) and os.path.isfile(normalized_path):
                try:
                    rel_path = os.path.relpath(normalized_path, IMAGES_DIR)
                    rel_path_web = rel_path.replace("\\", "/")
                    return f"/dataset_images/{urllib.parse.quote(rel_path_web)}"
                except ValueError:
                    continue

        return None

    def _format_recipe_dict(self, match_row):
        """Helper to format a dataframe row into a clean recipe dictionary."""
        return {
            "found": True,
            "name": str(match_row.get("name", "")).strip(),
            "matched_dish": str(match_row.get("matched_dish", "")).strip(),
            "predicted_variety": str(
                match_row.get("predicted_variety", "")
            ).strip(),
            "description": "" if pd.isna(match_row.get("description")) else str(match_row.get("description", "")).strip(),
            "cuisine": (
                str(match_row.get("cuisine", "Indian")).strip()
                if not pd.isna(match_row.get("cuisine"))
                else "Indian"
            ),
            "course": (
                str(match_row.get("course", "Main Course")).strip()
                if not pd.isna(match_row.get("course"))
                else "Main Course"
            ),
            "diet": (
                str(match_row.get("diet", "Vegetarian")).strip()
                if not pd.isna(match_row.get("diet"))
                else "Vegetarian"
            ),
            "prep_time": (
                str(match_row.get("prep_time", "30 Minutes")).strip()
                if not pd.isna(match_row.get("prep_time"))
                else "30 Minutes"
            ),
            "image_url": self.resolve_image_url(match_row.get("image_url")),
            "ingredients": self.parse_ingredients(match_row.get("ingredients")),
            "instructions": self.parse_instructions(match_row.get("instructions")),
        }

    def get_similar_dishes(self, dish=None, variety=None, top_predictions=None, max_cards=4):
        """
        Retrieves similar/alternative dish flashcards:
        1. Uses alternative predictions from top_predictions (2nd and 3rd).
        2. Uses RELATED_DISHES_MAP.
        3. Uses other varieties of the same dish class in the CSV.
        Verifies every dish exists in the CSV before adding.
        Returns: list of 1 to max_cards (3-5) distinct flashcards.
        """
        if self.df is None or len(self.df) == 0:
            return []

        flashcards = []
        seen_recipes = set()

        def add_candidate(candidate_name, confidence=None):
            if not candidate_name or len(flashcards) >= max_cards:
                return
            cand_clean = candidate_name.strip().lower().replace("_", " ")

            # Check matching in CSV (prefer variety, then dish, then name)
            matched_df = self.df[self.df["predicted_variety_clean"] == cand_clean]
            if matched_df.empty:
                matched_df = self.df[
                    (self.df["matched_dish_clean"] == cand_clean)
                    | (self.df["matched_dish_clean"].str.contains(re.escape(cand_clean)))
                ]
            if matched_df.empty:
                matched_df = self.df[self.df["name_clean"].str.contains(re.escape(cand_clean))]

            if not matched_df.empty:
                row = matched_df.iloc[0]
                recipe_name = str(row["name"]).strip()
                if recipe_name not in seen_recipes:
                    seen_recipes.add(recipe_name)
                    flashcards.append({
                        "name": recipe_name,
                        "dish": str(row["matched_dish"]).strip(),
                        "variety": str(row.get("predicted_variety", "")).strip(),
                        "cuisine": str(row.get("cuisine", "Indian")).strip(),
                        "prep_time": str(row.get("prep_time", "30 Minutes")).strip(),
                        "confidence": confidence,
                        "image_url": self.resolve_image_url(row.get("image_url")),
                        "recipe_url": f"/recipe?dish={urllib.parse.quote(str(row['matched_dish']).strip())}&variety={urllib.parse.quote(str(row.get('predicted_variety', '')).strip())}"
                    })

        # 1. Use alternative predictions (e.g. 2nd and 3rd predictions)
        if top_predictions and isinstance(top_predictions, list):
            for pred in top_predictions:
                if isinstance(pred, dict):
                    p_name = pred.get("name") or pred.get("variety")
                    p_conf = pred.get("match_percent") or pred.get("confidence")
                    add_candidate(p_name, p_conf)
                elif isinstance(pred, str):
                    add_candidate(pred, None)

        # 2. Check RELATED_DISHES_MAP
        target_keys = []
        if dish:
            target_keys.append(dish.strip().lower().replace("_", " "))
        if variety:
            target_keys.append(variety.strip().lower().replace("_", " "))

        for k in target_keys:
            for map_key, related_list in RELATED_DISHES_MAP.items():
                if map_key in k or k in map_key:
                    for rel in related_list:
                        add_candidate(rel)

        # 3. Check other varieties for the same dish class in the CSV
        if len(flashcards) < max_cards and dish:
            d_clean = dish.strip().lower().replace("_", " ")
            same_dish_df = self.df[
                (self.df["matched_dish_clean"] == d_clean)
                | (self.df["matched_dish_clean"].str.contains(re.escape(d_clean)))
            ]
            for _, row in same_dish_df.iterrows():
                v_name = str(row.get("predicted_variety", "")).strip()
                add_candidate(v_name)
                if len(flashcards) >= max_cards:
                    break

        return flashcards[:max_cards]

    def search_recipes(self, query, max_results=10):
        """
        Searches the existing recipe CSV across relevant fields:
        'name', 'matched_dish', 'predicted_variety', 'cuisine'.
        Returns matching recipe card items.
        """
        if not query or not str(query).strip() or self.df is None:
            return []

        q = str(query).strip().lower()
        # Case-insensitive substring search across clean columns
        matches = self.df[
            self.df["name_clean"].str.contains(re.escape(q))
            | self.df["matched_dish_clean"].str.contains(re.escape(q))
            | self.df["predicted_variety_clean"].str.contains(re.escape(q))
            | self.df["cuisine"].astype(str).str.lower().str.contains(re.escape(q))
        ]

        results = []
        for _, row in matches.head(max_results).iterrows():
            results.append({
                "name": str(row["name"]).strip(),
                "dish": str(row["matched_dish"]).strip(),
                "variety": str(row.get("predicted_variety", "")).strip(),
                "cuisine": str(row.get("cuisine", "Indian")).strip(),
                "prep_time": str(row.get("prep_time", "30 Minutes")).strip(),
                "diet": str(row.get("diet", "Vegetarian")).strip(),
                "image_url": self.resolve_image_url(row.get("image_url")),
                "recipe_url": f"/recipe?dish={urllib.parse.quote(str(row['matched_dish']).strip())}&variety={urllib.parse.quote(str(row.get('predicted_variety', '')).strip())}"
            })

        return results

    def get_recipe(self, query=None, dish=None, variety=None, top_predictions=None):
        """
        Retrieves recipe:
        1. Prefers matching 'predicted_variety'
        2. Falls back to matching 'matched_dish'
        3. Falls back to partial match in recipe 'name'
        4. If not found, attaches available alternative flashcards via get_similar_dishes.
        """
        if self.df is None or len(self.df) == 0:
            return {
                "found": False,
                "message": "Recipe dataset is not available or empty.",
                "alternatives": []
            }

        search_variety = (variety or query or "").strip().lower().replace("_", " ")
        search_dish = (dish or query or "").strip().lower().replace("_", " ")

        match_row = None

        # 1. Prefer matching predicted_variety
        if search_variety:
            matched_df = self.df[self.df["predicted_variety_clean"] == search_variety]
            if not matched_df.empty:
                match_row = matched_df.iloc[0]

        # 2. Fallback: match matched_dish
        if match_row is None and search_dish:
            matched_df = self.df[
                (self.df["matched_dish_clean"] == search_dish)
                | (self.df["matched_dish_clean"].str.contains(re.escape(search_dish)))
            ]
            if not matched_df.empty:
                match_row = matched_df.iloc[0]

        # 3. Fallback: partial search on name
        if match_row is None and (search_variety or search_dish):
            term = search_variety or search_dish
            matched_df = self.df[self.df["name_clean"].str.contains(re.escape(term))]
            if not matched_df.empty:
                match_row = matched_df.iloc[0]

        # 4. Recipe Not Found -> Gather alternatives
        if match_row is None:
            display_name = variety or dish or query or "Unknown Dish"
            alternatives = self.get_similar_dishes(
                dish=dish,
                variety=variety,
                top_predictions=top_predictions,
                max_cards=4
            )
            return {
                "found": False,
                "query": display_name,
                "message": f"Recipe unavailable for \"{display_name}\".",
                "alternatives": alternatives
            }

        # Format and return the found recipe
        return self._format_recipe_dict(match_row)


# Singleton instance
recipe_engine = RecipeEngine()

