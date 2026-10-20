
"""Smart Recipe Finder recipe retrieval, parsing, search, and fallbacks.

The source recipe CSV is read-only. Lookup helper columns are
created in memory and are not written back to the CSV.
"""

import os
import re
import urllib.parse

import pandas as pd

from ingredient_insights import ingredient_insights


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "all_varieties_recipe_with_official_source_verification.csv",
)

IMAGES_DIR = os.path.join(BASE_DIR, "images")


RELATED_DISHES_MAP = {
    "dosa": [
        "plain dosa", "masala dosa", "rava dosa",
        "set dosa", "onion dosa", "mysore masala dosa",
    ],
    "biryani": [
        "chicken biryani", "hyderabadi biryani",
        "vegetable biryani", "egg biryani", "mutton biryani",
    ],
    "samosa": [
        "aloo samosa", "vegetable samosa", "paneer samosa",
        "chicken samosa", "corn samosa",
    ],
    "idli": [
        "plain idli", "rava idli", "fried idli",
        "thatte idli", "kanchipuram idli",
    ],
    "paratha": [
        "aloo paratha", "plain aloo paratha",
        "stuffed aloo paratha", "crispy aloo paratha",
        "buttery aloo paratha",
    ],
    "dhokla": [
        "khaman dhokla", "khatta dhokla", "rava dhokla",
        "sandwich dhokla", "white dhokla",
    ],
    "paneer": [
        "kadai paneer", "palak paneer",
        "paneer butter masala", "paneer tikka masala",
    ],
    "kofta": [
        "classic malai kofta", "paneer malai kofta",
        "vegetable malai kofta",
    ],
}


def _clean(value):
    """Convert a value to a clean string."""
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def _clean_key(value):
    """Normalize a string for matching."""
    value = urllib.parse.unquote(_clean(value))
    value = value.lower().replace("_", " ")
    return re.sub(r"\s+", " ", value).strip()


class RecipeEngine:
    """Load, search, and format recipes."""

    def __init__(self, csv_path=DATASET_PATH):
        self.csv_path = csv_path
        self.df = None
        self.load_dataset()

    def load_dataset(self):
        """Load the CSV and create lookup columns in memory."""
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(
                f"Recipe dataset not found at: {self.csv_path}"
            )

        self.df = pd.read_csv(self.csv_path)

        required = [
            "matched_dish",
            "predicted_variety",
            "name",
        ]

        missing = [
            column
            for column in required
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                "Recipe dataset is missing required columns: "
                + ", ".join(missing)
            )

        optional_columns = [
            "description",
            "cuisine",
            "course",
            "diet",
            "prep_time",
            "cooking_time",
            "servings",
            "image_url",
            "ingredients",
            "instructions",
            "source",
            "source_url",
        ]

        for column in optional_columns:
            if column not in self.df.columns:
                self.df[column] = ""

        self.df["matched_dish_clean"] = (
            self.df["matched_dish"].map(_clean_key)
        )
        self.df["predicted_variety_clean"] = (
            self.df["predicted_variety"].map(_clean_key)
        )
        self.df["name_clean"] = self.df["name"].map(_clean_key)
        self.df["cuisine_clean"] = self.df["cuisine"].map(_clean_key)

        print(
            f"[RecipeEngine] Loaded {len(self.df)} recipes "
            f"from {os.path.basename(self.csv_path)}"
        )

    def parse_ingredients(self, raw_ingredients):
        """Parse ingredient lines and attach nutrition and alternatives."""
        if raw_ingredients is None:
            return []

        try:
            if pd.isna(raw_ingredients):
                return []
        except (TypeError, ValueError):
            pass

        raw_text = _clean(raw_ingredients)

        if not raw_text:
            return []

        lines = (
            raw_text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .split("\n")
        )

        output = []

        quantity_re = re.compile(
            r"^(?P<number>"
            r"\d+\s+\d+/\d+|"
            r"\d+/\d+|"
            r"\d+(?:\.\d+)?|"
            r"½|¼|¾"
            r")"
            r"(?:\s*(?P<unit>"
            r"cups?|tablespoons?|tbsp|teaspoons?|tsp|"
            r"sprigs?|grams?|g|kg|ml|"
            r"millilit(?:er|re)s?|lit(?:er|re)s?|"
            r"cloves?|pieces?|pinches?|inches?|"
            r"bunch(?:es)?|leaves|medium|large|small"
            r"))?"
            r"\s+(?P<name>.+)$",
            re.IGNORECASE,
        )

        for line in lines:
            line = re.sub(r"\s+", " ", line)
            line = line.strip().strip(" ,;•-")

            if len(line) < 2:
                continue

            if line.lower().startswith(
                (
                    "ingredients for",
                    "for the ",
                    "to serve",
                    "main ingredients",
                    "method:",
                    "instructions:",
                )
            ):
                continue

            match = quantity_re.match(line)

            if match:
                amount = " ".join(
                    part
                    for part in (
                        match.group("number"),
                        match.group("unit") or "",
                    )
                    if part
                ).strip()

                name = match.group("name").strip(" ,.;")
            else:
                amount = ""
                name = line.strip(" ,.;")

            if not name:
                continue

            try:
                nutrition = ingredient_insights.get_health_score(name)
            except Exception as exc:
                print(
                    f"[RecipeEngine] Nutrition lookup failed for "
                    f"{name!r}: {exc}"
                )
                nutrition = {
                    "available": False,
                    "score": None,
                    "label": "Unavailable",
                    "score_type": "unavailable",
                }

            try:
                alternatives = ingredient_insights.get_alternatives(
                    name,
                    limit=3,
                )
            except Exception as exc:
                print(
                    f"[RecipeEngine] Alternative lookup failed for "
                    f"{name!r}: {exc}"
                )
                alternatives = []

            output.append(
                {
                    "amount": amount,
                    "name": name,
                    "nutrition": nutrition,
                    "alternatives": alternatives,
                }
            )

        return output

    def _format_recipe_dict(self, row):
        """Convert a recipe row to a dictionary for the recipe page."""
        instructions = _clean(row.get("instructions", ""))

        steps = [
            step.strip()
            for step in re.split(r"\n+", instructions)
            if step.strip()
        ]

        dish = _clean(row.get("matched_dish", ""))
        variety = _clean(row.get("predicted_variety", ""))
        name = _clean(row.get("name", ""))

        return {
            "name": name or variety or dish or "Recipe",
            "dish": dish,
            "variety": variety,
            "matched_dish": dish,
            "predicted_variety": variety,
            "description": _clean(row.get("description", "")),
            "cuisine": _clean(row.get("cuisine", "")),
            "course": _clean(row.get("course", "")),
            "diet": _clean(row.get("diet", "")),
            "prep_time": _clean(row.get("prep_time", "")),
            "preparation_time": _clean(row.get("prep_time", "")),
            "cooking_time": _clean(row.get("cooking_time", "")),
            "servings": _clean(row.get("servings", "")),
            "image_url": _clean(row.get("image_url", "")),
            "ingredients": self.parse_ingredients(
                row.get("ingredients", "")
            ),
            "instructions": instructions,
            "steps": steps,
            "source": _clean(row.get("source", "")),
            "source_url": _clean(row.get("source_url", "")),
        }

    def get_recipe(
        self,
        dish=None,
        variety=None,
        top_predictions=None,
        *args,
        **kwargs,
    ):
        """Find a recipe using the dish, variety, or image predictions."""
        if self.df is None or self.df.empty:
            return None

        # First, try the requested variety.
        variety_key = _clean_key(variety)

        if variety_key:
            matches = self.df[
                self.df["predicted_variety_clean"] == variety_key
            ]

            if not matches.empty:
                return self._format_recipe_dict(matches.iloc[0])

            matches = self.df[
                self.df["name_clean"] == variety_key
            ]

            if not matches.empty:
                return self._format_recipe_dict(matches.iloc[0])

        # Next, try the requested dish.
        dish_key = _clean_key(dish)

        if dish_key:
            matches = self.df[
                self.df["matched_dish_clean"] == dish_key
            ]

            if not matches.empty:
                return self._format_recipe_dict(matches.iloc[0])

            matches = self.df[
                (self.df["predicted_variety_clean"] == dish_key)
                | (self.df["name_clean"] == dish_key)
            ]

            if not matches.empty:
                return self._format_recipe_dict(matches.iloc[0])

            # Partial matching as a fallback.
            matches = self.df[
                self.df["name_clean"].str.contains(
                    re.escape(dish_key),
                    na=False,
                    regex=True,
                )
            ]

            if not matches.empty:
                return self._format_recipe_dict(matches.iloc[0])

        # Finally, try image-classification predictions in order.
        if isinstance(top_predictions, list):
            for prediction in top_predictions:
                if isinstance(prediction, dict):
                    prediction_name = (
                        prediction.get("raw_name")
                        or prediction.get("name")
                        or prediction.get("dish")
                        or ""
                    )
                else:
                    prediction_name = str(prediction)

                prediction_key = _clean_key(prediction_name)

                if not prediction_key:
                    continue

                matches = self.df[
                    (self.df["matched_dish_clean"] == prediction_key)
                    | (
                        self.df["predicted_variety_clean"]
                        == prediction_key
                    )
                    | (self.df["name_clean"] == prediction_key)
                ]

                if not matches.empty:
                    return self._format_recipe_dict(matches.iloc[0])

                matches = self.df[
                    self.df["name_clean"].str.contains(
                        re.escape(prediction_key),
                        na=False,
                        regex=True,
                    )
                ]

                if not matches.empty:
                    return self._format_recipe_dict(matches.iloc[0])

        return None

    def search_recipes(self, query, limit=10):
        """Search recipe names, dishes, varieties, and cuisines."""
        if self.df is None or self.df.empty:
            return []

        query_key = _clean_key(query)

        if not query_key:
            return []

        mask = (
            self.df["name_clean"].str.contains(
                re.escape(query_key), na=False, regex=True
            )
            | self.df["matched_dish_clean"].str.contains(
                re.escape(query_key), na=False, regex=True
            )
            | self.df["predicted_variety_clean"].str.contains(
                re.escape(query_key), na=False, regex=True
            )
            | self.df["cuisine_clean"].str.contains(
                re.escape(query_key), na=False, regex=True
            )
        )

        results = self.df.loc[mask].head(max(1, int(limit)))

        return [
            self._format_recipe_dict(row)
            for _, row in results.iterrows()
        ]


# IMPORTANT: This instance must be created outside the class.
recipe_engine = RecipeEngine()
