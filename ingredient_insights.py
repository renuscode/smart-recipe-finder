
import os
import re
import joblib
import pandas as pd


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NUTRITION_FILE = os.path.join(
    BASE_DIR,
    "ingredient_nutrition_verified.csv"
)

SUBSTITUTION_FILE = os.path.join(
    BASE_DIR,
    "ingredient_substitutions.csv"
)

HEALTH_MODEL_FILE = os.path.join(
    BASE_DIR,
    "trained_models",
    "health_score_model.joblib"
)

ALTERNATIVE_MODEL_FILE = os.path.join(
    BASE_DIR,
    "trained_models",
    "alternative_ingredient_model.joblib"
)

# Fallback names only. The saved model's feature list is preferred.
NUTRIENT_FEATURES = (
    "calories",
    "protein_g",
    "fat_g",
    "carbs_g",
    "fiber_g",
    "sugar_g",
    "sodium_mg",
)


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(value):
    """Convert a value into clean text safely."""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_name(value):
    """Normalize ingredient names for exact matching."""

    value = clean_text(value)
    value = value.casefold().replace("_", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip(" ,.;:")


# --------------------------------------------------
# INGREDIENT INSIGHTS
# --------------------------------------------------

class IngredientInsights:

    def __init__(self):
        self.nutrition_df = None
        self.substitution_df = None
        self.health_bundle = None
        self.alternative_bundle = None

        self._load_data()

    # --------------------------------------------------
    # LOAD CSV FILES AND MODELS
    # --------------------------------------------------

    def _load_data(self):

        # Load nutrition data
        if os.path.isfile(NUTRITION_FILE):
            try:
                self.nutrition_df = pd.read_csv(
                    NUTRITION_FILE,
                    encoding="utf-8-sig"
                ).fillna("")

                required_columns = (
                    "ingredient",
                    "normalized_ingredient",
                    "matched_food_name",
                )

                for col in required_columns:
                    if col not in self.nutrition_df.columns:
                        self.nutrition_df[col] = ""

                self.nutrition_df["_ingredient_key"] = (
                    self.nutrition_df["ingredient"].map(normalize_name)
                )

                self.nutrition_df["_normalized_key"] = (
                    self.nutrition_df["normalized_ingredient"].map(
                        normalize_name
                    )
                )

                print(
                    "[IngredientInsights] Nutrition CSV loaded:",
                    len(self.nutrition_df),
                    "rows"
                )

            except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
                print(
                    "[IngredientInsights] Could not load nutrition CSV:",
                    exc
                )

                self.nutrition_df = None

        else:
            print(
                "[IngredientInsights] Nutrition CSV not found:",
                NUTRITION_FILE
            )

        # Load ingredient substitution data
        if os.path.isfile(SUBSTITUTION_FILE):
            try:
                self.substitution_df = pd.read_csv(
                    SUBSTITUTION_FILE,
                    encoding="utf-8-sig"
                ).fillna("")

                required_columns = (
                    "original_ingredient",
                    "normalized_ingredient",
                    "alternative_ingredient",
                )

                for col in required_columns:
                    if col not in self.substitution_df.columns:
                        self.substitution_df[col] = ""

                self.substitution_df["_original_key"] = (
                    self.substitution_df["original_ingredient"].map(
                        normalize_name
                    )
                )

                self.substitution_df["_normalized_key"] = (
                    self.substitution_df["normalized_ingredient"].map(
                        normalize_name
                    )
                )

                print(
                    "[IngredientInsights] Substitution CSV loaded:",
                    len(self.substitution_df),
                    "rows"
                )

            except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
                print(
                    "[IngredientInsights] Could not load substitution CSV:",
                    exc
                )

                self.substitution_df = None

        else:
            print(
                "[IngredientInsights] Substitution CSV not found:",
                SUBSTITUTION_FILE
            )

        # Load health-score model
        self.health_bundle = self._load_model(
            HEALTH_MODEL_FILE,
            "health-score"
        )

        # Load alternative-ingredient model
        self.alternative_bundle = self._load_model(
            ALTERNATIVE_MODEL_FILE,
            "alternative-ingredient"
        )

    @staticmethod
    def _load_model(path, model_name):
        """Load a saved model without crashing application startup."""

        if not os.path.isfile(path):
            print(
                f"[IngredientInsights] {model_name} model not found: {path}"
            )
            return None

        try:
            model = joblib.load(path)

            print(
                f"[IngredientInsights] {model_name} model loaded successfully"
            )

            return model

        except Exception as exc:
            print(
                f"[IngredientInsights] Could not load {model_name} model:",
                exc
            )

            return None

    # --------------------------------------------------
    # COMMON HELPERS
    # --------------------------------------------------

    @staticmethod
    def _unavailable_score():
        return {
            "available": False,
            "score": None,
            "label": "Score unavailable",
            "score_type": "provisional",
        }

    @staticmethod
    def _get_model_and_features(bundle):
        """
        Support either a dictionary containing model/features
        or a model saved directly using joblib.
        """

        if bundle is None:
            return None, []

        if isinstance(bundle, dict):
            model = bundle.get("model")

            features = bundle.get("features")

            if not features and model is not None:
                features = getattr(
                    model,
                    "feature_names_in_",
                    None
                )

            if not features:
                features = NUTRIENT_FEATURES

            return model, list(features)

        model = bundle

        features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if features is None:
            features = NUTRIENT_FEATURES

        return model, list(features)

    # --------------------------------------------------
    # NUTRITION SCORE
    # --------------------------------------------------

    def get_health_score(self, ingredient):
        """
        Calculate a provisional score for an exact ingredient match.

        A score is returned only when:
        - The nutrition CSV is available.
        - The model is loaded.
        - The ingredient name matches.
        - All required model features are valid numeric values.
        """

        if (
            not clean_text(ingredient)
            or self.nutrition_df is None
            or self.health_bundle is None
        ):
            return self._unavailable_score()

        key = normalize_name(ingredient)
        df = self.nutrition_df

        matches = df[
            (df["_normalized_key"] == key)
            | (df["_ingredient_key"] == key)
        ]

        # Do not score a record without a matched food name.
        if "matched_food_name" in matches.columns:
            matches = matches[
                matches["matched_food_name"]
                .astype(str)
                .str.strip()
                .ne("")
            ]

        if matches.empty:
            return self._unavailable_score()

        model, features = self._get_model_and_features(
            self.health_bundle
        )

        if model is None or not features:
            return self._unavailable_score()

        # Find a matching row with every required feature.
        for _, row in matches.iterrows():

            values = {}
            valid = True

            for feature in features:

                try:
                    value = float(row[feature])

                    if not pd.notna(value) or value < 0:
                        valid = False
                        break

                    values[feature] = value

                except (KeyError, TypeError, ValueError):
                    valid = False
                    break

            if not valid:
                continue

            # Preserve the exact feature order used by the model.
            input_df = pd.DataFrame(
                [[values[feature] for feature in features]],
                columns=features
            )

            try:
                prediction = model.predict(input_df)
                score = float(prediction[0])

                if not pd.notna(score):
                    continue

                score = round(
                    max(0.0, min(100.0, score)),
                    1
                )

                return {
                    "available": True,
                    "score": score,
                    "label": "Provisional nutrition score",
                    "score_type": "provisional",
                }

            except Exception as exc:
                print(
                    "[IngredientInsights] Nutrition prediction failed:",
                    exc
                )

                return self._unavailable_score()

        return self._unavailable_score()

    # --------------------------------------------------
    # INGREDIENT ALTERNATIVES
    # --------------------------------------------------

    def get_alternatives(self, ingredient, limit=3):
        """
        Return exact-name alternatives from the source CSV.

        This function does not invent alternatives or use
        the model to generate unverified ingredient names.
        """

        if (
            not clean_text(ingredient)
            or self.substitution_df is None
        ):
            return []

        key = normalize_name(ingredient)
        df = self.substitution_df

        matches = df[
            (df["_original_key"] == key)
            | (df["_normalized_key"] == key)
        ]

        matches = matches[
            matches["alternative_ingredient"]
            .astype(str)
            .str.strip()
            .ne("")
        ]

        try:
            limit = max(1, int(limit))
        except (TypeError, ValueError):
            limit = 3

        results = []
        seen = set()

        for _, row in matches.iterrows():

            alternative = clean_text(
                row.get("alternative_ingredient", "")
            )

            alt_key = normalize_name(alternative)

            if (
                not alt_key
                or alt_key == key
                or alt_key in seen
            ):
                continue

            seen.add(alt_key)

            results.append({
                "ingredient": alternative,

                "substitution_type": clean_text(
                    row.get("substitution_type", "")
                ),

                "quantity_guidance": clean_text(
                    row.get("quantity_guidance", "")
                ),

                "usage_notes": clean_text(
                    row.get("usage_notes", "")
                ),

                "taste_texture_change": clean_text(
                    row.get("taste_texture_change", "")
                ),

                "nutritional_rationale": clean_text(
                    row.get("nutritional_rationale", "")
                ),

                "dietary_notes": clean_text(
                    row.get("dietary_notes", "")
                ),

                "evidence_confidence": clean_text(
                    row.get("evidence_confidence", "")
                ),

                "source_name": clean_text(
                    row.get("source_name", "")
                ),

                "source_url": clean_text(
                    row.get("source_url", "")
                ),
            })

            if len(results) >= limit:
                break

        return results


# --------------------------------------------------
# CREATE SHARED INSTANCE
# --------------------------------------------------

ingredient_insights = IngredientInsights()