"""Persist a logged-in user's ingredient alternatives for each recipe in MongoDB."""
from datetime import datetime, timezone

from models_db import db
from ingredient_insights import normalize_name

_preferences = db["user_recipe_substitutions"]
_preferences.create_index(
    [("user_id", 1), ("recipe_key", 1), ("original_ingredient_key", 1)],
    unique=True,
    name="uniq_user_recipe_ingredient_choice",
)


def save_substitution(user_id, recipe_key, original_ingredient, selected_alternative):
    """Save a selection; an empty selection removes the saved alternative."""
    recipe_key = str(recipe_key or "").strip()
    original_ingredient = str(original_ingredient or "").strip()
    selected_alternative = str(selected_alternative or "").strip()
    ingredient_key = normalize_name(original_ingredient)

    if not recipe_key or not original_ingredient or not ingredient_key:
        raise ValueError("Recipe and original ingredient are required.")

    identity = {
        "user_id": user_id,
        "recipe_key": recipe_key,
        "original_ingredient_key": ingredient_key,
    }

    if not selected_alternative:
        _preferences.delete_one(identity)
        return {"saved": True, "removed": True}

    now = datetime.now(timezone.utc)
    _preferences.update_one(
        identity,
        {
            "$set": {
                "original_ingredient": original_ingredient,
                "selected_alternative": selected_alternative,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return {"saved": True, "removed": False}


def get_substitutions(user_id, recipe_key):
    """Return normalized original-ingredient keys mapped to selected alternatives."""
    cursor = _preferences.find(
        {"user_id": user_id, "recipe_key": str(recipe_key or "").strip()},
        {"_id": 0, "original_ingredient_key": 1, "selected_alternative": 1},
    )
    return {
        row["original_ingredient_key"]: row["selected_alternative"]
        for row in cursor
        if row.get("original_ingredient_key") and row.get("selected_alternative")
    }
