"""
models_db.py
Database and User Authentication Module for Smart Recipe Finder.

Uses:
- MongoDB Atlas
- PyMongo
- Werkzeug password hashing

MongoDB collections:
- users
- images
- dishes
- recipes
- alt_ingredients
- counters

Authentication:
Signup:
- Username
- Email
- Password
- Confirm Password

Login:
- Email
- Username
- Password
"""

import os
import re
from datetime import datetime

from dotenv import load_dotenv
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ENV_PATH = os.path.join(
    BASE_DIR,
    ".env"
)

load_dotenv(ENV_PATH)


# ============================================================
# MONGODB CONFIGURATION
# ============================================================

MONGODB_URI = os.getenv(
    "MONGODB_URI"
)

if not MONGODB_URI:
    raise RuntimeError(
        "MONGODB_URI is not set in the .env file."
    )


MONGODB_DATABASE = os.getenv(
    "MONGODB_DATABASE",
    "smart_recipe_finder"
)


# ============================================================
# MONGODB CONNECTION
# ============================================================

client = MongoClient(
    MONGODB_URI,
    serverSelectionTimeoutMS=10000
)

db = client[MONGODB_DATABASE]


# ============================================================
# COLLECTIONS
# ============================================================

users_collection = db["users"]

dishes_collection = db["dishes"]

images_collection = db["images"]

recipes_collection = db["recipes"]

alt_ingredients_collection = db[
    "alt_ingredients"
]

counters_collection = db["counters"]


# ============================================================
# EMAIL VALIDATION
# ============================================================

EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)


# ============================================================
# USERNAME VALIDATION
# ============================================================

USERNAME_REGEX = re.compile(
    r"^[a-zA-Z0-9_]+$"
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():
    """
    Initialize MongoDB collections and indexes.

    MongoDB creates collections automatically when data
    is inserted.

    Indexes are created for:
    - unique email addresses
    - unique usernames
    - dish searching
    - recipe searching
    - image searching
    - alternative ingredients
    """

    # --------------------------------------------------------
    # USERS
    # --------------------------------------------------------

    users_collection.create_index(
        "email",
        unique=True
    )

    users_collection.create_index(
        "user_name_normalized",
        unique=True,
        sparse=True
    )

    # --------------------------------------------------------
    # DISHES
    # --------------------------------------------------------

    dishes_collection.create_index(
        "dish_name_normalized"
    )

    # --------------------------------------------------------
    # RECIPES
    # --------------------------------------------------------

    recipes_collection.create_index(
        [
            ("dish_id", 1),
            ("recipe_name_normalized", 1)
        ]
    )

    # --------------------------------------------------------
    # IMAGES
    # --------------------------------------------------------

    images_collection.create_index(
        "user_id"
    )

    images_collection.create_index(
        "dish_id"
    )

    # --------------------------------------------------------
    # ALTERNATIVE INGREDIENTS
    # --------------------------------------------------------

    alt_ingredients_collection.create_index(
        "recipe_id"
    )

    # --------------------------------------------------------
    # TEST MONGODB CONNECTION
    # --------------------------------------------------------

    client.admin.command("ping")

    print(
        "[MongoDB] Connection successful."
    )


# ============================================================
# ID GENERATOR
# ============================================================

def get_next_id(counter_name):
    """
    Generate a numeric ID similar to SQLite AUTOINCREMENT.

    MongoDB normally uses ObjectId, but this project already
    uses numeric IDs such as:

        user_id
        dish_id
        image_id
        recipe_id
        ingredient_id

    Therefore numeric IDs are preserved.
    """

    counter = counters_collection.find_one_and_update(
        {
            "_id": counter_name
        },
        {
            "$inc": {
                "seq": 1
            }
        },
        upsert=True,
        return_document=ReturnDocument.AFTER
    )

    return counter["seq"]


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_email(email):
    """
    Normalize an email address for storage/search.
    """

    if not email:
        return ""

    return str(email).strip().lower()


def normalize_name(name):
    """
    Normalize a name for case-insensitive searching.
    """

    if not name:
        return ""

    return str(name).strip().lower()


# ============================================================
# EMAIL VALIDATION
# ============================================================

def validate_email_format(email):
    """
    Validate email format.
    """

    if not email or not isinstance(email, str):

        return (
            False,
            "Email ID cannot be empty."
        )

    email = email.strip()

    if len(email) < 5 or "@" not in email:

        return (
            False,
            "Please enter a valid email address."
        )

    if not EMAIL_REGEX.match(email):

        return (
            False,
            "Please enter a valid email address."
        )

    return True, ""


# ============================================================
# USERNAME VALIDATION
# ============================================================

def validate_username_format(username):
    """
    Validate username.

    Username:
    - cannot be empty
    - minimum 3 characters
    - maximum 30 characters
    - can contain letters, numbers and underscore
    """

    if not username or not isinstance(username, str):

        return (
            False,
            "Username cannot be empty."
        )

    username = username.strip()

    if len(username) < 3:

        return (
            False,
            "Username must contain at least 3 characters."
        )

    if len(username) > 30:

        return (
            False,
            "Username must not exceed 30 characters."
        )

    if not USERNAME_REGEX.match(username):

        return (
            False,
            "Username can contain only letters, numbers and underscore."
        )

    return True, ""


# ============================================================
# PASSWORD VALIDATION
# ============================================================

def validate_password_format(password):
    """
    Validate password.
    """

    if (
        not password
        or not isinstance(password, str)
        or not password.strip()
    ):

        return (
            False,
            "Password cannot be empty."
        )

    if len(password) < 8:

        return (
            False,
            "Password must contain at least 8 characters."
        )

    return True, ""


# ============================================================
# CHECK USER EXISTS BY EMAIL
# ============================================================

def user_exists(email):
    """
    Check whether a user already exists using email.
    """

    if not email:
        return False

    clean_email = normalize_email(
        email
    )

    user = users_collection.find_one(
        {
            "email": clean_email
        },
        {
            "_id": 1
        }
    )

    return user is not None


# ============================================================
# CHECK USERNAME EXISTS
# ============================================================

def username_exists(username):
    """
    Check whether a username already exists.
    """

    if not username:
        return False

    normalized_username = normalize_name(
        username
    )

    user = users_collection.find_one(
        {
            "user_name_normalized": normalized_username
        },
        {
            "_id": 1
        }
    )

    # Also check old records that may have user_name
    # but do not yet have user_name_normalized.
    if user is not None:
        return True

    user = users_collection.find_one(
        {
            "user_name": username.strip()
        },
        {
            "_id": 1
        }
    )

    return user is not None


# ============================================================
# REGISTER USER
# ============================================================

def register_user(
    username,
    email,
    password,
    confirm_password
):
    """
    Register a new user.

    Signup fields:
        username
        email
        password
        confirm_password

    Returns:
        (success, message, user_id)
    """

    # --------------------------------------------------------
    # Validate username
    # --------------------------------------------------------

    valid_username, error = (
        validate_username_format(
            username
        )
    )

    if not valid_username:

        return (
            False,
            error,
            None
        )

    clean_username = username.strip()

    normalized_username = normalize_name(
        clean_username
    )

    # --------------------------------------------------------
    # Validate email
    # --------------------------------------------------------

    valid_email, error = (
        validate_email_format(
            email
        )
    )

    if not valid_email:

        return (
            False,
            error,
            None
        )

    clean_email = normalize_email(
        email
    )

    # --------------------------------------------------------
    # Validate password
    # --------------------------------------------------------

    valid_password, error = (
        validate_password_format(
            password
        )
    )

    if not valid_password:

        return (
            False,
            error,
            None
        )

    # --------------------------------------------------------
    # Confirm password
    # --------------------------------------------------------

    if password != confirm_password:

        return (
            False,
            "Passwords do not match.",
            None
        )

    # --------------------------------------------------------
    # Check existing email
    # --------------------------------------------------------

    if user_exists(
        clean_email
    ):

        return (
            False,
            "An account with this email address already exists.",
            None
        )

    # --------------------------------------------------------
    # Check existing username
    # --------------------------------------------------------

    if username_exists(
        clean_username
    ):

        return (
            False,
            "This username is already taken. Please choose another username.",
            None
        )

    # --------------------------------------------------------
    # Hash password
    # --------------------------------------------------------

    password_hash = (
        generate_password_hash(
            password,
            method="pbkdf2:sha256"
        )
    )

    # --------------------------------------------------------
    # Generate user ID
    # --------------------------------------------------------

    new_user_id = get_next_id(
        "users"
    )

    # --------------------------------------------------------
    # Create user document
    # --------------------------------------------------------

    user_document = {
        "_id": new_user_id,
        "id": new_user_id,

        "user_name": clean_username,

        "user_name_normalized":
            normalized_username,

        "email": clean_email,

        "password_hash":
            password_hash,

        "created_at":
            datetime.utcnow()
    }

    # --------------------------------------------------------
    # Insert user
    # --------------------------------------------------------

    try:

        users_collection.insert_one(
            user_document
        )

        return (
            True,
            "Account created successfully.",
            new_user_id
        )

    except DuplicateKeyError as e:

        error_text = str(e)

        if "user_name" in error_text:

            return (
                False,
                "This username is already taken. Please choose another username.",
                None
            )

        return (
            False,
            "An account with this email address already exists.",
            None
        )

    except Exception as e:

        return (
            False,
            f"Registration error: {str(e)}",
            None
        )


# ============================================================
# AUTHENTICATE USER
# ============================================================

def authenticate_user(
    email,
    username,
    password
):
    """
    Authenticate user.

    Login fields:
        email
        username
        password

    Returns:
        (success, message, user_dict)
    """

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not email:

        return (
            False,
            "Email ID cannot be empty.",
            None
        )

    if not username:

        return (
            False,
            "Username cannot be empty.",
            None
        )

    if not password:

        return (
            False,
            "Password cannot be empty.",
            None
        )

    clean_email = normalize_email(
        email
    )

    clean_username = username.strip()

    normalized_username = normalize_name(
        clean_username
    )

    # --------------------------------------------------------
    # Find user using BOTH email and username
    # --------------------------------------------------------

    user = users_collection.find_one(
        {
            "email": clean_email,
            "$or": [
                {
                    "user_name_normalized":
                        normalized_username
                },
                {
                    "user_name":
                        clean_username
                }
            ]
        }
    )

    # --------------------------------------------------------
    # User not found
    # --------------------------------------------------------

    if not user:

        return (
            False,
            "Invalid email, username or password.",
            None
        )

    # --------------------------------------------------------
    # Check password
    # --------------------------------------------------------

    if not check_password_hash(
        user.get("password_hash", ""),
        password
    ):

        return (
            False,
            "Invalid email, username or password.",
            None
        )

    # --------------------------------------------------------
    # Successful login
    # --------------------------------------------------------

    return (
        True,
        "Login successful.",
        {
            "id": user.get(
                "id",
                user.get("_id")
            ),

            "email": user.get(
                "email",
                ""
            ),

            "user_name": user.get(
                "user_name",
                ""
            )
        }
    )


# ============================================================
# SAVE / GET DISH
# ============================================================

def save_dish(
    dish_name,
    description=None,
    cuisine=None,
    dish_type=None,
    course=None
):
    """
    Find an existing dish by name.

    If the dish already exists:
        return the existing dish_id.

    If the dish does not exist:
        create a new dish.

    dish_type is retained for compatibility with app.py
    but is not stored.
    """

    if not dish_name:
        return None

    clean_name = str(
        dish_name
    ).strip()

    if not clean_name:
        return None

    normalized_name = normalize_name(
        clean_name
    )

    # --------------------------------------------------------
    # Check existing dish
    # --------------------------------------------------------

    existing_dish = (
        dishes_collection.find_one(
            {
                "dish_name_normalized":
                    normalized_name
            }
        )
    )

    if existing_dish:

        dish_id = existing_dish.get(
            "dish_id",
            existing_dish.get("_id")
        )

        update_fields = {}

        if description is not None:
            update_fields[
                "description"
            ] = description

        if cuisine is not None:
            update_fields[
                "cuisine"
            ] = cuisine

        if course is not None:
            update_fields[
                "course"
            ] = course

        if update_fields:

            dishes_collection.update_one(
                {
                    "_id":
                        existing_dish["_id"]
                },
                {
                    "$set":
                        update_fields
                }
            )

        return dish_id

    # --------------------------------------------------------
    # Create new dish
    # --------------------------------------------------------

    new_dish_id = get_next_id(
        "dishes"
    )

    dish_document = {
        "_id": new_dish_id,

        "dish_id":
            new_dish_id,

        "dish_name":
            clean_name,

        "dish_name_normalized":
            normalized_name,

        "description":
            description,

        "cuisine":
            cuisine,

        "course":
            course
    }

    dishes_collection.insert_one(
        dish_document
    )

    return new_dish_id


# ============================================================
# FIND DISH BY NAME
# ============================================================

def get_dish_by_name(
    dish_name
):
    """
    Find an existing dish by name.

    Returns:
        dictionary or None
    """

    if not dish_name:
        return None

    clean_name = str(
        dish_name
    ).strip()

    if not clean_name:
        return None

    normalized_name = normalize_name(
        clean_name
    )

    dish = dishes_collection.find_one(
        {
            "dish_name_normalized":
                normalized_name
        }
    )

    return dish


# ============================================================
# UPDATE DISH
# ============================================================

def update_dish(
    dish_id,
    description=None,
    cuisine=None,
    dish_type=None,
    course=None
):
    """
    Update an existing dish.

    dish_type is retained for compatibility
    but is not stored.
    """

    if not dish_id:
        return 0

    update_fields = {
        "description": description,
        "cuisine": cuisine,
        "course": course
    }

    result = dishes_collection.update_one(
        {
            "dish_id": dish_id
        },
        {
            "$set": update_fields
        }
    )

    return result.modified_count


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(
    user_id,
    input_type,
    image_available,
    image_path=None,
    dish_id=None
):
    """
    Store uploaded/captured image information.

    Returns:
        image_id
    """

    image_id = get_next_id(
        "images"
    )

    image_document = {
        "_id": image_id,

        "image_id":
            image_id,

        "user_id":
            user_id,

        "input_type":
            input_type,

        "image_available":
            image_available,

        "image_path":
            image_path,

        "dish_id":
            dish_id,

        "created_at":
            datetime.utcnow()
    }

    images_collection.insert_one(
        image_document
    )

    return image_id


# ============================================================
# UPDATE IMAGE DISH
# ============================================================

def update_image_dish(
    image_id,
    dish_id
):
    """
    Connect an existing image record to a dish.
    """

    if not image_id or not dish_id:
        return 0

    result = images_collection.update_one(
        {
            "image_id":
                image_id
        },
        {
            "$set": {
                "dish_id":
                    dish_id
            }
        }
    )

    return result.modified_count


# ============================================================
# SAVE RECIPE
# ============================================================

def save_recipe(
    dish_id,
    recipe_name,
    ingredients=None,
    instruction=None,
    prepare_time=None
):
    """
    Store a recipe.

    Reuses an existing recipe if one already exists
    for the same dish_id and recipe_name.

    Returns:
        recipe_id
    """

    if not dish_id:

        raise ValueError(
            "Cannot save recipe without a valid dish_id."
        )

    if not recipe_name:

        raise ValueError(
            "Recipe name cannot be empty."
        )

    clean_recipe_name = str(
        recipe_name
    ).strip()

    if not clean_recipe_name:

        raise ValueError(
            "Recipe name cannot be empty."
        )

    normalized_recipe_name = (
        normalize_name(
            clean_recipe_name
        )
    )

    # --------------------------------------------------------
    # Check existing recipe
    # --------------------------------------------------------

    existing_recipe = (
        recipes_collection.find_one(
            {
                "dish_id":
                    dish_id,

                "recipe_name_normalized":
                    normalized_recipe_name
            }
        )
    )

    if existing_recipe:

        return existing_recipe.get(
            "recipe_id",
            existing_recipe.get("_id")
        )

    # --------------------------------------------------------
    # Create new recipe
    # --------------------------------------------------------

    recipe_id = get_next_id(
        "recipes"
    )

    recipe_document = {
        "_id":
            recipe_id,

        "recipe_id":
            recipe_id,

        "dish_id":
            dish_id,

        "recipe_name":
            clean_recipe_name,

        "recipe_name_normalized":
            normalized_recipe_name,

        "ingredients":
            ingredients,

        "instruction":
            instruction,

        "prepare_time":
            prepare_time,

        "created_at":
            datetime.utcnow()
    }

    recipes_collection.insert_one(
        recipe_document
    )

    return recipe_id


# ============================================================
# FIND RECIPE
# ============================================================

def get_recipe(
    recipe_id
):
    """
    Get a recipe by recipe ID.
    """

    if not recipe_id:
        return None

    recipe = recipes_collection.find_one(
        {
            "recipe_id":
                recipe_id
        }
    )

    return recipe


# ============================================================
# GET RECIPES FOR A DISH
# ============================================================

def get_recipes_by_dish(
    dish_id
):
    """
    Get all recipes belonging to a dish.
    """

    if not dish_id:
        return []

    recipes = list(
        recipes_collection.find(
            {
                "dish_id":
                    dish_id
            }
        ).sort(
            "recipe_id",
            1
        )
    )

    return recipes


# ============================================================
# GET ALTERNATIVE INGREDIENTS
# ============================================================

def get_alternative_ingredients(
    recipe_id
):
    """
    Get all alternative ingredients
    belonging to a recipe.
    """

    if not recipe_id:
        return []

    ingredients = list(
        alt_ingredients_collection.find(
            {
                "recipe_id":
                    recipe_id
            }
        ).sort(
            "ingredient_id",
            1
        )
    )

    return ingredients


# ============================================================
# SAVE ALTERNATIVE INGREDIENT
# ============================================================

def save_alternative_ingredient(
    recipe_id,
    alt_ingredient
):
    """
    Store one alternative ingredient.
    """

    if not recipe_id:

        raise ValueError(
            "Cannot save alternative ingredient without recipe_id."
        )

    if not alt_ingredient:
        return None

    ingredient_id = get_next_id(
        "alt_ingredients"
    )

    ingredient_document = {
        "_id":
            ingredient_id,

        "ingredient_id":
            ingredient_id,

        "recipe_id":
            recipe_id,

        "alt_ingredient":
            alt_ingredient
    }

    alt_ingredients_collection.insert_one(
        ingredient_document
    )

    return ingredient_id


# ============================================================
# GET DISH BY ID
# ============================================================

def get_dish(
    dish_id
):
    """
    Get dish information using dish ID.
    """

    if not dish_id:
        return None

    dish = dishes_collection.find_one(
        {
            "dish_id":
                dish_id
        }
    )

    return dish


# ============================================================
# GET IMAGE BY ID
# ============================================================

def get_image(
    image_id
):
    """
    Get image information using image ID.
    """

    if not image_id:
        return None

    image = images_collection.find_one(
        {
            "image_id":
                image_id
        }
    )

    return image


# ============================================================
# INITIALIZE DATABASE
# ============================================================

init_db()