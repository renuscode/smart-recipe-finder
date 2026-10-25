
"""
app.py
Main Flask Application for Smart Recipe Finder.

Features:
- User authentication
- Image upload and camera support
- OpenCLIP food recognition
- Recipe retrieval from the recipe dataset
- MongoDB database storage
- Recipe search
- Ingredient alternatives
- Ingredient health scores
- Hindi-to-English recipe translation
- Optional cooking assistant
- Local development and deployment support
"""

import os
import uuid
import base64
import binascii
import urllib.parse
import json
import re

from functools import wraps

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    send_from_directory,
    flash,
)


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

from models_db import (
    register_user,
    authenticate_user,
    save_image,
    save_dish,
    get_dish_by_name,
    update_dish,
    save_recipe,
)

from recipe_preferences import (
    save_substitution,
    get_substitutions,
)

from ingredient_insights import (
    ingredient_insights,
    normalize_name,
)


# ============================================================
# ML / RECIPE / CHATBOT MODULES
# ============================================================

from ai_client import ml_engine
from recipe_engine import recipe_engine
from chatbot_engine import cooking_assistant


# ============================================================
# FLASK APP CONFIGURATION
# ============================================================

app = Flask(__name__)

IS_PRODUCTION = (
    os.environ.get("PRODUCTION", "").strip().lower()
    in {"1", "true", "yes"}
    or bool(os.environ.get("RENDER"))
)

# Set FLASK_SECRET_KEY in your environment before deployment.
# The fallback is intended only for local development.
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "local-development-key-change-before-deployment",
)

# Maximum request size: 10 MB.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=IS_PRODUCTION,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = os.path.join(BASE_DIR, "images")

UPLOADS_DIR = os.path.join(
    BASE_DIR,
    "static",
    "uploads",
)

os.makedirs(UPLOADS_DIR, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}

MAX_CAMERA_IMAGE_BYTES = 9 * 1024 * 1024


# ============================================================
# HINDI-TO-ENGLISH RECIPE DISPLAY TRANSLATION
# ============================================================

try:
    from deep_translator import MyMemoryTranslator
except ImportError:
    MyMemoryTranslator = None


_TRANSLATION_CACHE = {}

_HINDI_TEXT_PATTERN = re.compile(r"[\u0900-\u097F]")

_TRANSLATION_UNAVAILABLE = (
    "English translation unavailable. Please try again later."
)


def translate_to_english(value):
    """Translate Devanagari text into English and cache successful results."""

    if not isinstance(value, str):
        return value

    text_value = value.strip()

    if not text_value or not _HINDI_TEXT_PATTERN.search(text_value):
        return value

    if text_value in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[text_value]

    if MyMemoryTranslator is None:
        app.logger.warning(
            "deep-translator is not installed. "
            "Install it with: python -m pip install deep-translator"
        )
        return _TRANSLATION_UNAVAILABLE

    try:
        chunks = []
        remaining = text_value
        max_chunk_length = 450

        while remaining:
            if len(remaining) <= max_chunk_length:
                chunks.append(remaining)
                break

            split_at = remaining.rfind(
                " ",
                0,
                max_chunk_length + 1,
            )

            if split_at < max_chunk_length // 2:
                split_at = max_chunk_length

            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip()

        translator = MyMemoryTranslator(
            source="hi-IN",
            target="en-GB",
        )

        translated_chunks = [
            translator.translate(chunk)
            for chunk in chunks
        ]

        if not all(
            isinstance(part, str) and part.strip()
            for part in translated_chunks
        ):
            raise ValueError(
                "Translation provider returned an empty result."
            )

        result = " ".join(
            part.strip()
            for part in translated_chunks
        )

        if _HINDI_TEXT_PATTERN.search(result):
            raise ValueError(
                "Translation result still contains Hindi text."
            )

    except Exception as exc:
        app.logger.warning(
            "Hindi-to-English translation failed: %s",
            exc,
        )
        return _TRANSLATION_UNAVAILABLE

    _TRANSLATION_CACHE[text_value] = result

    return result


def translate_recipe_fields_to_english(value, parent_key=""):
    """Recursively translate recipe text while preserving image and source URLs."""

    if isinstance(value, dict):
        for key, item in list(value.items()):
            if str(key).lower() in {
                "source_url",
                "image",
                "image_url",
                "user_image_url",
                "url",
            }:
                continue

            value[key] = translate_recipe_fields_to_english(
                item,
                str(key),
            )

        return value

    if isinstance(value, list):
        return [
            translate_recipe_fields_to_english(item, parent_key)
            for item in value
        ]

    if isinstance(value, str):
        return translate_to_english(value)

    return value


# ============================================================
# REQUEST ERROR HANDLERS
# ============================================================

@app.errorhandler(413)
def request_too_large(_error):
    """Return a useful error when an upload exceeds the size limit."""

    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "The uploaded image is too large. Maximum size is 10 MB.",
        }), 413

    return (
        "The uploaded file is too large. Maximum size is 10 MB.",
        413,
    )


# ============================================================
# AUTHENTICATION DECORATOR
# ============================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash(
                "Please log in to access this page.",
                "warning",
            )

            return redirect(url_for("login_page"))

        return f(*args, **kwargs)

    return decorated_function


# ============================================================
# HELPER: CONVERT DATA FOR DATABASE
# ============================================================

def convert_to_db_text(value):
    """Convert lists and dictionaries to JSON text for database storage."""

    if value is None:
        return ""

    if isinstance(value, (list, dict)):
        try:
            return json.dumps(
                value,
                ensure_ascii=False,
            )
        except (TypeError, ValueError):
            return str(value)

    return str(value)


# ============================================================
# HELPER: GET PREDICTED DISH
# ============================================================

def get_predicted_dish(result):
    """Extract the predicted dish from the ML result."""

    if not isinstance(result, dict):
        return None

    possible_keys = [
        "best_dish",
        "dish",
        "predicted_dish",
        "matched_dish",
        "class_name",
        "predicted_class",
        "food",
        "food_name",
    ]

    for key in possible_keys:
        value = result.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    predictions = result.get("predictions")

    if isinstance(predictions, list) and predictions:
        first = predictions[0]

        if isinstance(first, dict):
            for key in [
                "best_dish",
                "dish",
                "predicted_dish",
                "matched_dish",
                "class",
                "label",
                "name",
                "food",
            ]:
                value = first.get(key)

                if isinstance(value, str) and value.strip():
                    return value.strip()

    return None


# ============================================================
# HELPER: GET PREDICTED VARIETY
# ============================================================

def get_predicted_variety(result):
    """Extract the predicted recipe variety from the ML result."""

    if not isinstance(result, dict):
        return ""

    for key in [
        "predicted_variety",
        "variety",
        "recipe_variety",
        "matched_variety",
        "best_variety",
    ]:
        value = result.get(key)

        if value:
            return str(value).strip()

    return ""


# ============================================================
# STATIC DATASET IMAGES
# ============================================================

@app.route("/dataset_images/<path:filename>")
def dataset_images(filename):
    decoded_path = urllib.parse.unquote(filename)

    return send_from_directory(
        IMAGES_DIR,
        decoded_path,
    )


# ============================================================
# USER UPLOADED IMAGES
# ============================================================

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    # Only serve files from the application's upload directory.
    # Login is required because uploaded food images may be user-specific.
    return send_from_directory(
        UPLOADS_DIR,
        filename,
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home_page():
    return render_template(
        "index.html",
        user=session.get("user_email"),
    )


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["GET", "POST"])
def login_page():
    if "user_id" in session:
        return redirect(url_for("upload_page"))

    error_message = ""
    username_value = ""
    email_value = ""

    if request.method == "POST":
        username = request.form.get(
            "username",
            "",
        ).strip()

        email = request.form.get(
            "email",
            "",
        ).strip().lower()

        password = request.form.get(
            "password",
            "",
        )

        username_value = username
        email_value = email

        if not username:
            error_message = "Username cannot be empty."

        elif not email:
            error_message = "Email ID cannot be empty."

        elif not password:
            error_message = "Password cannot be empty."

        else:
            try:
                success, message, user_data = authenticate_user(
                    email,
                    username,
                    password,
                )

                if success and user_data:
                    session.clear()

                    session["user_id"] = user_data["id"]
                    session["user_email"] = user_data["email"]
                    session["user_name"] = user_data.get("user_name")

                    return redirect(url_for("upload_page"))

                error_message = message or "Invalid login details."

            except Exception:
                app.logger.exception("Login failed")
                error_message = "Login failed. Please try again."

    return render_template(
        "login.html",
        error=error_message,
        username=username_value,
        email=email_value,
        user=session.get("user_email"),
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if "user_id" in session:
        return redirect(url_for("upload_page"))

    error_message = ""
    username_value = ""
    email_value = ""

    if request.method == "POST":
        username = request.form.get(
            "username",
            "",
        ).strip()

        email = request.form.get(
            "email",
            "",
        ).strip().lower()

        password = request.form.get(
            "password",
            "",
        )

        confirm_password = request.form.get(
            "confirm_password",
            "",
        )

        username_value = username
        email_value = email

        try:
            success, message, user_id = register_user(
                username,
                email,
                password,
                confirm_password,
            )

            if success:
                session.clear()

                session["user_id"] = user_id
                session["user_email"] = email
                session["user_name"] = username

                return redirect(url_for("upload_page"))

            error_message = message or "Registration failed."

        except Exception:
            app.logger.exception("Signup failed")
            error_message = "Registration failed. Please try again."

    return render_template(
        "signup.html",
        error=error_message,
        username=username_value,
        email=email_value,
        user=session.get("user_email"),
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():
    session.clear()

    flash(
        "You have been successfully logged out.",
        "info",
    )

    return redirect(url_for("home_page"))


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route("/upload")
@login_required
def upload_page():
    return render_template(
        "upload.html",
        user=session.get("user_email"),
    )


# ============================================================
# API: CLASSIFY IMAGE
# ============================================================

@app.route("/api/classify", methods=["POST"])
@login_required
def api_classify():
    app.logger.info("Image classification request received")

    saved_filename = None
    saved_filepath = None
    user_image_url = None
    input_type = "upload"

    # --------------------------------------------------------
    # CASE 1: NORMAL FILE UPLOAD
    # --------------------------------------------------------

    if (
        "food_image" in request.files
        and request.files["food_image"].filename
    ):
        file = request.files["food_image"]

        extension = os.path.splitext(file.filename)[1].lower()

        if extension not in ALLOWED_IMAGE_EXTENSIONS:
            return jsonify({
                "success": False,
                "error": (
                    "Please upload a valid JPG, JPEG, PNG, WEBP or BMP image."
                ),
            }), 400

        saved_filename = (
            "upload_"
            + uuid.uuid4().hex
            + extension
        )

        saved_filepath = os.path.join(
            UPLOADS_DIR,
            saved_filename,
        )

        try:
            file.save(saved_filepath)

            if (
                not os.path.isfile(saved_filepath)
                or os.path.getsize(saved_filepath) == 0
            ):
                raise ValueError("The uploaded file is empty.")

        except Exception:
            app.logger.exception("Could not save uploaded image")

            if saved_filepath and os.path.exists(saved_filepath):
                try:
                    os.remove(saved_filepath)
                except OSError:
                    pass

            return jsonify({
                "success": False,
                "error": "Could not save the uploaded image.",
            }), 400

        user_image_url = "/uploads/" + saved_filename
        input_type = "upload"

        app.logger.info("Uploaded image saved: %s", saved_filename)

    # --------------------------------------------------------
    # CASE 2: CAMERA IMAGE
    # --------------------------------------------------------

    elif request.is_json:
        data = request.get_json(silent=True) or {}

        camera_image = data.get("camera_image")

        if not isinstance(camera_image, str) or not camera_image.strip():
            return jsonify({
                "success": False,
                "error": "No camera image received.",
            }), 400

        input_type = "camera"

        try:
            # Accept either a base64 string or a data URL.
            if camera_image.startswith("data:"):
                if "," not in camera_image:
                    raise ValueError("Invalid image data URL.")

                header, camera_image = camera_image.split(",", 1)

                if ";base64" not in header.lower():
                    raise ValueError("Camera image must use base64 encoding.")

            image_bytes = base64.b64decode(
                camera_image,
                validate=True,
            )

            if not image_bytes:
                raise ValueError("Camera image is empty.")

            if len(image_bytes) > MAX_CAMERA_IMAGE_BYTES:
                return jsonify({
                    "success": False,
                    "error": "Camera image is too large.",
                }), 413

            # Verify that the decoded data resembles a supported image.
            # This is a basic signature check, not a full image decoder.
            valid_signature = (
                image_bytes.startswith(b"\xff\xd8\xff")
                or image_bytes.startswith(b"\x89PNG\r\n\x1a\n")
                or image_bytes.startswith(b"RIFF")
                and image_bytes[8:12] == b"WEBP"
                or image_bytes.startswith(b"BM")
            )

            if not valid_signature:
                raise ValueError("Unsupported or invalid image data.")

            saved_filename = (
                "camera_"
                + uuid.uuid4().hex
                + ".jpg"
            )

            # Preserve the actual image format for the file extension.
            if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                saved_filename = (
                    "camera_" + uuid.uuid4().hex + ".png"
                )
            elif (
                image_bytes.startswith(b"RIFF")
                and image_bytes[8:12] == b"WEBP"
            ):
                saved_filename = (
                    "camera_" + uuid.uuid4().hex + ".webp"
                )
            elif image_bytes.startswith(b"BM"):
                saved_filename = (
                    "camera_" + uuid.uuid4().hex + ".bmp"
                )

            saved_filepath = os.path.join(
                UPLOADS_DIR,
                saved_filename,
            )

            with open(saved_filepath, "wb") as image_file:
                image_file.write(image_bytes)

            user_image_url = "/uploads/" + saved_filename

            app.logger.info("Camera image saved: %s", saved_filename)

        except (ValueError, binascii.Error, OSError) as exc:
            app.logger.warning("Invalid camera image: %s", exc)

            if saved_filepath and os.path.exists(saved_filepath):
                try:
                    os.remove(saved_filepath)
                except OSError:
                    pass

            return jsonify({
                "success": False,
                "error": "Invalid camera image. Please capture it again.",
            }), 400

    else:
        return jsonify({
            "success": False,
            "error": "No food image received.",
        }), 400

    # --------------------------------------------------------
    # ML CLASSIFICATION
    # --------------------------------------------------------

    try:
        result = ml_engine.classify_image(saved_filepath)

        app.logger.info("ML classification completed")

    except Exception:
        app.logger.exception("ML classification failed")

        return jsonify({
            "success": False,
            "error": "Image classification failed. Please try again.",
        }), 500

    if not isinstance(result, dict):
        return jsonify({
            "success": False,
            "error": "The ML model returned an invalid result.",
        }), 500

    if result.get("success") is False:
        return jsonify(result), 400

    # --------------------------------------------------------
    # GET DISH AND VARIETY
    # --------------------------------------------------------

    predicted_dish = get_predicted_dish(result)
    predicted_variety = get_predicted_variety(result)

    app.logger.info(
        "Predicted dish=%s, variety=%s",
        predicted_dish,
        predicted_variety,
    )

    if not predicted_dish:
        return jsonify({
            "success": False,
            "error": "Could not identify the food dish.",
            "ml_result": result,
        }), 422

    # --------------------------------------------------------
    # FIND EXISTING DISH OR CREATE ONE
    # --------------------------------------------------------

    try:
        existing_dish = get_dish_by_name(predicted_dish)

        if existing_dish:
            dish_id = existing_dish.get("dish_id")

            # Support documents whose ID is stored as MongoDB _id.
            if dish_id is None:
                dish_id = existing_dish.get("_id")

            app.logger.info(
                "Existing dish found: %s -> %s",
                predicted_dish,
                dish_id,
            )

        else:
            dish_id = save_dish(
                dish_name=predicted_dish,
                description=None,
                cuisine=None,
                dish_type=None,
                course=None,
            )

            app.logger.info(
                "Created dish: %s -> %s",
                predicted_dish,
                dish_id,
            )

    except Exception:
        app.logger.exception("Could not find or create dish")

        return jsonify({
            "success": False,
            "error": "Could not create or find the dish in the database.",
        }), 500

    if not dish_id:
        return jsonify({
            "success": False,
            "error": "Dish ID could not be created.",
        }), 500

    # --------------------------------------------------------
    # GET RECIPE
    # --------------------------------------------------------

    recipe_data = None

    try:
        recipe_data = recipe_engine.get_recipe(
            dish=predicted_dish,
            variety=predicted_variety,
        )

        app.logger.info(
            "Recipe lookup completed; found=%s",
            recipe_data.get("found", False)
            if isinstance(recipe_data, dict)
            else False,
        )

    except Exception:
        app.logger.exception("Recipe engine failed")
        recipe_data = None

    # --------------------------------------------------------
    # UPDATE DISH METADATA
    # --------------------------------------------------------

    if isinstance(recipe_data, dict):
        try:
            description = recipe_data.get("description")
            cuisine = recipe_data.get("cuisine")
            course = recipe_data.get("course")

            if (
                description is None
                or str(description).strip().lower() == "nan"
            ):
                description = ""

            update_dish(
                dish_id=dish_id,
                description=description,
                cuisine=cuisine,
                dish_type=None,
                course=course,
            )

        except Exception:
            app.logger.exception("Could not update dish metadata")

    # --------------------------------------------------------
    # SAVE IMAGE METADATA
    # --------------------------------------------------------

    try:
        image_id = save_image(
            user_id=session["user_id"],
            input_type=input_type,
            image_available=1,
            image_path=saved_filepath,
            dish_id=dish_id,
        )

    except Exception:
        app.logger.exception("Could not save image metadata")

        return jsonify({
            "success": False,
            "error": "Could not save image information to the database.",
        }), 500

    # --------------------------------------------------------
    # SAVE RECIPE
    # --------------------------------------------------------

    recipe_id = None

    if (
        isinstance(recipe_data, dict)
        and recipe_data.get("found")
    ):
        try:
            recipe_name = (
                recipe_data.get("name")
                or predicted_variety
                or predicted_dish
            )

            recipe_id = save_recipe(
                dish_id=dish_id,
                recipe_name=recipe_name,
                ingredients=convert_to_db_text(
                    recipe_data.get("ingredients")
                ),
                instruction=convert_to_db_text(
                    recipe_data.get("instructions")
                ),
                prepare_time=recipe_data.get("prep_time"),
            )

        except Exception:
            # Recipe display can still work even if recipe persistence fails.
            app.logger.exception("Could not save recipe metadata")

    # --------------------------------------------------------
    # STORE SESSION DATA
    # --------------------------------------------------------

    if user_image_url:
        session["last_user_image_url"] = user_image_url

    session["recipe_ready"] = True

    # --------------------------------------------------------
    # BUILD RESPONSE
    # --------------------------------------------------------

    response_data = dict(result)

    response_data.update({
        "success": True,
        "predicted_dish": predicted_dish,
        "predicted_variety": predicted_variety,
        "dish_id": dish_id,
        "image_id": image_id,
        "recipe_id": recipe_id,
        "database_dish_id": dish_id,
        "database_image_id": image_id,
        "database_recipe_id": recipe_id,
        "user_image_url": user_image_url,
    })

    if isinstance(recipe_data, dict):
        response_data["recipe_found"] = recipe_data.get(
            "found",
            False,
        )
        response_data["recipe"] = recipe_data

    app.logger.info(
        "Classification and database processing completed for %s",
        predicted_dish,
    )

    return jsonify(response_data)


# ============================================================
# RECIPE PAGE
# ============================================================

@app.route("/recipe")
@login_required
def recipe_page():
    dish = request.args.get("dish", "").strip()

    variety = request.args.get("variety", "").strip()

    user_image = (
        request.args.get("user_image", "").strip()
        or session.get("last_user_image_url", "")
    )

    top_predictions_raw = request.args.get(
        "top_predictions",
        "",
    )

    top_predictions = []

    if top_predictions_raw:
        try:
            parsed_predictions = json.loads(top_predictions_raw)

            if isinstance(parsed_predictions, list):
                top_predictions = parsed_predictions

        except (ValueError, TypeError):
            top_predictions = []

    # --------------------------------------------------------
    # DEFAULT RECIPE
    # --------------------------------------------------------

    if not dish:
        dish = "Dosa"

        if not variety:
            variety = "plain dosa"

    # --------------------------------------------------------
    # GET RECIPE
    # --------------------------------------------------------

    try:
        recipe_data = recipe_engine.get_recipe(
            dish=dish,
            variety=variety,
            top_predictions=top_predictions,
        )

    except Exception:
        app.logger.exception("Could not retrieve recipe")

        flash(
            "Could not load the recipe. Please try again.",
            "warning",
        )

        return redirect(url_for("upload_page"))

    # --------------------------------------------------------
    # ENRICH INGREDIENTS WITH VERIFIED ALTERNATIVES
    # --------------------------------------------------------

    if isinstance(recipe_data, dict):
        recipe_ingredients = recipe_data.get("ingredients") or []

        if isinstance(recipe_ingredients, list):
            normalized_ingredients = []

            for ingredient_item in recipe_ingredients:
                if isinstance(ingredient_item, str):
                    ingredient_item = {
                        "name": ingredient_item,
                        "amount": "",
                    }

                elif not isinstance(ingredient_item, dict):
                    continue

                ingredient_name = str(
                    ingredient_item.get("name")
                    or ingredient_item.get("ingredient")
                    or ""
                ).strip()

                if not ingredient_name:
                    continue

                ingredient_item["name"] = ingredient_name

                existing_alternatives = (
                    ingredient_item.get("alternatives") or []
                )

                if not isinstance(existing_alternatives, list):
                    existing_alternatives = []

                combined_alternatives = []
                seen_alternatives = set()

                # Keep existing alternatives while removing duplicates.
                for alternative in existing_alternatives:
                    if isinstance(alternative, str):
                        alternative = {"ingredient": alternative}

                    if not isinstance(alternative, dict):
                        continue

                    alternative_name = str(
                        alternative.get("ingredient")
                        or alternative.get("alternative_ingredient")
                        or alternative.get("name")
                        or ""
                    ).strip()

                    alternative_key = normalize_name(alternative_name)

                    if (
                        not alternative_key
                        or alternative_key == normalize_name(ingredient_name)
                        or alternative_key in seen_alternatives
                    ):
                        continue

                    alternative["ingredient"] = alternative_name

                    seen_alternatives.add(alternative_key)
                    combined_alternatives.append(alternative)

                # Load additional alternatives from ingredient_insights.
                try:
                    source_alternatives = (
                        ingredient_insights.get_alternatives(
                            ingredient_name,
                            limit=5,
                        )
                    ) or []

                except Exception:
                    app.logger.exception(
                        "Could not load alternatives for %s",
                        ingredient_name,
                    )
                    source_alternatives = []

                if isinstance(source_alternatives, list):
                    for alternative in source_alternatives:
                        if isinstance(alternative, str):
                            alternative = {
                                "ingredient": alternative,
                            }

                        if not isinstance(alternative, dict):
                            continue

                        alternative_name = str(
                            alternative.get("ingredient")
                            or alternative.get("alternative_ingredient")
                            or alternative.get("name")
                            or ""
                        ).strip()

                        alternative_key = normalize_name(alternative_name)

                        if (
                            alternative_key
                            and alternative_key
                            != normalize_name(ingredient_name)
                            and alternative_key not in seen_alternatives
                        ):
                            alternative["ingredient"] = alternative_name
                            seen_alternatives.add(alternative_key)
                            combined_alternatives.append(alternative)

                ingredient_item["alternatives"] = combined_alternatives

                normalized_ingredients.append(ingredient_item)

            recipe_data["ingredients"] = normalized_ingredients

    # --------------------------------------------------------
    # SAVE RECIPE TO DATABASE
    # --------------------------------------------------------

    if (
        isinstance(recipe_data, dict)
        and recipe_data.get("found")
    ):
        try:
            actual_dish = (
                recipe_data.get("matched_dish")
                or dish
            )

            recipe_name = (
                recipe_data.get("name")
                or variety
                or actual_dish
            )

            description = recipe_data.get("description")
            cuisine = recipe_data.get("cuisine")
            course = recipe_data.get("course")

            dish_id = save_dish(
                dish_name=actual_dish,
                description=description,
                cuisine=cuisine,
                dish_type=None,
                course=course,
            )

            if dish_id:
                update_dish(
                    dish_id=dish_id,
                    description=description,
                    cuisine=cuisine,
                    dish_type=None,
                    course=course,
                )

                recipe_id = save_recipe(
                    dish_id=dish_id,
                    recipe_name=recipe_name,
                    ingredients=convert_to_db_text(
                        recipe_data.get("ingredients")
                    ),
                    instruction=convert_to_db_text(
                        recipe_data.get("instructions")
                    ),
                    prepare_time=recipe_data.get("prep_time"),
                )

                app.logger.info(
                    "Recipe database save completed: %s",
                    recipe_id,
                )

        except Exception:
            app.logger.exception(
                "Could not save recipe from recipe page"
            )

    # --------------------------------------------------------
    # STABLE RECIPE KEY AND USER SUBSTITUTIONS
    # --------------------------------------------------------

    actual_dish_for_key = dish
    recipe_name_for_key = variety or dish

    if isinstance(recipe_data, dict):
        actual_dish_for_key = (
            recipe_data.get("matched_dish")
            or recipe_data.get("dish")
            or dish
        )

        recipe_name_for_key = (
            recipe_data.get("name")
            or variety
            or actual_dish_for_key
        )

    recipe_key = (
        f"{normalize_name(actual_dish_for_key)}::"
        f"{normalize_name(recipe_name_for_key)}"
    )

    try:
        saved_substitutions = get_substitutions(
            session["user_id"],
            recipe_key,
        ) or {}

    except Exception:
        app.logger.exception("Could not load saved substitutions")
        saved_substitutions = {}

    # --------------------------------------------------------
    # TRANSLATE RECIPE TEXT
    # --------------------------------------------------------

    if isinstance(recipe_data, dict):
        for ingredient_item in recipe_data.get("ingredients") or []:
            if not isinstance(ingredient_item, dict):
                continue

            original_name = str(
                ingredient_item.get("name")
                or ingredient_item.get("ingredient")
                or ""
            ).strip()

            if not original_name:
                continue

            translated_name = translate_to_english(original_name)

            old_key = normalize_name(original_name)
            new_key = normalize_name(translated_name)

            if (
                old_key in saved_substitutions
                and new_key != old_key
            ):
                saved_value = saved_substitutions.pop(old_key)

                saved_substitutions[new_key] = translate_to_english(
                    str(saved_value)
                )

        translate_recipe_fields_to_english(recipe_data)

    # --------------------------------------------------------
    # DISPLAY RECIPE
    # --------------------------------------------------------

    return render_template(
        "recipe.html",
        recipe=recipe_data,
        user_image=user_image,
        user=session.get("user_email"),
        top_predictions=top_predictions,
        recipe_key=recipe_key,
        saved_substitutions=saved_substitutions,
        normalize_name=normalize_name,
    )


# ============================================================
# SAVE USER INGREDIENT SUBSTITUTION
# ============================================================

@app.route(
    "/api/recipe/substitutions",
    methods=["POST"],
)
@login_required
def api_save_recipe_substitution():
    data = request.get_json(silent=True) or {}

    recipe_key = str(
        data.get("recipe_key", "")
    ).strip()

    original = str(
        data.get("original_ingredient", "")
    ).strip()

    selected = str(
        data.get("selected_alternative", "")
    ).strip()

    if not recipe_key or not original:
        return jsonify({
            "success": False,
            "error": "Recipe and ingredient are required.",
        }), 400

    if "::" not in recipe_key:
        return jsonify({
            "success": False,
            "error": "Invalid recipe key.",
        }), 400

    try:
        save_substitution(
            user_id=session["user_id"],
            recipe_key=recipe_key,
            original_ingredient=original,
            selected_alternative=selected,
        )

        return jsonify({
            "success": True,
            "message": "Ingredient choice saved.",
        })

    except Exception:
        app.logger.exception("Could not save ingredient substitution")

        return jsonify({
            "success": False,
            "error": "Could not save this choice to the database.",
        }), 500


# ============================================================
# CONSOLIDATED RECIPE HEALTH SCORE
# ============================================================

@app.route(
    "/api/recipe/health-score",
    methods=["POST"],
)
@login_required
def api_recipe_health_score():
    data = request.get_json(silent=True) or {}

    recipe_key = str(
        data.get("recipe_key", "")
    ).strip()

    ingredients = data.get("ingredients", [])

    if (
        not recipe_key
        or not isinstance(ingredients, list)
        or not ingredients
    ):
        return jsonify({
            "success": False,
            "error": "Recipe ingredients are required.",
        }), 400

    if "::" not in recipe_key:
        return jsonify({
            "success": False,
            "error": "Invalid recipe.",
        }), 400

    scores = []
    unavailable_count = 0

    for item in ingredients:
        if not isinstance(item, dict):
            unavailable_count += 1
            continue

        ingredient_name = str(
            item.get("selected_ingredient")
            or item.get("original_ingredient")
            or ""
        ).strip()

        if not ingredient_name:
            unavailable_count += 1
            continue

        try:
            result = ingredient_insights.get_health_score(
                ingredient_name
            )

            if (
                isinstance(result, dict)
                and result.get("available")
                and result.get("score") is not None
            ):
                score = float(result["score"])

                if 0 <= score <= 100:
                    scores.append(score)
                else:
                    unavailable_count += 1

            else:
                unavailable_count += 1

        except (ValueError, TypeError, KeyError):
            app.logger.exception(
                "Invalid ingredient score for %s",
                ingredient_name,
            )
            unavailable_count += 1

        except Exception:
            app.logger.exception(
                "Ingredient score lookup failed for %s",
                ingredient_name,
            )
            unavailable_count += 1

    total_count = len(ingredients)

    if not scores:
        return jsonify({
            "success": True,
            "overall_score": None,
            "scored_count": 0,
            "total_count": total_count,
            "unavailable_count": total_count,
            "label": "Provisional nutrition estimate",
            "message": (
                "A score could not be calculated because verified "
                "nutrition scores were unavailable for these ingredients."
            ),
        })

    return jsonify({
        "success": True,
        "overall_score": round(
            sum(scores) / len(scores),
            1,
        ),
        "scored_count": len(scores),
        "total_count": total_count,
        "unavailable_count": max(
            0,
            total_count - len(scores),
        ),
        "label": "Provisional nutrition estimate",
    })


# ============================================================
# RECIPE JSON API
# ============================================================

@app.route("/api/recipe")
@login_required
def api_recipe():
    dish = request.args.get(
        "dish",
        "",
    ).strip()

    variety = request.args.get(
        "variety",
        "",
    ).strip()

    if not dish and not variety:
        return jsonify({
            "success": False,
            "error": "Please provide a dish or recipe variety.",
        }), 400

    try:
        recipe_data = recipe_engine.get_recipe(
            dish=dish,
            variety=variety,
        )

        return jsonify(recipe_data)

    except Exception:
        app.logger.exception("Recipe JSON API failed")

        return jsonify({
            "success": False,
            "error": "Could not retrieve the recipe.",
        }), 500


# ============================================================
# RECIPE SEARCH API
# ============================================================

@app.route("/api/search_recipes")
@login_required
def api_search_recipes():
    query = request.args.get(
        "q",
        "",
    ).strip()

    if not query:
        return jsonify({
            "results": [],
        })

    try:
        results = recipe_engine.search_recipes(
            query,
            max_results=10,
        )

        return jsonify({
            "results": results,
        })

    except Exception:
        app.logger.exception("Recipe search failed")

        return jsonify({
            "success": False,
            "error": "Recipe search is temporarily unavailable.",
            "results": [],
        }), 500


# ============================================================
# CHATBOT API
# ============================================================

@app.route(
    "/api/chat",
    methods=["POST"],
)
@login_required
def api_chat():
    try:
        data = request.get_json(silent=True) or {}

        message = data.get("message", "")

        if not isinstance(message, str):
            message = ""

        message = message.strip()

        recipe_context = data.get("recipe_context")

        if not message:
            return jsonify({
                "success": True,
                "reply": "Please type a question about this recipe.",
                "source": "local",
            })

        result = cooking_assistant.answer(
            message,
            recipe_context,
        )

        if isinstance(result, dict):
            return jsonify(result)

        return jsonify({
            "success": True,
            "reply": str(result),
            "source": "local",
        })

    except Exception:
        app.logger.exception("Cooking assistant failed")

        return jsonify({
            "success": True,
            "reply": (
                "The cooking assistant is temporarily unavailable. "
                "Please try again later."
            ),
            "source": "error",
        })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    host = (
        "0.0.0.0"
        if IS_PRODUCTION
        else "127.0.0.1"
    )

    print("=" * 60)
    print("SMART RECIPE FINDER SERVER STARTING")
    print(f"Host: {host}")
    print(f"Port: {port}")

    if not IS_PRODUCTION:
        print(f"Open: http://127.0.0.1:{port}")

    print("=" * 60)

    app.run(
        host=host,
        port=port,
        debug=False,
    )
