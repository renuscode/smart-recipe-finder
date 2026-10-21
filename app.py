"""
app.py
Main Flask Application for Smart Recipe Finder.

Features:
- User authentication
- Image upload
- Camera image support
- OpenCLIP food recognition
- Recipe retrieval
- MongoDB database storage
- Recipe search
- Optional cooking assistant
"""

import os
import uuid
import base64
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
    flash
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
    save_recipe
)

from recipe_preferences import save_substitution, get_substitutions
from ingredient_insights import ingredient_insights, normalize_name


# ============================================================
# ML / RECIPE / CHATBOT MODULES
# ============================================================

from ai_client import ml_engine
from recipe_engine import recipe_engine
from chatbot_engine import cooking_assistant


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "smart-recipe-finder-2026-secret-key"
)


# ============================================================
# OPTIONAL HINDI-TO-ENGLISH RECIPE TRANSLATION
# ============================================================
# Install with: python -m pip install deep-translator
# Translation requires an internet connection. If the package/service is
# unavailable, the application continues running without translation.
try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

_TRANSLATION_CACHE = {}
_HINDI_TEXT_PATTERN = re.compile(r"[\u0900-\u097F]")

def translate_to_english(value):
    """Translate Devanagari text to English while leaving English unchanged."""
    if not isinstance(value, str):
        return value
    text_value = value.strip()
    if not text_value or not _HINDI_TEXT_PATTERN.search(text_value):
        return value
    if text_value in _TRANSLATION_CACHE:
        return _TRANSLATION_CACHE[text_value]
    if GoogleTranslator is None:
        app.logger.warning(
            "Hindi recipe text found, but deep-translator is not installed. "
            "Run: python -m pip install deep-translator"
        )
        return value
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text_value)
        result = translated.strip() if isinstance(translated, str) and translated.strip() else value
    except Exception:
        app.logger.exception("Could not translate recipe text to English")
        result = value
    _TRANSLATION_CACHE[text_value] = result
    return result

def translate_recipe_fields_to_english(value, parent_key=""):
    """Recursively translate recipe display values without changing source URLs."""
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if str(key).lower() in {"source_url", "image", "image_url", "user_image_url", "url"}:
                continue
            value[key] = translate_recipe_fields_to_english(item, str(key))
        return value
    if isinstance(value, list):
        return [translate_recipe_fields_to_english(item, parent_key) for item in value]
    if isinstance(value, str):
        return translate_to_english(value)
    return value


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

IMAGES_DIR = os.path.join(
    BASE_DIR,
    "images"
)

UPLOADS_DIR = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(
    UPLOADS_DIR,
    exist_ok=True
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
                "warning"
            )

            return redirect(
                url_for("login_page")
            )

        return f(
            *args,
            **kwargs
        )

    return decorated_function


# ============================================================
# HELPER: CONVERT DATA FOR DATABASE
# ============================================================

def convert_to_db_text(value):

    if value is None:
        return ""

    if isinstance(
        value,
        (list, dict)
    ):

        try:

            return json.dumps(
                value,
                ensure_ascii=False
            )

        except Exception:

            return str(value)

    return str(value)


# ============================================================
# STATIC DATASET IMAGES
# ============================================================

@app.route(
    "/dataset_images/<path:filename>"
)
def dataset_images(filename):

    decoded_path = urllib.parse.unquote(
        filename
    )

    return send_from_directory(
        IMAGES_DIR,
        decoded_path
    )


# ============================================================
# USER UPLOADED IMAGES
# ============================================================

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOADS_DIR,
        filename
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home_page():

    return render_template(
        "index.html",
        user=session.get(
            "user_email"
        )
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login_page():

    if "user_id" in session:

        return redirect(
            url_for("upload_page")
        )

    error_message = ""

    username_value = ""

    email_value = ""


    if request.method == "POST":

        # ----------------------------------------------------
        # GET FORM VALUES
        # ----------------------------------------------------

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        username_value = username

        email_value = email


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not username:

            error_message = (
                "Username cannot be empty."
            )

        elif not email:

            error_message = (
                "Email ID cannot be empty."
            )

        elif not password:

            error_message = (
                "Password cannot be empty."
            )

        else:

            try:

                success, message, user_data = (
                    authenticate_user(
                        email,
                        username,
                        password
                    )
                )


                # ------------------------------------------------
                # LOGIN SUCCESS
                # ------------------------------------------------

                if success:

                    session["user_id"] = (
                        user_data["id"]
                    )

                    session["user_email"] = (
                        user_data["email"]
                    )

                    session["user_name"] = (
                        user_data.get(
                            "user_name"
                        )
                    )

                    return redirect(
                        url_for("upload_page")
                    )


                # ------------------------------------------------
                # LOGIN FAILED
                # ------------------------------------------------

                else:

                    error_message = message


            except Exception as e:

                print(
                    "LOGIN ERROR:",
                    e
                )

                error_message = (
                    "Login failed. Please try again."
                )


    return render_template(
        "login.html",
        error=error_message,
        username=username_value,
        email=email_value,
        user=session.get(
            "user_email"
        )
    )


# ============================================================
# SIGNUP
# ============================================================

@app.route(
    "/signup",
    methods=["GET", "POST"]
)
def signup_page():

    if "user_id" in session:

        return redirect(
            url_for("upload_page")
        )

    error_message = ""

    username_value = ""

    email_value = ""


    if request.method == "POST":

        # ----------------------------------------------------
        # GET FORM VALUES
        # ----------------------------------------------------

        username = request.form.get(
            "username",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        username_value = username

        email_value = email


        try:

            success, message, user_id = (
                register_user(
                    username,
                    email,
                    password,
                    confirm_password
                )
            )


            # ------------------------------------------------
            # SIGNUP SUCCESS
            # ------------------------------------------------

            if success:

                session["user_id"] = (
                    user_id
                )

                session["user_email"] = (
                    email.lower()
                )

                session["user_name"] = (
                    username.lower()
                )

                return redirect(
                    url_for("upload_page")
                )


            # ------------------------------------------------
            # SIGNUP FAILED
            # ------------------------------------------------

            else:

                error_message = message


        except Exception as e:

            print(
                "SIGNUP ERROR:",
                e
            )

            error_message = (
                "Registration failed. Please try again."
            )


    return render_template(
        "signup.html",
        error=error_message,
        username=username_value,
        email=email_value,
        user=session.get(
            "user_email"
        )
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route(
    "/logout"
)
def logout():

    session.clear()

    flash(
        "You have been successfully logged out.",
        "info"
    )

    return redirect(
        url_for("home_page")
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route(
    "/upload"
)
@login_required
def upload_page():

    return render_template(
        "upload.html",
        user=session.get(
            "user_email"
        )
    )


# ============================================================
# HELPER: GET PREDICTED DISH
# ============================================================

def get_predicted_dish(result):

    """
    Extract dish name from ML result.

    Current ml_engine returns:

        best_dish

    Other possible keys are also supported.
    """

    if not isinstance(
        result,
        dict
    ):

        return None


    # IMPORTANT:
    # best_dish MUST be first.

    possible_keys = [

        "best_dish",

        "dish",

        "predicted_dish",

        "matched_dish",

        "class_name",

        "predicted_class",

        "food",

        "food_name"
    ]


    for key in possible_keys:

        value = result.get(
            key
        )


        if value:

            if isinstance(
                value,
                str
            ):

                value = value.strip()


                if value:

                    return value


    # --------------------------------------------------------
    # NESTED PREDICTIONS
    # --------------------------------------------------------

    predictions = result.get(
        "predictions"
    )


    if isinstance(
        predictions,
        list
    ):

        if len(predictions) > 0:

            first = predictions[0]


            if isinstance(
                first,
                dict
            ):

                for key in [

                    "best_dish",

                    "dish",

                    "predicted_dish",

                    "matched_dish",

                    "class",

                    "label",

                    "name",

                    "food"

                ]:

                    value = first.get(
                        key
                    )


                    if value:

                        return str(
                            value
                        ).strip()


    return None


# ============================================================
# HELPER: GET PREDICTED VARIETY
# ============================================================

def get_predicted_variety(result):

    if not isinstance(
        result,
        dict
    ):

        return ""


    possible_keys = [

        "predicted_variety",

        "variety",

        "recipe_variety",

        "matched_variety",

        "best_variety"
    ]


    for key in possible_keys:

        value = result.get(
            key
        )


        if value:

            return str(
                value
            ).strip()


    return ""


# ============================================================
# API: CLASSIFY IMAGE
# ============================================================

@app.route(
    "/api/classify",
    methods=["POST"]
)
@login_required
def api_classify():

    print()
    print("=" * 60)
    print("[API CLASSIFY] REQUEST RECEIVED")
    print("=" * 60)


    saved_filename = None

    saved_filepath = None

    user_image_url = None

    input_type = "upload"


    # ========================================================
    # CASE 1: NORMAL UPLOAD
    # ========================================================

    if (
        "food_image" in request.files
        and request.files["food_image"].filename
    ):

        file = request.files[
            "food_image"
        ]


        extension = os.path.splitext(
            file.filename
        )[1].lower()


        if extension not in [

            ".jpg",

            ".jpeg",

            ".png",

            ".webp",

            ".bmp"

        ]:

            return jsonify({

                "success": False,

                "error":
                    "Please upload a valid JPG, JPEG, PNG, WEBP or BMP image."

            }), 400


        unique_id = uuid.uuid4().hex[:10]


        saved_filename = (
            "upload_"
            + unique_id
            + extension
        )


        saved_filepath = os.path.join(

            UPLOADS_DIR,

            saved_filename
        )


        file.save(
            saved_filepath
        )


        user_image_url = (
            "/uploads/"
            + saved_filename
        )


        input_type = "upload"


        print(
            "[API CLASSIFY] uploaded image:",
            saved_filepath
        )


    # ========================================================
    # CASE 2: CAMERA IMAGE
    # ========================================================

    elif request.is_json:

        data = request.get_json(
            force=True,
            silent=True
        ) or {}


        camera_image = data.get(
            "camera_image"
        )


        if camera_image:

            input_type = "camera"


            try:

                if "," in camera_image:

                    camera_image = (
                        camera_image
                        .split(
                            ",",
                            1
                        )[1]
                    )


                image_bytes = (
                    base64.b64decode(
                        camera_image
                    )
                )


                unique_id = (
                    uuid.uuid4().hex[:10]
                )


                saved_filename = (
                    "camera_"
                    + unique_id
                    + ".jpg"
                )


                saved_filepath = os.path.join(

                    UPLOADS_DIR,

                    saved_filename
                )


                with open(
                    saved_filepath,
                    "wb"
                ) as f:

                    f.write(
                        image_bytes
                    )


                user_image_url = (
                    "/uploads/"
                    + saved_filename
                )


                print(
                    "[API CLASSIFY] camera image:",
                    saved_filepath
                )


            except Exception as e:

                print(
                    "[API CLASSIFY] camera error:",
                    e
                )


                return jsonify({

                    "success": False,

                    "error":
                        "Invalid camera image."

                }), 400


        else:

            return jsonify({

                "success": False,

                "error":
                    "No food image received."

            }), 400


    else:

        return jsonify({

            "success": False,

            "error":
                "No food image received."

        }), 400


    # ========================================================
    # ML CLASSIFICATION
    # ========================================================

    try:

        print(
            "[API CLASSIFY] ML classification started"
        )


        result = ml_engine.classify_image(
            saved_filepath
        )


        print(
            "[API CLASSIFY] ML RESULT =",
            result
        )


    except Exception as e:

        print(
            "[API CLASSIFY] ML ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                "Image classification failed.",

            "details":
                str(e)

        }), 500


    # ========================================================
    # CHECK ML RESULT
    # ========================================================

    if not isinstance(
        result,
        dict
    ):

        return jsonify({

            "success": False,

            "error":
                "Invalid ML result."

        }), 500


    if result.get(
        "success"
    ) is False:

        return jsonify(
            result
        ), 400


    # ========================================================
    # GET DISH + VARIETY
    # ========================================================

    predicted_dish = (
        get_predicted_dish(
            result
        )
    )


    predicted_variety = (
        get_predicted_variety(
            result
        )
    )


    print(
        "[API CLASSIFY] predicted_dish =",
        predicted_dish
    )


    print(
        "[API CLASSIFY] predicted_variety =",
        predicted_variety
    )


    # ========================================================
    # VERY IMPORTANT:
    # DO NOT SAVE IMAGE WITH NULL DISH ID
    # ========================================================

    if not predicted_dish:

        print(
            "[API CLASSIFY] ERROR: no dish returned by ML"
        )


        return jsonify({

            "success": False,

            "error":
                "Could not identify the food dish.",

            "ml_result":
                result

        }), 422


    # ========================================================
    # FIND EXISTING DISH OR CREATE NEW DISH
    # ========================================================

    dish_id = None


    try:

        existing_dish = (
            get_dish_by_name(
                predicted_dish
            )
        )


        if existing_dish:

            dish_id = (
                existing_dish["dish_id"]
            )


            print(
                "[DB] Existing dish found:",
                predicted_dish,
                "->",
                dish_id
            )


        else:

            dish_id = save_dish(

                dish_name=
                    predicted_dish,

                description=None,

                cuisine=None,

                dish_type=None,

                course=None
            )


            print(
                "[DB] New dish created:",
                predicted_dish,
                "->",
                dish_id
            )


    except Exception as e:

        print(
            "[DB] DISH ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                "Could not create/find dish.",

            "details":
                str(e)

        }), 500


    # ========================================================
    # FINAL DISH ID CHECK
    # ========================================================

    if not dish_id:

        print(
            "[DB] ERROR: dish_id is still None"
        )


        return jsonify({

            "success": False,

            "error":
                "Dish ID could not be created."

        }), 500


    print(
        "[API CLASSIFY] FINAL dish_id =",
        dish_id
    )


    # ========================================================
    # GET RECIPE FROM RECIPE ENGINE
    # ========================================================

    recipe_data = None


    try:

        recipe_data = (
            recipe_engine.get_recipe(

                dish=
                    predicted_dish,

                variety=
                    predicted_variety
            )
        )


        print(
            "[API CLASSIFY] recipe found =",
            recipe_data.get(
                "found"
            )
            if isinstance(
                recipe_data,
                dict
            )
            else False
        )


    except Exception as e:

        print(
            "[RECIPE ENGINE] ERROR:",
            e
        )


        recipe_data = None


    # ========================================================
    # UPDATE DISH METADATA
    # ========================================================

    if isinstance(
        recipe_data,
        dict
    ):

        try:

            description = (
                recipe_data.get(
                    "description"
                )
            )

            if (
                description is None
                or str(description).strip().lower() == "nan"
            ):

                description = ""


            cuisine = (
                recipe_data.get(
                    "cuisine"
                )
            )


            course = (
                recipe_data.get(
                    "course"
                )
            )


            update_dish(

                dish_id=
                    dish_id,

                description=
                    description,

                cuisine=
                    cuisine,

                dish_type=
                    None,

                course=
                    course
            )


            print()
            print("=" * 60)
            print("DISH METADATA UPDATED")
            print("=" * 60)

            print(
                "Dish ID    :",
                dish_id
            )

            print(
                "Description:",
                description
            )

            print(
                "Cuisine    :",
                cuisine
            )

            print(
                "Course     :",
                course
            )

            print("=" * 60)
            print()


        except Exception as e:

            print(
                "WARNING: Could not update dish metadata:",
                e
            )


    # ========================================================
    # SAVE IMAGE
    # ========================================================

    image_id = None


    try:

        image_id = save_image(

            user_id=
                session["user_id"],

            input_type=
                input_type,

            image_available=
                1,

            image_path=
                saved_filepath,

            dish_id=
                dish_id
        )


        print()
        print("=" * 60)
        print("IMAGE DATABASE STORAGE SUCCESS")
        print("=" * 60)

        print(
            "User ID :",
            session["user_id"]
        )

        print(
            "Image ID:",
            image_id
        )

        print(
            "Dish ID :",
            dish_id
        )

        print(
            "Path    :",
            saved_filepath
        )

        print("=" * 60)
        print()


    except Exception as e:

        print(
            "[DB] IMAGE SAVE ERROR:",
            e
        )


        return jsonify({

            "success": False,

            "error":
                "Could not save image information.",

            "details":
                str(e)

        }), 500


    # ========================================================
    # SAVE RECIPE
    # ========================================================

    recipe_id = None


    if isinstance(
        recipe_data,
        dict
    ):

        if recipe_data.get(
            "found"
        ):

            try:

                recipe_name = (

                    recipe_data.get(
                        "name"
                    )

                    or

                    predicted_variety

                    or

                    predicted_dish
                )


                ingredients = (
                    recipe_data.get(
                        "ingredients"
                    )
                )


                instructions = (
                    recipe_data.get(
                        "instructions"
                    )
                )


                prepare_time = (
                    recipe_data.get(
                        "prep_time"
                    )
                )


                recipe_id = save_recipe(

                    dish_id=
                        dish_id,

                    recipe_name=
                        recipe_name,

                    ingredients=
                        convert_to_db_text(
                            ingredients
                        ),

                    instruction=
                        convert_to_db_text(
                            instructions
                        ),

                    prepare_time=
                        prepare_time
                )


                print(
                    "[API CLASSIFY] recipe_id =",
                    recipe_id
                )


            except Exception as e:

                print(
                    "[DB] RECIPE SAVE ERROR:",
                    e
                )


    # Keep the most recently recognized/uploaded food image so the recipe page
    # can display it even if the frontend does not pass user_image in its URL.
    if user_image_url:
        session["last_user_image_url"] = user_image_url

    # ========================================================
    # MARK RECIPE AS READY AFTER SUCCESSFUL CLASSIFICATION
    # ========================================================

    session["recipe_ready"] = True


    # ========================================================
    # BUILD RESPONSE
    # ========================================================

    response_data = dict(
        result
    )


    response_data.update({

        "success":
            True,

        "predicted_dish":
            predicted_dish,

        "predicted_variety":
            predicted_variety,

        "dish_id":
            dish_id,

        "image_id":
            image_id,

        "recipe_id":
            recipe_id,

        "database_dish_id":
            dish_id,

        "database_image_id":
            image_id,

        "database_recipe_id":
            recipe_id,

        "user_image_url":
            user_image_url

    })


    if isinstance(
        recipe_data,
        dict
    ):

        response_data["recipe_found"] = (
            recipe_data.get(
                "found",
                False
            )
        )

        response_data["recipe"] = (
            recipe_data
        )


    # ========================================================
    # FINAL LOG
    # ========================================================

    print()
    print("=" * 60)
    print("DATABASE STORAGE SUCCESS")
    print("=" * 60)

    print(
        "User ID       :",
        session["user_id"]
    )

    print(
        "Image ID      :",
        image_id
    )

    print(
        "Dish ID       :",
        dish_id
    )

    print(
        "Recipe ID     :",
        recipe_id
    )

    print(
        "Dish          :",
        predicted_dish
    )

    print(
        "Variety       :",
        predicted_variety
    )

    print(
        "Image path    :",
        saved_filepath
    )

    print("=" * 60)
    print()


    return jsonify(
        response_data
    )


# ============================================================
# RECIPE PAGE
# ============================================================

@app.route(
    "/recipe"
)
@login_required
def recipe_page():

    dish = request.args.get(
        "dish",
        ""
    ).strip()


    variety = request.args.get(
        "variety",
        ""
    ).strip()


    user_image = (
        request.args.get("user_image", "").strip()
        or session.get("last_user_image_url", "")
    )


    top_predictions_raw = (
        request.args.get(
            "top_predictions",
            ""
        )
    )


    top_predictions = []


    if top_predictions_raw:

        try:

            top_predictions = (
                json.loads(
                    top_predictions_raw
                )
            )

        except Exception:

            top_predictions = []


    # ========================================================
    # DEFAULT RECIPE
    # ========================================================

    if not dish:

        dish = "Dosa"

        if not variety:

            variety = "plain dosa"


    # ========================================================
    # GET RECIPE
    # ========================================================

    recipe_data = (
        recipe_engine.get_recipe(

            dish=
                dish,

            variety=
                variety,

            top_predictions=
                top_predictions
        )
    )

    # Enrich each ingredient with alternatives from the verified substitution
    # data source. This prevents an empty recipe-engine alternatives field from
    # making the alternative choices disappear on the recipe page.
    if isinstance(recipe_data, dict):
        recipe_ingredients = recipe_data.get("ingredients") or []
        normalized_ingredients = []
        for ingredient_item in recipe_ingredients:
            if isinstance(ingredient_item, str):
                ingredient_item = {"name": ingredient_item, "amount": ""}
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

            existing_alternatives = ingredient_item.get("alternatives") or []
            if not isinstance(existing_alternatives, list):
                existing_alternatives = []

            combined_alternatives = []
            seen_alternatives = set()
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
                key = normalize_name(alternative_name)
                if not key or key == normalize_name(ingredient_name) or key in seen_alternatives:
                    continue
                alternative["ingredient"] = alternative_name
                seen_alternatives.add(key)
                combined_alternatives.append(alternative)

            try:
                source_alternatives = ingredient_insights.get_alternatives(
                    ingredient_name, limit=5
                )
            except Exception:
                app.logger.exception(
                    "Could not load alternatives for ingredient %s", ingredient_name
                )
                source_alternatives = []

            for alternative in source_alternatives:
                alternative_name = str(alternative.get("ingredient", "")).strip()
                key = normalize_name(alternative_name)
                if key and key != normalize_name(ingredient_name) and key not in seen_alternatives:
                    seen_alternatives.add(key)
                    combined_alternatives.append(alternative)

            ingredient_item["alternatives"] = combined_alternatives
            normalized_ingredients.append(ingredient_item)

        recipe_data["ingredients"] = normalized_ingredients

    # ========================================================
    # SAVE RECIPE
    # ========================================================

    if isinstance(
        recipe_data,
        dict
    ):

        if recipe_data.get(
            "found"
        ):

            try:

                actual_dish = (

                    recipe_data.get(
                        "matched_dish"
                    )

                    or

                    dish
                )


                recipe_name = (

                    recipe_data.get(
                        "name"
                    )

                    or

                    variety

                    or

                    actual_dish
                )


                ingredients = (
                    recipe_data.get(
                        "ingredients"
                    )
                )


                instructions = (
                    recipe_data.get(
                        "instructions"
                    )
                )


                prepare_time = (
                    recipe_data.get(
                        "prep_time"
                    )
                )


                description = (
                    recipe_data.get(
                        "description"
                    )
                )


                cuisine = (
                    recipe_data.get(
                        "cuisine"
                    )
                )


                course = (
                    recipe_data.get(
                        "course"
                    )
                )


                # --------------------------------------------
                # GET / CREATE DISH
                # --------------------------------------------

                dish_id = save_dish(

                    dish_name=
                        actual_dish,

                    description=
                        description,

                    cuisine=
                        cuisine,

                    dish_type=
                        None,

                    course=
                        course
                )


                # --------------------------------------------
                # UPDATE EXISTING DISH METADATA
                # --------------------------------------------

                update_dish(

                    dish_id=
                        dish_id,

                    description=
                        description,

                    cuisine=
                        cuisine,

                    dish_type=
                        None,

                    course=
                        course
                )


                # --------------------------------------------
                # SAVE RECIPE
                # --------------------------------------------

                recipe_id = save_recipe(

                    dish_id=
                        dish_id,

                    recipe_name=
                        recipe_name,

                    ingredients=
                        convert_to_db_text(
                            ingredients
                        ),

                    instruction=
                        convert_to_db_text(
                            instructions
                        ),

                    prepare_time=
                        prepare_time
                )


                print()
                print("=" * 60)
                print("RECIPE SAVED TO DATABASE")
                print("=" * 60)

                print(
                    "Dish ID    :",
                    dish_id
                )

                print(
                    "Recipe ID  :",
                    recipe_id
                )

                print(
                    "Dish       :",
                    actual_dish
                )

                print(
                    "Recipe     :",
                    recipe_name
                )

                print("=" * 60)
                print()


            except Exception as e:

                print(
                    "WARNING: Could not save recipe:",
                    e
                )


    # ========================================================
    # STABLE RECIPE KEY AND SAVED USER SUBSTITUTIONS
    # ========================================================
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
            session["user_id"], recipe_key
        )
    except Exception:
        app.logger.exception("Could not load saved ingredient substitutions")
        saved_substitutions = {}

    # Translate Hindi recipe fields into English for display. Preserve saved
    # ingredient choices by moving their keys to the translated ingredient names.
    if isinstance(recipe_data, dict):
        for ingredient_item in recipe_data.get("ingredients") or []:
            if not isinstance(ingredient_item, dict):
                continue
            original_name = str(
                ingredient_item.get("name") or ingredient_item.get("ingredient") or ""
            ).strip()
            if not original_name:
                continue
            translated_name = translate_to_english(original_name)
            old_key = normalize_name(original_name)
            new_key = normalize_name(translated_name)
            if old_key in saved_substitutions and new_key != old_key:
                saved_value = saved_substitutions.pop(old_key)
                saved_substitutions[new_key] = translate_to_english(str(saved_value))
        translate_recipe_fields_to_english(recipe_data)

    # ========================================================
    # DISPLAY RECIPE
    # ========================================================

    return render_template(

        "recipe.html",

        recipe=
            recipe_data,

        user_image=
            user_image,

        user=
            session.get(
                "user_email"
            ),

        top_predictions=
            top_predictions,

        recipe_key=recipe_key,

        saved_substitutions=saved_substitutions,

        normalize_name=normalize_name
    )


# ============================================================
# SAVE USER INGREDIENT SUBSTITUTION
# ============================================================

@app.route("/api/recipe/substitutions", methods=["POST"])
@login_required
def api_save_recipe_substitution():
    data = request.get_json(silent=True) or {}
    recipe_key = str(data.get("recipe_key", "")).strip()
    original = str(data.get("original_ingredient", "")).strip()
    selected = str(data.get("selected_alternative", "")).strip()

    if not recipe_key or not original:
        return jsonify({"success": False, "error": "Recipe and ingredient are required."}), 400

    # Ensure the recipe key belongs to the recipe currently being selected.
    # The browser sends the stable key generated by the /recipe route.
    try:
        save_substitution(
            user_id=session["user_id"],
            recipe_key=recipe_key,
            original_ingredient=original,
            selected_alternative=selected,
        )
        return jsonify({"success": True, "message": "Ingredient choice saved."})
    except Exception:
        app.logger.exception("Could not save ingredient substitution")
        return jsonify({"success": False, "error": "Could not save this choice to the database."}), 500


# ============================================================
# CONSOLIDATED RECIPE HEALTH SCORE
# ============================================================

@app.route("/api/recipe/health-score", methods=["POST"])
@login_required
def api_recipe_health_score():
    data = request.get_json(silent=True) or {}
    recipe_key = str(data.get("recipe_key", "")).strip()
    ingredients = data.get("ingredients", [])

    if not recipe_key or not isinstance(ingredients, list) or not ingredients:
        return jsonify({"success": False, "error": "Recipe ingredients are required."}), 400

    # Require the stable key format generated by the recipe page.
    if "::" not in recipe_key:
        return jsonify({"success": False, "error": "Invalid recipe."}), 400

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
            result = ingredient_insights.get_health_score(ingredient_name)
            if isinstance(result, dict) and result.get("available") and result.get("score") is not None:
                score = float(result["score"])
                if 0 <= score <= 100:
                    scores.append(score)
                else:
                    unavailable_count += 1
            else:
                unavailable_count += 1
        except Exception:
            app.logger.exception("Ingredient score lookup failed for %s", ingredient_name)
            unavailable_count += 1

    if not scores:
        return jsonify({
            "success": True,
            "overall_score": None,
            "scored_count": 0,
            "total_count": len(ingredients),
            "unavailable_count": len(ingredients),
            "label": "Provisional nutrition estimate",
        })

    return jsonify({
        "success": True,
        "overall_score": round(sum(scores) / len(scores), 1),
        "scored_count": len(scores),
        "total_count": len(ingredients),
        "unavailable_count": max(0, len(ingredients) - len(scores)),
        "label": "Provisional nutrition estimate",
    })


# ============================================================
# RECIPE JSON API
# ============================================================

@app.route(
    "/api/recipe"
)
@login_required
def api_recipe():

    dish = request.args.get(
        "dish",
        ""
    ).strip()


    variety = request.args.get(
        "variety",
        ""
    ).strip()


    recipe_data = (
        recipe_engine.get_recipe(

            dish=
                dish,

            variety=
                variety
        )
    )


    return jsonify(
        recipe_data
    )


# ============================================================
# RECIPE SEARCH API
# ============================================================

@app.route(
    "/api/search_recipes"
)
@login_required
def api_search_recipes():

    q = request.args.get(
        "q",
        ""
    ).strip()


    results = (
        recipe_engine.search_recipes(

            q,

            max_results=10
        )
    )


    return jsonify({

        "results":
            results

    })


# ============================================================
# CHATBOT API
# ============================================================

@app.route(
    "/api/chat",
    methods=["POST"]
)
@login_required
def api_chat():

    try:

        data = request.get_json(

            force=True,

            silent=True
        ) or {}


        message = (
            data.get(
                "message",
                ""
            )
            .strip()
        )


        recipe_context = (
            data.get(
                "recipe_context",
                None
            )
        )


        if not message:

            return jsonify({

                "success":
                    True,

                "reply":
                    "Please type a question about this recipe.",

                "source":
                    "local"

            })


        result = (
            cooking_assistant.answer(

                message,

                recipe_context
            )
        )


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "CHAT ERROR:",
            e
        )


        return jsonify({

            "success":
                True,

            "reply":
                "The cooking assistant is temporarily unavailable. Please try again later.",

            "source":
                "error"

        })


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "SMART RECIPE FINDER SERVER STARTING"
    )

    print(
        "Serving at: http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False
    )

