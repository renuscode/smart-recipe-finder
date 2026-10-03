"""
app.py
Main Flask Application for Smart Recipe Finder.

Features:
- User authentication
- Image upload
- Camera image support
- OpenCLIP food recognition
- Recipe retrieval
- SQLite database storage
- Recipe search
- Optional cooking assistant
"""

import os
import uuid
import base64
import urllib.parse
import json

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


# ============================================================
# ML / RECIPE / CHATBOT MODULES
# ============================================================

from ml_engine import ml_engine
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
# HELPER: CONVERT DATA FOR SQLITE
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

    email_value = ""


    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        email_value = email


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not email:

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
                        password
                    )
                )


                if success:

                    session["user_id"] = (
                        user_data["id"]
                    )

                    session["user_email"] = (
                        user_data["email"]
                    )

                    return redirect(
                        url_for("upload_page")
                    )

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

    email_value = ""


    if request.method == "POST":

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

        email_value = email


        try:

            success, message, user_id = (
                register_user(
                    email,
                    password,
                    confirm_password
                )
            )


            if success:

                session["user_id"] = (
                    user_id
                )

                session["user_email"] = (
                    email.lower()
                )

                return redirect(
                    url_for("upload_page")
                )

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

            if description is None or str(description).strip().lower() == "nan":
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


    user_image = request.args.get(
        "user_image",
        ""
    ).strip()


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
            top_predictions
    )


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

