"""
models_db.py
Database and User Authentication Module for Smart Recipe Finder.

Uses:
- SQLite
- Werkzeug password hashing

Database tables:
- users
- images
- dishes
- recipes
- alt_ingredients
"""

import os
import re
import sqlite3
from datetime import datetime

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


# ============================================================
# DATABASE PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

INSTANCE_DIR = os.path.join(
    BASE_DIR,
    "instance"
)

DB_PATH = os.path.join(
    INSTANCE_DIR,
    "recipe_finder.db"
)


# ============================================================
# EMAIL VALIDATION
# ============================================================

EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():
    """
    Create and return a SQLite database connection.

    Foreign keys are enabled for:

        users -> images
        dishes -> images
        dishes -> recipes
        recipes -> alt_ingredients
    """

    os.makedirs(
        INSTANCE_DIR,
        exist_ok=True
    )

    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = sqlite3.Row

    # Enable foreign keys
    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    return conn


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():
    """
    Create required tables if they do not already exist.

    IMPORTANT:
    This function does NOT insert sample/default data.

    The current dishes table DOES NOT contain a 'type' column.
    """

    conn = get_db_connection()
    cursor = conn.cursor()

    # ========================================================
    # USERS TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ========================================================
    # ADD user_name IF IT DOES NOT EXIST
    # ========================================================

    columns = [
        row["name"]
        for row in cursor.execute(
            "PRAGMA table_info(users)"
        ).fetchall()
    ]

    if "user_name" not in columns:

        cursor.execute("""
            ALTER TABLE users
            ADD COLUMN user_name TEXT
        """)

    # ========================================================
    # DISHES TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dishes (
            dish_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dish_name TEXT,
            description TEXT,
            cuisine TEXT,
            course TEXT
        )
    """)

    # ========================================================
    # IMAGES TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            image_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            input_type TEXT,
            image_available INTEGER,
            image_path TEXT,
            dish_id INTEGER,

            FOREIGN KEY (user_id)
                REFERENCES users(id),

            FOREIGN KEY (dish_id)
                REFERENCES dishes(dish_id)
        )
    """)

    # ========================================================
    # RECIPES TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            recipe_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dish_id INTEGER,
            recipe_name TEXT,
            ingredients TEXT,
            instruction TEXT,
            prepare_time TEXT,

            FOREIGN KEY (dish_id)
                REFERENCES dishes(dish_id)
        )
    """)

    # ========================================================
    # ALTERNATIVE INGREDIENTS TABLE
    # ========================================================

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alt_ingredients (
            ingredient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER,
            alt_ingredient TEXT,

            FOREIGN KEY (recipe_id)
                REFERENCES recipes(recipe_id)
        )
    """)

    conn.commit()
    conn.close()


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
# CHECK USER EXISTS
# ============================================================

def user_exists(email):
    """
    Check whether a user already exists.
    """

    if not email:
        return False

    email = email.strip().lower()

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id
        FROM users
        WHERE LOWER(email) = ?
        """,
        (email,)
    )

    user = cursor.fetchone()

    conn.close()

    return user is not None


# ============================================================
# REGISTER USER
# ============================================================

def register_user(
    email,
    password,
    confirm_password
):
    """
    Register a new user.

    Returns:
        (success, message, user_id)
    """

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

    clean_email = (
        email.strip().lower()
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
    # Check existing user
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
    # Hash password
    # --------------------------------------------------------

    password_hash = (
        generate_password_hash(
            password,
            method="pbkdf2:sha256"
        )
    )

    # --------------------------------------------------------
    # Insert user
    # --------------------------------------------------------

    try:

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO users
            (
                email,
                password_hash,
                created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                clean_email,
                password_hash,
                datetime.utcnow()
            )
        )

        conn.commit()

        new_user_id = (
            cursor.lastrowid
        )

        conn.close()

        return (
            True,
            "Account created successfully.",
            new_user_id
        )

    except sqlite3.IntegrityError:

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
    password
):
    """
    Authenticate user.

    Returns:
        (success, message, user_dict)
    """

    if not email or not password:

        return (
            False,
            "Invalid email or password.",
            None
        )

    clean_email = (
        email.strip().lower()
    )

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            email,
            password_hash,
            user_name
        FROM users
        WHERE LOWER(email) = ?
        """,
        (clean_email,)
    )

    user = cursor.fetchone()

    conn.close()

    if not user:

        return (
            False,
            "Invalid email or password.",
            None
        )

    if not check_password_hash(
        user["password_hash"],
        password
    ):

        return (
            False,
            "Invalid email or password.",
            None
        )

    return (
        True,
        "Login successful.",
        {
            "id": user["id"],
            "email": user["email"],
            "user_name": user["user_name"]
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

    IMPORTANT:
    dish_type is retained in the function arguments for compatibility
    with older app.py code, but it is NOT stored because the current
    database no longer has a 'type' column.
    """

    if not dish_name:
        return None

    clean_name = dish_name.strip()

    if not clean_name:
        return None

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        # ----------------------------------------------------
        # Check if dish already exists
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT dish_id
            FROM dishes
            WHERE LOWER(TRIM(dish_name))
                  = LOWER(TRIM(?))
            LIMIT 1
            """,
            (clean_name,)
        )

        existing_dish = cursor.fetchone()

        # ----------------------------------------------------
        # Existing dish
        # ----------------------------------------------------

        if existing_dish:

            dish_id = existing_dish["dish_id"]

            # Only update fields when useful new information
            # is available.
            cursor.execute(
                """
                UPDATE dishes
                SET
                    description = COALESCE(?, description),
                    cuisine = COALESCE(?, cuisine),
                    course = COALESCE(?, course)
                WHERE dish_id = ?
                """,
                (
                    description,
                    cuisine,
                    course,
                    dish_id
                )
            )

            conn.commit()

            return dish_id

        # ----------------------------------------------------
        # New dish
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO dishes
            (
                dish_name,
                description,
                cuisine,
                course
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                clean_name,
                description,
                cuisine,
                course
            )
        )

        conn.commit()

        return cursor.lastrowid

    finally:

        conn.close()


# ============================================================
# FIND DISH BY NAME
# ============================================================

def get_dish_by_name(
    dish_name
):
    """
    Find an existing dish by name.

    Returns:
        sqlite3.Row or None
    """

    if not dish_name:
        return None

    clean_name = dish_name.strip()

    if not clean_name:
        return None

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT *
            FROM dishes
            WHERE LOWER(TRIM(dish_name))
                  = LOWER(TRIM(?))
            LIMIT 1
            """,
            (
                clean_name,
            )
        )

        return cursor.fetchone()

    finally:

        conn.close()


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

    dish_type is retained for compatibility with older code,
    but is not stored because the current database has no
    type column.
    """

    if not dish_id:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            UPDATE dishes
            SET
                description = ?,
                cuisine = ?,
                course = ?
            WHERE dish_id = ?
            """,
            (
                description,
                cuisine,
                course,
                dish_id
            )
        )

        conn.commit()

        return cursor.rowcount

    finally:

        conn.close()


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

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO images
            (
                user_id,
                input_type,
                image_available,
                image_path,
                dish_id
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                input_type,
                image_available,
                image_path,
                dish_id
            )
        )

        conn.commit()

        return cursor.lastrowid

    finally:

        conn.close()


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

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            UPDATE images
            SET dish_id = ?
            WHERE image_id = ?
            """,
            (
                dish_id,
                image_id
            )
        )

        conn.commit()

        return cursor.rowcount

    finally:

        conn.close()


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
    Reuses existing recipe if one already exists for this dish_id and recipe_name
    to avoid duplicate insertions on page refresh.

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

    clean_recipe_name = str(recipe_name).strip()

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        # Check if identical recipe already exists for this dish
        cursor.execute(
            """
            SELECT recipe_id
            FROM recipes
            WHERE dish_id = ?
              AND LOWER(TRIM(recipe_name)) = LOWER(TRIM(?))
            LIMIT 1
            """,
            (dish_id, clean_recipe_name)
        )
        existing_recipe = cursor.fetchone()
        if existing_recipe:
            return existing_recipe["recipe_id"]

        cursor.execute(
            """
            INSERT INTO recipes
            (
                dish_id,
                recipe_name,
                ingredients,
                instruction,
                prepare_time
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                dish_id,
                clean_recipe_name,
                ingredients,
                instruction,
                prepare_time
            )
        )

        conn.commit()

        return cursor.lastrowid

    finally:

        conn.close()


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

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT *
            FROM recipes
            WHERE recipe_id = ?
            """,
            (
                recipe_id,
            )
        )

        return cursor.fetchone()

    finally:

        conn.close()


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

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT *
            FROM recipes
            WHERE dish_id = ?
            ORDER BY recipe_id
            """,
            (
                dish_id,
            )
        )

        return cursor.fetchall()

    finally:

        conn.close()


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

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT *
            FROM alt_ingredients
            WHERE recipe_id = ?
            ORDER BY ingredient_id
            """,
            (
                recipe_id,
            )
        )

        return cursor.fetchall()

    finally:

        conn.close()


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

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO alt_ingredients
            (
                recipe_id,
                alt_ingredient
            )
            VALUES (?, ?)
            """,
            (
                recipe_id,
                alt_ingredient
            )
        )

        conn.commit()

        return cursor.lastrowid

    finally:

        conn.close()


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

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT *
            FROM dishes
            WHERE dish_id = ?
            """,
            (
                dish_id,
            )
        )

        return cursor.fetchone()

    finally:

        conn.close()


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

    conn = get_db_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            SELECT *
            FROM images
            WHERE image_id = ?
            """,
            (
                image_id,
            )
        )

        return cursor.fetchone()

    finally:

        conn.close()


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

init_db()