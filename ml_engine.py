"""
ml_engine.py
Machine Learning & Food Recognition Engine for Smart Recipe Finder.

Uses OpenAI CLIP (ViT-B/32) for food and food-variety recognition.
Images are loaded using PIL before being passed to CLIP.
"""

import os
from io import BytesIO

import torch
import clip
import pandas as pd
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "ViT-B-32.pt"
)

IMAGES_DIR = os.path.join(
    BASE_DIR,
    "images"
)

RECIPES_CSV = os.path.join(
    BASE_DIR,
    "all_varieties_recipe_with_official_source_verification.csv"
)


class MLEngine:

    def __init__(self, model_path=MODEL_PATH):

        self.model_path = model_path

        # Use GPU if available, otherwise CPU
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None
        self.preprocess = None

        # --------------------------------------------------
        # DISH CLASSES
        # --------------------------------------------------

        self.dish_classes = []
        self.dish_prompts = []
        self.dish_text_features = None

        # --------------------------------------------------
        # FOOD VARIETIES
        # --------------------------------------------------

        self.variety_list = []
        self.variety_prompts = []
        self.variety_text_features = None

        # Maps a dish to its possible varieties
        self.dish_to_varieties = {}

        # --------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------

        self.load_model()

        # --------------------------------------------------
        # PREPARE DISH AND VARIETY EMBEDDINGS
        # --------------------------------------------------

        self.setup_classes_and_embeddings()


    # ======================================================
    # LOAD CLIP MODEL
    # ======================================================

    def load_model(self):

        """
        Loads the local CLIP ViT-B/32 model.
        """

        if not os.path.exists(self.model_path):

            raise FileNotFoundError(
                f"Model file not found at {self.model_path}"
            )

        print(
            f"[ML Engine] Loading CLIP model "
            f"from {self.model_path} "
            f"on {self.device}..."
        )

        self.model, self.preprocess = clip.load(
            self.model_path,
            device=self.device,
            jit=False
        )

        self.model.eval()

        print(
            "[ML Engine] Model loaded successfully."
        )


    # ======================================================
    # SETUP DISH CLASSES AND VARIETY EMBEDDINGS
    # ======================================================

    def setup_classes_and_embeddings(self):

        """
        Finds the food classes from the images directory
        and prepares CLIP text embeddings for food classes
        and recipe varieties.
        """

        # --------------------------------------------------
        # 1. DISCOVER DISH CLASSES
        # --------------------------------------------------

        if os.path.exists(IMAGES_DIR):

            self.dish_classes = sorted([
                folder
                for folder in os.listdir(IMAGES_DIR)
                if os.path.isdir(
                    os.path.join(IMAGES_DIR, folder)
                )
            ])

        else:

            self.dish_classes = []


        # --------------------------------------------------
        # FALLBACK TO RECIPE CSV
        # --------------------------------------------------

        if (
            not self.dish_classes
            and os.path.exists(RECIPES_CSV)
        ):

            df = pd.read_csv(RECIPES_CSV)

            self.dish_classes = sorted(
                df["matched_dish"]
                .dropna()
                .unique()
                .tolist()
            )


        # --------------------------------------------------
        # CREATE DISH PROMPTS
        # --------------------------------------------------

        self.dish_prompts = [

            f"a photo of "
            f"{dish.replace('_', ' ')}, "
            f"Indian food"

            for dish in self.dish_classes

        ]


        print(
            f"[ML Engine] Precomputing text features "
            f"for {len(self.dish_classes)} dish classes..."
        )


        # --------------------------------------------------
        # CREATE DISH TEXT EMBEDDINGS
        # --------------------------------------------------

        with torch.no_grad():

            dish_tokens = clip.tokenize(
                self.dish_prompts
            ).to(self.device)

            feats = self.model.encode_text(
                dish_tokens
            )

            self.dish_text_features = (
                feats
                /
                feats.norm(
                    dim=-1,
                    keepdim=True
                )
            )


        # --------------------------------------------------
        # 2. LOAD RECIPE VARIETIES
        # --------------------------------------------------

        if os.path.exists(RECIPES_CSV):

            df = pd.read_csv(
                RECIPES_CSV
            )


            # ------------------------------------------------
            # BUILD DISH → VARIETY MAPPING
            # ------------------------------------------------

            for _, row in df.iterrows():

                dish = str(
                    row["matched_dish"]
                ).strip()

                variety = str(
                    row["predicted_variety"]
                ).strip()


                if dish and variety:

                    dish_clean = (
                        dish
                        .lower()
                        .replace("_", " ")
                    )


                    if (
                        dish_clean
                        not in self.dish_to_varieties
                    ):

                        self.dish_to_varieties[
                            dish_clean
                        ] = []


                    if (
                        variety
                        not in self.dish_to_varieties[
                            dish_clean
                        ]
                    ):

                        self.dish_to_varieties[
                            dish_clean
                        ].append(
                            variety
                        )


            # ------------------------------------------------
            # UNIQUE VARIETIES
            # ------------------------------------------------

            self.variety_list = sorted(

                df["predicted_variety"]
                .dropna()
                .unique()
                .tolist()

            )


            # ------------------------------------------------
            # CREATE VARIETY PROMPTS
            # ------------------------------------------------

            self.variety_prompts = [

                f"a clear photo of "
                f"{variety}, Indian food"

                for variety in self.variety_list

            ]


            print(
                f"[ML Engine] Precomputing text "
                f"features for "
                f"{len(self.variety_list)} "
                f"recipe varieties..."
            )


            # ------------------------------------------------
            # CREATE VARIETY TEXT EMBEDDINGS
            # ------------------------------------------------

            with torch.no_grad():

                variety_tokens = clip.tokenize(
                    self.variety_prompts
                ).to(self.device)


                v_feats = self.model.encode_text(
                    variety_tokens
                )


                self.variety_text_features = (

                    v_feats
                    /
                    v_feats.norm(
                        dim=-1,
                        keepdim=True
                    )

                )


        print(
            "[ML Engine] Text feature cache "
            "successfully initialized."
        )


    # ======================================================
    # IMAGE PREPROCESSING
    # ======================================================

    def preprocess_image_file(
        self,
        image_path_or_bytes
    ):

        """
        Loads an image using PIL.

        Accepts either:

        1. File path
        2. Binary image bytes

        Returns a PyTorch tensor prepared
        using CLIP's preprocessing pipeline.
        """

        try:

            # ------------------------------------------------
            # IMAGE PROVIDED AS BYTES
            # ------------------------------------------------

            if isinstance(
                image_path_or_bytes,
                bytes
            ):

                pil_img = Image.open(
                    BytesIO(
                        image_path_or_bytes
                    )
                ).convert("RGB")


            # ------------------------------------------------
            # IMAGE PROVIDED AS FILE PATH
            # ------------------------------------------------

            else:

                pil_img = Image.open(
                    image_path_or_bytes
                ).convert("RGB")


            # ------------------------------------------------
            # CLIP PREPROCESSING
            # ------------------------------------------------

            image_tensor = (

                self.preprocess(
                    pil_img
                )
                .unsqueeze(0)
                .to(self.device)

            )


            return image_tensor


        except Exception as e:

            raise ValueError(
                f"Could not load image: {str(e)}"
            )


    # ======================================================
    # CLASSIFY IMAGE
    # ======================================================

    def classify_image(
        self,
        image_path_or_bytes,
        top_k=3
    ):

        """
        Performs food recognition and variety prediction.

        Returns:

            best_dish
            predicted_variety
            confidence_percent
            top_dishes
            second_variety
            second_confidence_percent
            third_variety
            third_confidence_percent
        """

        # --------------------------------------------------
        # PREPROCESS IMAGE
        # --------------------------------------------------

        try:

            image_tensor = (
                self.preprocess_image_file(
                    image_path_or_bytes
                )
            )

        except Exception as e:

            return {

                "success": False,

                "error":
                    f"Image processing failed: {str(e)}"

            }


        # --------------------------------------------------
        # CLIP PREDICTION
        # --------------------------------------------------

        with torch.no_grad():

            # ----------------------------------------------
            # ENCODE IMAGE
            # ----------------------------------------------

            image_features = (
                self.model.encode_image(
                    image_tensor
                )
            )


            # Normalize image embedding

            image_features = (

                image_features
                /
                image_features.norm(
                    dim=-1,
                    keepdim=True
                )

            )


            # ----------------------------------------------
            # CLIP LOGIT SCALE
            # ----------------------------------------------

            logit_scale = (
                self.model
                .logit_scale
                .exp()
            )


            # =================================================
            # 1. PREDICT DISH
            # =================================================

            dish_similarity = (

                image_features
                @
                self.dish_text_features.T

            )


            dish_logits = (
                logit_scale
                *
                dish_similarity
            )


            dish_probs = (
                dish_logits
                .softmax(dim=-1)[0]
            )


            # Number of dish predictions

            k_dishes = min(
                top_k,
                len(self.dish_classes)
            )


            top_dish_probs, top_dish_indices = (
                dish_probs.topk(k_dishes)
            )


            # ----------------------------------------------
            # BEST DISH
            # ----------------------------------------------

            best_dish_raw = (
                self.dish_classes[
                    top_dish_indices[0].item()
                ]
            )


            best_dish_clean = (
                best_dish_raw
                .replace("_", " ")
                .title()
            )


            # ----------------------------------------------
            # TOP DISHES
            # ----------------------------------------------

            top_dishes = []


            for i in range(k_dishes):

                raw_name = (
                    self.dish_classes[
                        top_dish_indices[i].item()
                    ]
                )


                d_name = (
                    raw_name
                    .replace("_", " ")
                    .title()
                )


                d_prob = round(
                    top_dish_probs[i].item()
                    * 100,
                    1
                )


                top_dishes.append({

                    "name": d_name,

                    "raw_name": raw_name,

                    "match_percent": d_prob

                })


            # =================================================
            # 2. PREDICT FOOD VARIETY
            # =================================================

            best_dish_key = (

                best_dish_raw
                .lower()
                .replace("_", " ")

            )


            # Get varieties belonging to
            # the predicted dish

            candidate_varieties = (

                self.dish_to_varieties.get(
                    best_dish_key,
                    []
                )

            )


            # =================================================
            # CASE 1:
            # MULTIPLE VARIETIES FOR THIS DISH
            # =================================================

            if (
                candidate_varieties
                and
                len(candidate_varieties) > 1
            ):

                variety_prompts = [

                    f"a clear photo of "
                    f"{v}, Indian food"

                    for v in candidate_varieties

                ]


                v_tokens = clip.tokenize(
                    variety_prompts
                ).to(self.device)


                v_feats = (
                    self.model.encode_text(
                        v_tokens
                    )
                )


                v_feats = (

                    v_feats
                    /
                    v_feats.norm(
                        dim=-1,
                        keepdim=True
                    )

                )


                v_similarity = (

                    image_features
                    @
                    v_feats.T

                )


                v_logits = (
                    logit_scale
                    *
                    v_similarity
                )


                v_probs = (
                    v_logits
                    .softmax(dim=-1)[0]
                )


                k_v = min(
                    3,
                    len(candidate_varieties)
                )


                top_v_probs, top_v_indices = (
                    v_probs.topk(k_v)
                )


                # ------------------------------------------
                # BEST VARIETY
                # ------------------------------------------

                predicted_variety = (

                    candidate_varieties[
                        top_v_indices[0].item()
                    ]

                )


                confidence_percent = round(

                    top_v_probs[0].item()
                    * 100,
                    1

                )


                # ------------------------------------------
                # SECOND VARIETY
                # ------------------------------------------

                second_variety = (

                    candidate_varieties[
                        top_v_indices[1].item()
                    ]

                    if k_v > 1
                    else ""

                )


                second_conf = (

                    round(
                        top_v_probs[1].item()
                        * 100,
                        1
                    )

                    if k_v > 1
                    else 0.0

                )


                # ------------------------------------------
                # THIRD VARIETY
                # ------------------------------------------

                third_variety = (

                    candidate_varieties[
                        top_v_indices[2].item()
                    ]

                    if k_v > 2
                    else ""

                )


                third_conf = (

                    round(
                        top_v_probs[2].item()
                        * 100,
                        1
                    )

                    if k_v > 2
                    else 0.0

                )


            # =================================================
            # CASE 2:
            # ONLY ONE / NO DISH-SPECIFIC VARIETY
            # =================================================

            else:

                # Compare against all varieties

                v_similarity = (

                    image_features
                    @
                    self.variety_text_features.T

                )


                v_logits = (
                    logit_scale
                    *
                    v_similarity
                )


                v_probs = (
                    v_logits
                    .softmax(dim=-1)[0]
                )


                k_v = min(
                    3,
                    len(self.variety_list)
                )


                top_v_probs, top_v_indices = (
                    v_probs.topk(k_v)
                )


                # ------------------------------------------
                # BEST VARIETY
                # ------------------------------------------

                predicted_variety = (

                    self.variety_list[
                        top_v_indices[0].item()
                    ]

                )


                confidence_percent = round(

                    top_v_probs[0].item()
                    * 100,
                    1

                )


                # ------------------------------------------
                # SECOND VARIETY
                # ------------------------------------------

                second_variety = (

                    self.variety_list[
                        top_v_indices[1].item()
                    ]

                    if k_v > 1
                    else ""

                )


                second_conf = (

                    round(
                        top_v_probs[1].item()
                        * 100,
                        1
                    )

                    if k_v > 1
                    else 0.0

                )


                # ------------------------------------------
                # THIRD VARIETY
                # ------------------------------------------

                third_variety = (

                    self.variety_list[
                        top_v_indices[2].item()
                    ]

                    if k_v > 2
                    else ""

                )


                third_conf = (

                    round(
                        top_v_probs[2].item()
                        * 100,
                        1
                    )

                    if k_v > 2
                    else 0.0

                )


            # =================================================
            # FINAL RESULT
            # =================================================

            return {

                "success": True,

                "best_dish":
                    best_dish_clean,

                "best_dish_raw":
                    best_dish_raw,

                "predicted_variety":
                    predicted_variety,

                "confidence_percent":
                    confidence_percent,

                "top_dishes":
                    top_dishes,

                "second_variety":
                    second_variety,

                "second_confidence_percent":
                    second_conf,

                "third_variety":
                    third_variety,

                "third_confidence_percent":
                    third_conf

            }


# ==========================================================
# SINGLETON ML ENGINE
# ==========================================================

ml_engine = MLEngine()