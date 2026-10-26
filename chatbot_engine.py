
"""
chatbot_engine.py
Cooking & Recipe Assistant for Smart Recipe Finder.

Features:
- Answers questions using the currently displayed recipe.
- Supports ingredients, preparation instructions, individual steps,
  cooking time, cuisine, diet, and recipe descriptions.
- Handles recipe data supplied as strings, dictionaries, or lists.
- Uses Gemini optionally when an API key is configured.
- Falls back to the local rule-based engine if Gemini is unavailable.
- Returns consistent response dictionaries for the Flask API.
"""

import os
import re
import json
import urllib.request
import urllib.error


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")


def load_env_vars():
    """Load .env values without requiring python-dotenv."""
    env_vars = {}

    if not os.path.isfile(ENV_FILE):
        return env_vars

    try:
        with open(ENV_FILE, "r", encoding="utf-8-sig") as file:
            for line in file:
                line = line.strip()

                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                env_vars[key.strip()] = value.strip().strip("'\"")

    except (OSError, UnicodeError):
        pass

    return env_vars


ENV_CONFIG = load_env_vars()


class CookingAssistant:
    def __init__(self):
        self.gemini_key = (
            os.environ.get("GEMINI_API_KEY")
            or ENV_CONFIG.get("GEMINI_API_KEY")
        )

        self.openai_key = (
            os.environ.get("OPENAI_API_KEY")
            or ENV_CONFIG.get("OPENAI_API_KEY")
        )

        # You can override this in .env if necessary.
        self.gemini_model = (
            os.environ.get("GEMINI_MODEL")
            or ENV_CONFIG.get("GEMINI_MODEL")
            or "gemini-2.5-flash"
        )

    # ---------------------------------------------------------
    # DATA NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def _to_text(value):
        """Convert a simple value to clean text safely."""
        if value is None:
            return ""

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, (int, float)):
            return str(value)

        return ""

    @staticmethod
    def _to_list(value):
        """Normalize recipe data into a list without splitting strings."""
        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, dict):
            return [value]

        if isinstance(value, str):
            value = value.strip()

            if not value:
                return []

            # Handle JSON-encoded lists if a caller supplies them.
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass

            # Preserve a single plain-text instruction or ingredient.
            return [value]

        return []

    def format_ingredients_text(self, ingredients):
        """Format recipe ingredients for display."""
        items = self._to_list(ingredients)

        if not items:
            return "No ingredient information is available for this recipe."

        lines = []

        for item in items:
            if isinstance(item, dict):
                name = self._to_text(
                    item.get("name")
                    or item.get("ingredient")
                    or item.get("original_ingredient")
                )

                amount = self._to_text(
                    item.get("amount")
                    or item.get("quantity")
                    or item.get("measure")
                )

                if name and amount:
                    lines.append(f"- {name} ({amount})")
                elif name:
                    lines.append(f"- {name}")
                elif amount:
                    lines.append(f"- {amount}")

            elif isinstance(item, str) and item.strip():
                lines.append(f"- {item.strip()}")

        if not lines:
            return "Ingredient details are not listed for this dish."

        return "\n".join(lines)

    def format_instructions_text(self, instructions):
        """Format recipe instructions into numbered steps."""
        steps = self._to_list(instructions)

        if not steps:
            return "No step-by-step preparation instructions are available."

        lines = []

        for step in steps:
            if isinstance(step, dict):
                step_text = self._to_text(
                    step.get("instruction")
                    or step.get("step")
                    or step.get("description")
                    or step.get("text")
                )
            else:
                step_text = self._to_text(step)

            if step_text:
                lines.append(step_text)

        if not lines:
            return "Step-by-step preparation instructions are not available."

        return "\n\n".join(
            f"{index:02d}. {step}"
            for index, step in enumerate(lines, start=1)
        )

    @staticmethod
    def _contains_any(question, phrases):
        """Match complete words or phrases rather than arbitrary substrings."""
        question = question.lower()

        return any(
            re.search(
                r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)",
                question,
            )
            for phrase in phrases
        )

    # ---------------------------------------------------------
    # LOCAL RULE-BASED CHATBOT
    # ---------------------------------------------------------

    def answer_with_local_engine(self, question, context=None):
        """
        Answer recipe questions from the supplied recipe context.
        This method does not invent missing recipe information.
        """
        q = self._to_text(question).lower()

        if isinstance(context, str):
            try:
                context = json.loads(context)
            except (json.JSONDecodeError, TypeError):
                context = {}

        if not isinstance(context, dict):
            context = {}

        recipe_name = self._to_text(
            context.get("name")
            or context.get("dish")
            or context.get("recipe_name")
        )

        has_recipe = bool(recipe_name)

        cuisine = self._to_text(context.get("cuisine")) or "Not specified"
        course = self._to_text(context.get("course")) or "Not specified"
        diet = self._to_text(context.get("diet")) or "Not specified"
        prep_time = self._to_text(
            context.get("prep_time") or context.get("cooking_time")
        )
        description = self._to_text(context.get("description"))

        ingredients = self._to_list(context.get("ingredients"))
        instructions = self._to_list(
            context.get("instructions")
            or context.get("steps")
            or context.get("method")
        )

        # Normalize instruction dictionaries into readable text.
        clean_steps = []

        for item in instructions:
            if isinstance(item, dict):
                step_text = self._to_text(
                    item.get("instruction")
                    or item.get("step")
                    or item.get("description")
                    or item.get("text")
                )
            else:
                step_text = self._to_text(item)

            if step_text:
                clean_steps.append(step_text)

        # 1. Greetings and help
        if self._contains_any(
            q,
            [
                "hello", "hi", "hey", "good morning",
                "good afternoon", "good evening",
                "who are you", "what can you do", "help",
            ],
        ):
            if has_recipe:
                return (
                    f"Hello! I am your Cooking Assistant. "
                    f"You are viewing **{recipe_name}**.\n\n"
                    "You can ask me about:\n"
                    "- Ingredients and quantities\n"
                    "- How to prepare or cook this dish\n"
                    "- A specific step, such as 'Explain step 2'\n"
                    "- Preparation time\n"
                    "- Cuisine, diet, and dish details"
                )

            return (
                "Hello! I am your Smart Recipe Cooking Assistant.\n\n"
                "Open a recipe first, then ask me about its ingredients, "
                "preparation steps, cooking time, or cuisine."
            )

        # 2. Specific step questions should be checked before general steps.
        step_match = re.search(r"\bstep\s*(\d+)\b", q)

        if step_match:
            if not has_recipe:
                return "Please open a recipe first so I can explain its steps."

            if not clean_steps:
                return (
                    f"Step-by-step instructions for **{recipe_name}** "
                    "are not available in the recipe data."
                )

            step_num = int(step_match.group(1))

            if 1 <= step_num <= len(clean_steps):
                return (
                    f"**Step {step_num} for {recipe_name}:**\n\n"
                    f"{clean_steps[step_num - 1]}"
                )

            return (
                f"This recipe has {len(clean_steps)} available steps. "
                f"Please ask for a step between 1 and {len(clean_steps)}."
            )

        # 3. Ingredients
        if self._contains_any(
            q,
            [
                "ingredient", "ingredients", "what do i need",
                "what is needed", "what are the items",
                "grocery", "shopping list", "quantity", "quantities",
            ],
        ):
            if not has_recipe:
                return "Please open a recipe first to view its ingredients."

            ingredient_text = self.format_ingredients_text(ingredients)

            return (
                f"**Ingredients for {recipe_name}:**\n\n"
                f"{ingredient_text}"
            )

        # 4. Preparation time
        if self._contains_any(
            q,
            [
                "how long", "prep time", "preparation time",
                "cooking time", "how much time", "duration",
                "time required", "time taken",
            ],
        ):
            if not has_recipe:
                return "Please open a recipe first to check its preparation time."

            if prep_time:
                return (
                    f"The recorded preparation time for **{recipe_name}** "
                    f"is **{prep_time}**."
                )

            return (
                f"The preparation time for **{recipe_name}** "
                "is not specified in the available recipe data."
            )

        # 5. Preparation instructions / how to cook
        if self._contains_any(
            q,
            [
                "how to make", "how do i make",
                "how do i prepare", "how can i prepare",
                "how to prepare", "how do i cook",
                "how can i cook", "how to cook",
                "cook this", "make this", "prepare this",
                "preparation", "preparation method",
                "cooking method", "instructions",
                "show steps", "list steps", "recipe method",
                "recipe steps", "method",
            ],
        ) or q.strip() in {
            "steps", "recipe", "how to prepare it", "how to cook it"
        }:
            if not has_recipe:
                return (
                    "Please open or select a recipe first so I can provide "
                    "the preparation instructions for that dish."
                )

            if not clean_steps:
                return (
                    f"Preparation instructions for **{recipe_name}** "
                    "are not available in the recipe data."
                )

            displayed_steps = clean_steps[:10]

            formatted = "\n\n".join(
                f"{index:02d}. {step}"
                for index, step in enumerate(displayed_steps, start=1)
            )

            extra = ""

            if len(clean_steps) > len(displayed_steps):
                extra = (
                    f"\n\nShowing the first {len(displayed_steps)} steps "
                    f"out of {len(clean_steps)}. See the recipe page "
                    "for the remaining steps."
                )

            return (
                f"**How to prepare {recipe_name}:**\n\n"
                f"{formatted}{extra}"
            )

        # 6. Cuisine, course, and diet
        if self._contains_any(
            q,
            [
                "cuisine", "type of food", "course",
                "diet", "vegetarian", "non vegetarian",
                "non-vegetarian", "food category",
            ],
        ):
            if not has_recipe:
                return "Please open a recipe to see its cuisine and category."

            return (
                f"**Details for {recipe_name}**\n\n"
                f"- Cuisine: {cuisine}\n"
                f"- Course: {course}\n"
                f"- Diet: {diet}\n"
                f"- Preparation time: {prep_time or 'Not specified'}"
            )

        # 7. Recipe description
        if self._contains_any(
            q,
            [
                "what is this", "tell me about",
                "about this dish", "describe this",
                "description", "what is",
            ],
        ):
            if not has_recipe:
                return (
                    "Upload a food image or select a dish, then open its "
                    "recipe to see its description."
                )

            if description:
                return f"**About {recipe_name}:**\n\n{description}"

            return (
                f"A description for **{recipe_name}** is not available "
                "in the recipe data."
            )

        # 8. Recommendations from the known application collection
        if self._contains_any(
            q,
            [
                "recommend", "suggest a dish", "suggestions",
                "what can i try", "popular dishes",
                "what dishes", "other dishes",
            ],
        ):
            return (
                "You can explore these dishes in Smart Recipe Finder:\n\n"
                "- Dosa\n"
                "- Biryani\n"
                "- Idli\n"
                "- Samosa\n"
                "- Dhokla\n"
                "- Paneer dishes\n\n"
                "Search for a dish or upload a food image to view its recipe."
            )

        # 9. Fallback
        if has_recipe:
            return (
                f"I'm here to help with **{recipe_name}**. "
                "Try asking one of these questions:\n\n"
                "- What ingredients do I need?\n"
                "- How do I cook this dish?\n"
                "- Explain step 2.\n"
                "- How long does it take?\n"
                "- What cuisine is it?"
            )

        return (
            "I couldn't identify the recipe information for that question. "
            "Please open a recipe first, then ask me about its ingredients, "
            "preparation instructions, cooking time, or cuisine."
        )

    # ---------------------------------------------------------
    # OPTIONAL GEMINI API
    # ---------------------------------------------------------

    def _call_gemini_api(self, question, context):
        """Call Gemini using the configured model and API key."""
        if not self.gemini_key:
            return None

        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent"
            f"?key={self.gemini_key}"
        )

        if not isinstance(context, dict):
            context = {}

        # Limit context size so a large recipe cannot create an excessive prompt.
        ctx_text = (
            f"Recipe name: {self._to_text(context.get('name') or context.get('dish'))}\n"
            f"Cuisine: {self._to_text(context.get('cuisine'))}\n"
            f"Course: {self._to_text(context.get('course'))}\n"
            f"Diet: {self._to_text(context.get('diet'))}\n"
            f"Preparation time: {self._to_text(context.get('prep_time'))}\n"
            f"Description: {self._to_text(context.get('description'))}\n"
            f"Ingredients: {json.dumps(context.get('ingredients', []), ensure_ascii=False, default=str)[:8000]}\n"
            f"Instructions: {json.dumps(context.get('instructions', []), ensure_ascii=False, default=str)[:12000]}\n"
        )

        prompt = (
            "You are the cooking assistant for Smart Recipe Finder.\n"
            "Answer the user's question using the supplied recipe information.\n"
            "Do not claim that missing information is present in the recipe.\n"
            "Do not invent ingredient quantities or recipe instructions.\n"
            "Do not provide medical advice, health scores, or nutrition claims.\n"
            "Do not invent ingredient substitutions; those are handled by a "
            "separate feature in this application.\n"
            "If the question is unrelated to cooking, politely explain your scope.\n"
            "Use a friendly, concise answer.\n\n"
            f"Recipe context:\n{ctx_text}\n"
            f"User question: {question}"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 700,
            },
        }

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=8) as response:
            response_data = json.loads(
                response.read().decode("utf-8")
            )

        candidates = response_data.get("candidates", [])

        if not candidates:
            return None

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        answer_parts = [
            part.get("text", "").strip()
            for part in parts
            if isinstance(part, dict) and part.get("text")
        ]

        answer = "\n".join(answer_parts).strip()

        return answer or None

    # ---------------------------------------------------------
    # PUBLIC ENTRY POINT USED BY FLASK
    # ---------------------------------------------------------

    def answer(self, question, context=None):
        """
        Return a consistent dictionary consumed by the Flask /api/chat route.
        """
        question = self._to_text(question)

        if not question:
            return {
                "success": True,
                "reply": "Please type a question about the recipe.",
                "source": "local",
            }

        # Prefer Gemini when configured. If it fails, use the local engine.
        if self.gemini_key:
            try:
                reply = self._call_gemini_api(question, context)

                if reply:
                    return {
                        "success": True,
                        "reply": reply,
                        "source": "gemini",
                    }

            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                OSError,
                ValueError,
                KeyError,
                TypeError,
            ):
                # A failed API call must not stop the website's chatbot.
                pass

            except Exception:
                # Keep the chatbot available if an unexpected API error occurs.
                pass

        reply = self.answer_with_local_engine(question, context)

        return {
            "success": True,
            "reply": reply,
            "source": "local",
        }


# Shared instance imported by app.py
cooking_assistant = CookingAssistant()
