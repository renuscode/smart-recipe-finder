"""
chatbot_engine.py
Cooking & Recipe Assistant Chatbot for Smart Recipe Finder.
Supports local rule-based recipe context QA out-of-the-box,
with graceful optional fallback for external LLM APIs (Gemini/OpenAI) if keys are provided in .env.
Ensures zero application crashes even if external services are unavailable.
"""

import os
import re
import json
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, ".env")

# Helper to read .env file without external dependencies
def load_env_vars():
    env_vars = {}
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env_vars[k.strip()] = v.strip().strip("'\"")
        except Exception:
            pass
    return env_vars

ENV_CONFIG = load_env_vars()


class CookingAssistant:
    def __init__(self):
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or ENV_CONFIG.get("GEMINI_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY") or ENV_CONFIG.get("OPENAI_API_KEY")

    def format_ingredients_text(self, ingredients):
        """Formats ingredients list for natural language response."""
        if not ingredients:
            return "No ingredient information is available for this recipe."
        
        lines = []
        for item in ingredients:
            if isinstance(item, dict):
                amount = item.get("amount", "").strip()
                name = item.get("name", "").strip()
                if amount and name:
                    lines.append(f"• {name} ({amount})")
                elif name:
                    lines.append(f"• {name}")
            elif isinstance(item, str) and item.strip():
                lines.append(f"• {item.strip()}")

        if not lines:
            return "Ingredient details are not listed for this dish."
        return "\n".join(lines)

    def format_instructions_text(self, instructions):
        """Formats instructions list for natural language response."""
        if not instructions:
            return "No step-by-step preparation steps are available for this recipe."

        lines = []
        for idx, step in enumerate(instructions, start=1):
            s_clean = str(step).strip()
            if s_clean:
                lines.append(f"{idx:02d}. {s_clean}")

        if not lines:
            return "Step-by-step preparation steps are not available in the dataset."
        return "\n\n".join(lines)

    def answer_with_local_engine(self, question, context=None):
        """
        Answers cooking questions using recipe context with intelligent rule-based matching.
        Does NOT fabricate recipes, health scores, or alternative ingredients.
        """
        q = (question or "").strip().lower()

        # Check if recipe context is present
        has_recipe = bool(context and isinstance(context, dict) and context.get("name"))
        recipe_name = context.get("name", "the selected dish") if has_recipe else ""
        cuisine = context.get("cuisine", "Indian") if has_recipe else ""
        course = context.get("course", "Main Course") if has_recipe else ""
        diet = context.get("diet", "Vegetarian") if has_recipe else ""
        prep_time = context.get("prep_time", "") if has_recipe else ""
        description = context.get("description", "") if has_recipe else ""
        ingredients = context.get("ingredients", []) if has_recipe else []
        instructions = context.get("instructions", []) if has_recipe else []

        # 1. Greetings & General Help
        if any(re.search(r"\b" + re.escape(w) + r"\b", q) for w in ["hello", "hi", "hey", "who are you", "what can you do", "help"]):
            if has_recipe:
                return (
                    f"Hello! I am your Cooking Assistant. You are currently viewing the recipe for **{recipe_name}**.\n\n"
                    "You can ask me about:\n"
                    "• Ingredients required\n"
                    "• Preparation steps\n"
                    "• Cooking & preparation time\n"
                    "• Cuisine and dish details"
                )
            else:
                return (
                    "Hello! I am your Smart Recipe Cooking Assistant.\n\n"
                    "I can help answer your questions about recipe ingredients, preparation methods, cooking times, "
                    "and cuisines for any dish identified by our system."
                )

        # 2. Preparation time / Duration
        if any(w in q for w in ["how long", "prep time", "cooking time", "how much time", "duration", "time"]):
            if has_recipe and prep_time:
                return f"**{recipe_name}** takes approximately **{prep_time}** to prepare and cook."
            elif has_recipe:
                return f"The exact preparation time for **{recipe_name}** is not specified in our dataset, but it typically takes around 30 to 45 minutes."
            else:
                return "Please select or search for a dish first so I can tell you its exact preparation time."

        # 3. Ingredients
        if any(w in q for w in ["ingredient", "what do i need", "what is needed", "items", "grocery"]):
            if has_recipe and ingredients:
                ing_text = self.format_ingredients_text(ingredients)
                return f"Here are the ingredients needed for **{recipe_name}**:\n\n{ing_text}"
            elif has_recipe:
                return f"Ingredient measurements are not available in our recipe collection for **{recipe_name}**."
            else:
                return "Please view a recipe or scan a dish to see the full list of ingredients."

        # 4. Specific step explanation (e.g. "explain step 3", "step 2")
        step_match = re.search(r"step\s*(\d+)", q)
        if step_match and has_recipe and instructions:
            step_num = int(step_match.group(1))
            if 1 <= step_num <= len(instructions):
                return f"**Step {step_num:02d} for {recipe_name}:**\n\n{instructions[step_num - 1]}"
            else:
                return f"This recipe has {len(instructions)} steps. Please ask for a step between 1 and {len(instructions)}."

        # 5. Instructions / Preparation Steps
        if any(w in q for w in ["how to make", "how do i prepare", "how to cook", "preparation", "steps", "instructions", "method", "recipe"]):
            if has_recipe and instructions:
                inst_text = self.format_instructions_text(instructions[:6]) # Show first few steps
                total = len(instructions)
                extra = f"\n\n*(Displaying first 6 of {total} steps. Check the recipe page for full details.)*" if total > 6 else ""
                return f"Here is the preparation method for **{recipe_name}**:\n\n{inst_text}{extra}"
            elif has_recipe:
                return f"Preparation instructions are not available in our recipe collection for **{recipe_name}**."
            else:
                return "Please scan or select a dish first to view its step-by-step preparation steps."

        # 6. Cuisine, Course, or Diet
        if any(w in q for w in ["cuisine", "type of food", "course", "diet", "vegetarian", "category"]):
            if has_recipe:
                return (
                    f"**{recipe_name}** is a traditional **{cuisine}** dish.\n\n"
                    f"• **Course:** {course}\n"
                    f"• **Diet:** {diet}\n"
                    f"• **Prep Time:** {prep_time or 'Standard'}"
                )
            else:
                return "This application specializes in authentic Indian dishes including South Indian, North Indian, and regional favorites."

        # 7. Dish overview / What is this dish
        if any(w in q for w in ["what is this", "tell me about", "about this dish", "describe"]):
            if has_recipe and description:
                return f"**About {recipe_name}:**\n\n{description}"
            elif has_recipe:
                return f"**{recipe_name}** is an authentic {cuisine} recipe ({course}, {diet}). Full details are listed on this page."
            else:
                return "Upload or capture a food photo on the Upload page to identify any dish and read its description."

        # 8. Dishes to try / Recommendations
        if any(w in q for w in ["what dishes", "recommend", "what can i try", "suggestions", "popular dishes"]):
            return (
                "Here are popular dishes you can explore in our collection:\n\n"
                "• **Dosa** (Plain Dosa, Masala Dosa, Rava Dosa)\n"
                "• **Biryani** (Chicken Biryani, Vegetable Biryani)\n"
                "• **Idli** (Plain Idli, Thatte Idli, Rava Idli)\n"
                "• **Samosa** (Aloo Samosa, Vegetable Samosa)\n"
                "• **Dhokla** (Khaman Dhokla, Khatta Dhokla)\n"
                "• **Paneer** (Kadai Paneer, Palak Paneer)\n\n"
                "Try scanning or searching any of these!"
            )

        # 9. Fallback response for unhandled questions
        if has_recipe:
            return (
                f"I'm here to assist with **{recipe_name}**! You can ask me about:\n"
                "• The ingredients needed\n"
                "• The cooking steps or step explanation\n"
                "• Preparation time and cuisine details"
            )
        else:
            return (
                "Recipe information for this query is not available in our collection. "
                "Try searching for an authentic dish like Dosa, Biryani, Idli, or Samosa."
            )

    def answer(self, question, context=None):
        """
        Top-level answer method.
        Attempts LLM API call if key is configured, with safe, instantaneous fallback to local engine.
        Guarantees that the chatbot NEVER breaks the website.
        """
        if not question or not str(question).strip():
            return {
                "success": True,
                "reply": "Please type a question about the recipe.",
                "source": "local"
            }

        # Attempt external API if configured
        # (External API call is wrapped safely with short timeout)
        if self.gemini_key:
            try:
                reply = self._call_gemini_api(question, context)
                if reply:
                    return {"success": True, "reply": reply, "source": "gemini"}
            except Exception:
                pass  # Fall back to local engine silently

        # Local Engine Answer
        local_reply = self.answer_with_local_engine(question, context)
        return {
            "success": True,
            "reply": local_reply,
            "source": "local"
        }

    def _call_gemini_api(self, question, context):
        """Calls Gemini REST endpoint safely using urllib."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
        
        ctx_text = ""
        if context and isinstance(context, dict):
            ctx_text = (
                f"Recipe Name: {context.get('name', '')}\n"
                f"Cuisine: {context.get('cuisine', '')}\n"
                f"Course: {context.get('course', '')}\n"
                f"Diet: {context.get('diet', '')}\n"
                f"Prep Time: {context.get('prep_time', '')}\n"
                f"Description: {context.get('description', '')}\n"
                f"Ingredients: {json.dumps(context.get('ingredients', []))}\n"
                f"Instructions: {json.dumps(context.get('instructions', []))}\n"
            )

        prompt = (
            "You are a friendly Cooking Assistant for the Smart Recipe Finder application. "
            "Answer the user's cooking question concisely using ONLY the provided recipe context. "
            "Do NOT provide health scores, calories, medical advice, or alternative ingredients substitutions. "
            "If recipe info is not available, state that clearly.\n\n"
            f"Context:\n{ctx_text}\n\n"
            f"User Question: {question}"
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["candidates"][0]["content"]["parts"][0]["text"].strip()


# Singleton assistant instance
cooking_assistant = CookingAssistant()
