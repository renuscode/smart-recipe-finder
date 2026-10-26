import os
import mimetypes
import requests


class RemoteAIClient:
    def classify_image(self, image_path_or_bytes, top_k=3):
        base_url = os.environ.get("AI_SERVICE_URL", "").rstrip("/")
        api_key = os.environ.get("AI_SERVICE_API_KEY", "").strip()
        if not base_url:
            raise RuntimeError("AI_SERVICE_URL is not configured.")

        if not api_key:
            raise RuntimeError("AI_SERVICE_API_KEY is not configured.")

        if isinstance(image_path_or_bytes, bytes):
            image_bytes = image_path_or_bytes
            filename = "food-image.jpg"
        else:
            filename = os.path.basename(str(image_path_or_bytes))
            with open(image_path_or_bytes, "rb") as image_file:
                image_bytes = image_file.read()

        content_type = mimetypes.guess_type(filename)[0] or "image/jpeg"

        response = requests.post(
            f"{base_url}/classify",
            headers={
    "x-api-key": api_key,
    "ngrok-skip-browser-warning": "true",
},            files={"file": (filename, image_bytes, content_type)},
            timeout=180,
        )

        if not response.ok:
            raise RuntimeError(
                f"AI service returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        return response.json()


if os.environ.get("AI_SERVICE_URL"):
    ml_engine = RemoteAIClient()
else:
    # Local development continues to use the existing CLIP model.
    from ml_engine import ml_engine
