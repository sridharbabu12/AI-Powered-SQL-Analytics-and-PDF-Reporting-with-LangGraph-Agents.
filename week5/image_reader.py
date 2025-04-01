from PIL import Image
import httpx
from io import BytesIO

async def read_image(image_path: str) -> Image.Image:
    """Read an image from the specified path."""
    return Image.open(image_path)

async def query_openai_api(image: Image.Image, api_key: str) -> dict:
    """Send the image to the OpenAI API and return the response."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_data = buffered.getvalue()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("image.png", image_data, "image/png")},
        )
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()

async def process_image(image_path: str, api_key: str) -> dict:
    """Read the image and query the OpenAI API."""
    image = await read_image(image_path)
    return await query_openai_api(image, api_key)
