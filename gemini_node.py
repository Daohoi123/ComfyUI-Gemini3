import torch
import numpy as np
from PIL import Image
import io
import time
from google import genai
from google.genai import types

class Gemini3ProImageNode:
    """
    ComfyUI Node for Google Gemini 3 Pro (Image Preview).
    Features:
    - Supports 'gemini-3-pro-image-preview' model.
    - Multi-API Key support with Load Balancing.
    - Auto-retry mechanism for 429 (Resource Exhausted) errors (Max 20 retries).
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "api_key_1": ("STRING", {"multiline": False, "default": "", "placeholder": "Primary API Key (Required)"}),
                "prompt": ("STRING", {"multiline": True, "default": "Make this image cyberpunk style", "placeholder": "Enter your prompt here..."}),
                "model_name": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview"}),
            },
            "optional": {
                "api_key_2": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 1 (Optional)"}),
                "api_key_3": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 2 (Optional)"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "text_response")
    FUNCTION = "process_image"
    CATEGORY = "Gemini AI"

    def process_image(self, image, api_key_1, prompt, model_name, api_key_2="", api_key_3=""):
        # 1. Collect valid API keys
        keys = [k.strip() for k in [api_key_1, api_key_2, api_key_3] if k and k.strip()]
        
        if not keys:
            raise ValueError("No API Key provided! Please enter at least one API Key.")

        # 2. Prepare input image
        img_tensor = image[0] 
        i = 255. * img_tensor.cpu().numpy()
        pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Internal function to call API
        def call_api(current_key):
            client = genai.Client(api_key=current_key)
            
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(image_size="1K")
            )

            print(f"Gemini Node: Sending request to {model_name} using Key ending in ...{current_key[-4:]}")

            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                        ],
                    ),
                ],
                config=generate_content_config,
            )

            out_tensor = None
            full_text = ""
            image_found = False
            
            for chunk in response_stream:
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue

                for part in chunk.candidates[0].content.parts:
                    # Handle Image Data
                    if part.inline_data and part.inline_data.data:
                        print("Gemini Node: Received image data.")
                        image_data = part.inline_data.data
                        out_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
                        out_np = np.array(out_pil).astype(np.float32) / 255.0
                        out_tensor = torch.from_numpy(out_np)[None,]
                        image_found = True
                    
                    # Handle Text Data
                    if part.text:
                        full_text += part.text

            if not image_found:
                raise ValueError(f"Model returned text only (No Image): {full_text}")

            return out_tensor, full_text

        # 3. Retry Logic with Load Balancing (Max 10 retries)
        max_retries = 10 
        retry_count = 0

        while retry_count < max_retries:
            for index, key in enumerate(keys):
                try:
                    return call_api(key)
                except Exception as e:
                    error_msg = str(e)
                    # Check for 429 (Quota Exceeded / Rate Limit)
                    if "429" in error_msg or "exhausted" in error_msg.lower() or "quota" in error_msg.lower():
                        print(f"⚠️ Key #{index + 1} (...{key[-4:]}) 429 Error. Switching key...")
                        continue 
                    else:
                        # Raise other errors immediately
                        raise ValueError(f"API Error (Key {index+1}): {error_msg}")
            
            # If all keys failed with 429
            retry_count += 1
            print(f"🛑 All keys exhausted (429). Waiting 15s... (Retry {retry_count}/{max_retries})")
            time.sleep(15)
        
        raise ValueError(f"Failed after {max_retries} retries due to API rate limits.")