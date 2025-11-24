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
    - Multi-API Key support (Inputs grouped together).
    - Robust Error Handling: Switches keys on ANY error (429, 500, 503, Network error, etc.).
    - Auto-retry: Waits 15s and retries if all keys fail (Max 20 retries).
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                # Grouped API Keys for better UI UX
                "api_key_1": ("STRING", {"multiline": False, "default": "", "placeholder": "Primary API Key (Required)"}),
                "api_key_2": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 1 (Optional)"}),
                "api_key_3": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 2 (Optional)"}),
                
                "prompt": ("STRING", {"multiline": True, "default": "Make this image cyberpunk style", "placeholder": "Enter your prompt here..."}),
                "model_name": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "text_response")
    FUNCTION = "process_image"
    CATEGORY = "Gemini AI"

    def process_image(self, image, api_key_1, prompt, model_name, api_key_2="", api_key_3=""):
        # 1. Collect valid API Keys
        keys = [k.strip() for k in [api_key_1, api_key_2, api_key_3] if k and k.strip()]
        
        if not keys:
            raise ValueError("No API Key provided! Please enter at least one API Key.")

        # 2. Prepare Input Image
        img_tensor = image[0] 
        i = 255. * img_tensor.cpu().numpy()
        pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        # Internal API Call Function
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
                    if part.inline_data and part.inline_data.data:
                        print("Gemini Node: Received image data.")
                        image_data = part.inline_data.data
                        out_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
                        out_np = np.array(out_pil).astype(np.float32) / 255.0
                        out_tensor = torch.from_numpy(out_np)[None,]
                        image_found = True
                    
                    if part.text:
                        full_text += part.text

            if not image_found:
                raise ValueError(f"Model returned text only (No Image): {full_text}")

            return out_tensor, full_text

        # 3. Robust Retry Logic
        max_retries = 10 
        retry_count = 0

        while retry_count < max_retries:
            for index, key in enumerate(keys):
                try:
                    return call_api(key)
                except Exception as e:
                    # Catch all errors (Network, 429, 500, etc.)
                    error_msg = str(e)
                    print(f"⚠️ Key #{index + 1} (...{key[-4:]}) Failed. Error: {error_msg}")
                    print(f"➡️ Switching to next key...")
                    continue # Try next key
            
            # If all keys failed
            retry_count += 1
            print(f"🛑 All keys failed. Waiting 15s... (Retry {retry_count}/{max_retries})")
            time.sleep(15)
        
        raise ValueError(f"Failed after {max_retries} retries. All API keys are experiencing issues.")
