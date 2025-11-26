import torch
import numpy as np
from PIL import Image
import io
import time
from google import genai
from google.genai import types

class Gemini3ProImageNode:
    """
    ComfyUI Node for Google Gemini 3 Pro.
    Features:
    - Supports 'gemini-3-pro-image-preview'.
    - Supports up to 3 optional input images.
    - Multi-API Key support (Keys grouped together in UI).
    - Auto-retry on error (Max 10 retries).
    - Configurable output resolution.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # UI Layout: Group API Keys together at the top
                "api_key_1": ("STRING", {"multiline": False, "default": "", "placeholder": "Primary API Key (Required)"}),
                "api_key_2": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 1 (Optional)"}),
                "api_key_3": ("STRING", {"multiline": False, "default": "", "placeholder": "Backup API Key 2 (Optional)"}),
                
                "prompt": ("STRING", {"multiline": True, "default": "Make this image cyberpunk style", "placeholder": "Enter your prompt here..."}),
                "model_name": (["gemini-3-pro-image-preview"], {"default": "gemini-3-pro-image-preview"}),
                
                # New Resolution Selection
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
            },
            "optional": {
                # Input Images (Connectors)
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("processed_image", "text_response")
    FUNCTION = "process_image"
    CATEGORY = "Gemini AI"

    def process_image(self, api_key_1, prompt, model_name, resolution, api_key_2="", api_key_3="", image_1=None, image_2=None, image_3=None):
        # 1. Validate Prompt
        if not prompt or not prompt.strip():
            raise ValueError("Prompt is required! Please enter a text prompt.")

        # 2. Collect Valid API Keys
        keys = [k.strip() for k in [api_key_1, api_key_2, api_key_3] if k and k.strip()]
        if not keys:
            raise ValueError("No API Key provided! Please enter at least one API Key.")

        # 3. Process Input Images
        def tensor_to_bytes(image_tensor):
            i = 255. * image_tensor[0].cpu().numpy()
            pil_img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            return buffered.getvalue()

        image_parts = []
        if image_1 is not None:
            image_parts.append(types.Part.from_bytes(data=tensor_to_bytes(image_1), mime_type="image/png"))
        if image_2 is not None:
            image_parts.append(types.Part.from_bytes(data=tensor_to_bytes(image_2), mime_type="image/png"))
        if image_3 is not None:
            image_parts.append(types.Part.from_bytes(data=tensor_to_bytes(image_3), mime_type="image/png"))

        print(f"Gemini Node: Processing with {len(image_parts)} input images. Output Resolution: {resolution}")

        # 4. Internal API Call Function
        def call_api(current_key):
            client = genai.Client(api_key=current_key)
            
            # Configure Generation with selected Resolution
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(image_size=resolution)
            )

            print(f"Gemini Node: Sending request to {model_name} using Key ending in ...{current_key[-4:]}")

            # Construct content parts (Prompt + Images)
            content_parts = [types.Part.from_text(text=prompt)] + image_parts

            response_stream = client.models.generate_content_stream(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=content_parts,
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

        # 5. Robust Retry Logic (Max 10 retries)
        max_retries = 10 
        retry_count = 0

        while retry_count < max_retries:
            for index, key in enumerate(keys):
                try:
                    return call_api(key)
                except Exception as e:
                    error_msg = str(e)
                    print(f"⚠️ Key #{index + 1} (...{key[-4:]}) Failed. Error: {error_msg}")
                    print(f"➡️ Switching to next key...")
                    continue 
            
            retry_count += 1
            print(f"🛑 All keys failed. Waiting 15s... (Retry {retry_count}/{max_retries})")
            time.sleep(15)
        
        raise ValueError(f"Failed after {max_retries} retries. All API keys are experiencing issues.")
