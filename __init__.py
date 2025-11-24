from .gemini_node import Gemini3ProImageNode

NODE_CLASS_MAPPINGS = {
    "Gemini3ProImageNode": Gemini3ProImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Gemini3ProImageNode": "Gemini 3 Pro"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
