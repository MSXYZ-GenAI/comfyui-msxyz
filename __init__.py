from .LightExplosion import LightExplosion

NODE_CLASS_MAPPINGS = {
    "LightExplosion": LightExplosion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LightExplosion": "Color Correction & Adjustment"
}


from .video_aa_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']