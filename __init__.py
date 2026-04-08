from .VideoAdaptiveAA import VideoAdaptiveAA

NODE_CLASS_MAPPINGS = {
    "VideoAdaptiveAA": VideoAdaptiveAA
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoAdaptiveAA": "🚀 Video Adaptive Anti-Aliasing"
}


from .video_aa_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
