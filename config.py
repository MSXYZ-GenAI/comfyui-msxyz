# Video TAA + DLAA node defaults
# MSXYZ

NODE_DEFAULTS = {
    
    "single_image_detail_texture_intensity": 1.15,
    
    # Model blend
    "model_weight": 1.00,
    "dlaa_blend_scale": 1.00,

    # TAA / jitter
    "taa_alpha": 0.10,
    "jitter_scale": 0.20,
    "edge_threshold": 0.15,

    # Highlight handling
    "tone_curve_bias": 0.6,
    "highlight_pre_blend": 0.15,
    "highlight_post_blend": 0.08,
    "highlight_threshold": 0.85,
    "highlight_slope": 12.0,

    # Detail shaping
    "detail_base_scale": 9.0,
    "detail_ref_scale": 0.02,
    "detail_min_scale": 6.0,
    "detail_max_scale": 12.0,
    "detail_min_gain": 0.10,
    "detail_max_gain": 0.26,
    "detail_edge_boost": 0.35,
    "detail_highlight_suppression": 0.5,

    # Edge detail
    "edge_sharp_threshold": 0.08,
    "edge_sharp_slope": 12.0,
    "edge_aa_slope": 14.0,
    "edge_detail_limit_scale": 0.7,

    # Motion handling
    "motion_gate_scale": 12.0,
    "jitter_motion_damping": 8.0,

    # Dark-area detail behavior
    "detail_dark_luma_start": 0.20,
    "detail_dark_luma_range": 0.30,
    "detail_dark_mix_base": 0.6,
    "detail_dark_mix_scale": 0.4,

    # Detail limits
    "fine_detail_limit": 0.15,
    "detail_highlight_pre_scale": 0.45,
    "detail_highlight_post_scale": 0.50,
    "detail_blend_boost": 1.12,

    # Auto preset motion thresholds
    "auto_default_scene_motion": 0.02,
    "auto_static_motion_threshold": 0.015,
    "auto_balanced_motion_threshold": 0.045,

    # Luma / saturation boost
    "luma_boost_base": 0.03,
    "saturation_boost_base": 0.06,
    "luma_highlight_protect": 0.6,
    "saturation_highlight_protect": 0.5,

    # Texture pass
    "texture_pass_enabled": True,
    "texture_tile_overlap": 32,
    "texture_log_interval": 30,
    
    # Texture shimmer
    "detail_shimmer_strength": 0.25,
    "detail_shimmer_threshold": 0.008,
    "detail_shimmer_slope": 100.0,
    "detail_shimmer_max_blend": 0.35,
    "detail_edge_aa_strength": 0.38,
}