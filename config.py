# Video TAA + DLAA node defaults
# MSXYZ


def _prefixed(prefix, **values):
    return {
        f"{prefix}_{name}": value
        for name, value in values.items()
    }


NODE_DEFAULTS = {
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
    **_prefixed(
        "detail_dark",
        luma_start=0.20,
        luma_range=0.30,
        mix_base=0.6,
        mix_scale=0.4,
    ),

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
    **_prefixed(
        "detail_shimmer",
        strength=0.25,
        threshold=0.008,
        slope=100.0,
        max_blend=0.35,
    ),
    "detail_edge_aa_strength": 0.38,
    "photo_edge_aa_strength": 0.25,

    # Fine-line AA for thin details
    **_prefixed(
        "detail_fine_line",
        aa_strength=0.45,
        dark_threshold=0.34,
        edge_threshold=0.075,
        blur_strength=0.44,
    ),

    # Specular detail
    **_prefixed(
        "detail_specular",
        strength=0.12,
        threshold=0.52,
        slope=8.0,
        limit=0.014,
        edge_boost=0.22,
    ),

    # Micro-contrast
    **_prefixed(
        "detail_micro_contrast",
        strength=0.045,
        radius=5,
        limit=0.016,
        highlight_protect=0.70,
    ),

    # Edge dehalo
    **_prefixed(
        "detail_dehalo",
        strength=0.14,
        threshold=0.095,
        dark_protect=0.35,
        light_protect=0.55,
    ),

    # Chroma edge cleanup
    **_prefixed(
        "detail_chroma",
        cleanup_strength=0.18,  # best range 0.10 - 0.25
        edge_threshold=0.055,
        saturation_threshold=0.070,
        cleanup_limit=0.014,
        dark_protect=0.30,
    ),

    # Subpixel edge reconstruction
    **_prefixed(
        "detail_subpixel",
        edge_strength=0.14,  # best range 0.10 - 0.20
        edge_threshold=0.070,
        edge_slope=16.0,
        sample_scale=0.35,
        blend_limit=0.24,
        delta_limit=0.016,
        motion_protect=0.75,
    ),

    # Temporal specular stabilizer
    **_prefixed(
        "detail_specular_temporal",
        strength=0.05,  # best range 0.05 - 0.15
        threshold=0.56,
        slope=10.0,
        detail_threshold=0.006,
        blend_limit=0.18,
        delta_limit=0.014,
        motion_protect=0.85,
    ),

    # Local tone mapping
    **_prefixed(
        "detail_local_tonemap",
        strength=0.07,  # best range 0.04 - 0.10
        radius=11,
        limit=0.018,
        shadow_lift=0.012,
        shadow_threshold=0.38,
        highlight_protect=0.75,
        motion_protect=0.65,
    ),

    # Fur / hair directional stabilizer
    **_prefixed(
        "detail_fur",
        stabilizer_strength=0.08,
        edge_threshold=0.05,
        detail_threshold=0.012,
        blend_limit=0.20,
        motion_protect=0.85,
    ),
}