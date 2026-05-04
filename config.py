# Video TAA + DLAA node defaults
# MSXYZ


# Internal tuning values from visual tests.
INTERNAL_TUNING = {
    # Fine-line AA
    "fine_line_dark_slope": 12.0,

    # Specular detail
    "specular_clip_luma": 0.92,
    "specular_clip_slope": 20.0,
    "specular_edge_threshold": 0.04,
    "specular_edge_slope": 12.0,
    "specular_highlight_suppression": 0.60,

    # Micro contrast
    "micro_detail_threshold": 0.006,
    "micro_detail_slope": 80.0,
    "micro_highlight_luma": 0.85,
    "micro_highlight_slope": 12.0,

    # Dehalo
    "dehalo_dark_protect_luma": 0.22,
    "dehalo_light_protect_luma": 0.78,
    "dehalo_protect_slope": 12.0,

    # Chroma cleanup
    "chroma_saturation_slope": 20.0,
    "chroma_fringe_threshold_scale": 0.25,
    "chroma_fringe_slope": 80.0,
    "chroma_dark_protect_slope": 12.0,

    # Temporal specular stabilizer
    "temporal_spec_detail_slope": 80.0,
    "temporal_spec_motion_scale": 0.08,

    # Local tone mapping
    "local_tonemap_highlight_luma": 0.82,
    "local_tonemap_highlight_slope": 12.0,
    "local_tonemap_shadow_slope": 10.0,
    "local_tonemap_ratio_min": 0.90,
    "local_tonemap_ratio_max": 1.10,

    # Jitter
    "jitter_damping_min": 0.45,

    # Fur / hair stabilizer
    "fur_detail_slope": 80.0,
}


# Main node defaults.
# Only a small set of controls is exposed in ComfyUI.
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

    # Preset edge AA strength
    "detail_edge_aa_strength": 0.38,
    "photo_edge_aa_strength": 0.25,

    # Fine-line AA for thin details
    "detail_fine_line_aa_strength": 0.45,
    "detail_fine_line_dark_threshold": 0.34,
    "detail_fine_line_edge_threshold": 0.075,
    "detail_fine_line_blur_strength": 0.44,

    # Specular detail
    "detail_specular_strength": 0.12,
    "detail_specular_threshold": 0.52,
    "detail_specular_slope": 8.0,
    "detail_specular_limit": 0.014,
    "detail_specular_edge_boost": 0.22,

    # Micro-contrast
    "detail_micro_contrast_strength": 0.045,
    "detail_micro_contrast_radius": 5,
    "detail_micro_contrast_limit": 0.016,
    "detail_micro_contrast_highlight_protect": 0.70,

    # Edge dehalo
    "detail_dehalo_strength": 0.14,
    "detail_dehalo_threshold": 0.095,
    "detail_dehalo_dark_protect": 0.35,
    "detail_dehalo_light_protect": 0.55,

    # Chroma edge cleanup
    "detail_chroma_cleanup_strength": 0.18,
    "detail_chroma_edge_threshold": 0.055,
    "detail_chroma_saturation_threshold": 0.070,
    "detail_chroma_cleanup_limit": 0.014,
    "detail_chroma_dark_protect": 0.30,

    # Subpixel edge reconstruction
    "detail_subpixel_edge_strength": 0.14,
    "detail_subpixel_edge_threshold": 0.070,
    "detail_subpixel_edge_slope": 16.0,
    "detail_subpixel_sample_scale": 0.35,
    "detail_subpixel_blend_limit": 0.24,
    "detail_subpixel_delta_limit": 0.016,
    "detail_subpixel_motion_protect": 0.75,

    # Temporal specular stabilizer
    "detail_specular_temporal_strength": 0.05,
    "detail_specular_temporal_threshold": 0.56,
    "detail_specular_temporal_slope": 10.0,
    "detail_specular_temporal_detail_threshold": 0.006,
    "detail_specular_temporal_blend_limit": 0.18,
    "detail_specular_temporal_delta_limit": 0.014,
    "detail_specular_temporal_motion_protect": 0.85,

    # Local tone mapping
    "detail_local_tonemap_strength": 0.07,
    "detail_local_tonemap_radius": 11,
    "detail_local_tonemap_limit": 0.018,
    "detail_local_tonemap_shadow_lift": 0.012,
    "detail_local_tonemap_shadow_threshold": 0.38,
    "detail_local_tonemap_highlight_protect": 0.75,
    "detail_local_tonemap_motion_protect": 0.65,

    # Fur / hair directional stabilizer
    "detail_fur_stabilizer_strength": 0.08,
    "detail_fur_edge_threshold": 0.05,
    "detail_fur_detail_threshold": 0.012,
    "detail_fur_blend_limit": 0.20,
    "detail_fur_motion_protect": 0.85,
}


NODE_DEFAULT_FIELDS = (
    "model_weight",
    "dlaa_blend_scale",
    "taa_alpha",
    "jitter_scale",
    "edge_threshold",
    "tone_curve_bias",
    "highlight_pre_blend",
    "highlight_post_blend",
    "highlight_threshold",
    "highlight_slope",
    "edge_sharp_threshold",
    "edge_sharp_slope",
    "edge_aa_slope",
    "edge_detail_limit_scale",
    "motion_gate_scale",
    "jitter_motion_damping",

    "detail_base_scale",
    "detail_ref_scale",
    "detail_min_scale",
    "detail_max_scale",
    "detail_min_gain",
    "detail_max_gain",
    "detail_edge_boost",
    "detail_highlight_suppression",

    "detail_dark_luma_start",
    "detail_dark_luma_range",
    "detail_dark_mix_base",
    "detail_dark_mix_scale",

    "fine_detail_limit",
    "detail_highlight_pre_scale",
    "detail_highlight_post_scale",
    "detail_blend_boost",

    "detail_shimmer_strength",
    "detail_shimmer_threshold",
    "detail_shimmer_slope",
    "detail_shimmer_max_blend",

    "detail_edge_aa_strength",
    "photo_edge_aa_strength",

    "auto_default_scene_motion",
    "auto_static_motion_threshold",
    "auto_balanced_motion_threshold",

    "luma_boost_base",
    "saturation_boost_base",
    "luma_highlight_protect",
    "saturation_highlight_protect",

    "texture_pass_enabled",
    "texture_tile_overlap",
    "texture_log_interval",

    "detail_fine_line_aa_strength",
    "detail_fine_line_dark_threshold",
    "detail_fine_line_edge_threshold",
    "detail_fine_line_blur_strength",

    "detail_specular_strength",
    "detail_specular_threshold",
    "detail_specular_slope",
    "detail_specular_limit",
    "detail_specular_edge_boost",

    "detail_micro_contrast_strength",
    "detail_micro_contrast_radius",
    "detail_micro_contrast_limit",
    "detail_micro_contrast_highlight_protect",

    "detail_dehalo_strength",
    "detail_dehalo_threshold",
    "detail_dehalo_dark_protect",
    "detail_dehalo_light_protect",

    "detail_chroma_cleanup_strength",
    "detail_chroma_edge_threshold",
    "detail_chroma_saturation_threshold",
    "detail_chroma_cleanup_limit",
    "detail_chroma_dark_protect",

    "detail_subpixel_edge_strength",
    "detail_subpixel_edge_threshold",
    "detail_subpixel_edge_slope",
    "detail_subpixel_sample_scale",
    "detail_subpixel_blend_limit",
    "detail_subpixel_delta_limit",
    "detail_subpixel_motion_protect",

    "detail_specular_temporal_strength",
    "detail_specular_temporal_threshold",
    "detail_specular_temporal_slope",
    "detail_specular_temporal_detail_threshold",
    "detail_specular_temporal_blend_limit",
    "detail_specular_temporal_delta_limit",
    "detail_specular_temporal_motion_protect",

    "detail_local_tonemap_strength",
    "detail_local_tonemap_radius",
    "detail_local_tonemap_limit",
    "detail_local_tonemap_shadow_lift",
    "detail_local_tonemap_shadow_threshold",
    "detail_local_tonemap_highlight_protect",
    "detail_local_tonemap_motion_protect",

    "detail_fur_stabilizer_strength",
    "detail_fur_edge_threshold",
    "detail_fur_detail_threshold",
    "detail_fur_blend_limit",
    "detail_fur_motion_protect",
)