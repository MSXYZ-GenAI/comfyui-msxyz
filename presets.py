# Video TAA + DLAA presets
# MSXYZ

PRESETS = {
    "Balanced": {
        "detail_boost": 1.00,
        "edge_boost": 1.00,
        "temporal_strength": 0.35,
        "micro_limit": 0.040,
        "luma_boost_mult": 1.00,
        "saturation_boost_mult": 1.00,
        "motion_threshold": 0.08,
        "taa_strength": 0.45,
        "dlaa_strength": 0.65,
        "tone_strength": 0.12,
        "edge_sharp_strength": 0.14,
        "motion_sensitivity": 0.08,
        "jitter_scale": 0.20,
    },
    "Detail": {
        "detail_boost": 1.44,
        "edge_boost": 1.36,
        "temporal_strength": 0.00,
        "micro_limit": 0.050,
        "luma_boost_mult": 1.08,
        "saturation_boost_mult": 1.03,
        "motion_threshold": 0.050,
        "taa_strength": 0.00,
        "dlaa_strength": 1.25,
        "tone_strength": 0.06,
        "edge_sharp_strength": 0.14,
        "motion_sensitivity": 0.060,
        "jitter_scale": 0.0,
    },
    "Smooth": {
        "detail_boost": 0.85,
        "edge_boost": 0.75,
        "temporal_strength": 0.45,
        "micro_limit": 0.025,
        "luma_boost_mult": 0.80,
        "saturation_boost_mult": 0.70,
        "motion_threshold": 0.10,
        "taa_strength": 0.65,
        "dlaa_strength": 0.65,
        "tone_strength": 0.16,
        "edge_sharp_strength": 0.08,
        "motion_sensitivity": 0.10,
        "jitter_scale": 0.25,
    },
}


AUTO_STATIC = {
    "detail_boost": 1.12,
    "edge_boost": 1.15,
    "temporal_strength": 0.28,
    "micro_limit": 0.045,
    "luma_boost_mult": 1.15,
    "saturation_boost_mult": 1.05,
    "motion_threshold": 0.07,
    "taa_strength": 0.35,
    "dlaa_strength": 0.70,
    "tone_strength": 0.10,
    "edge_sharp_strength": 0.18,
    "motion_sensitivity": 0.07,
    "jitter_scale": 0.15,
}


AUTO_BALANCED = {
    "detail_boost": 1.00,
    "edge_boost": 1.00,
    "temporal_strength": 0.35,
    "micro_limit": 0.040,
    "luma_boost_mult": 1.00,
    "saturation_boost_mult": 1.00,
    "motion_threshold": 0.08,
    "taa_strength": 0.45,
    "dlaa_strength": 0.65,
    "tone_strength": 0.12,
    "edge_sharp_strength": 0.14,
    "motion_sensitivity": 0.08,
    "jitter_scale": 0.20,
}


AUTO_MOTION = {
    "detail_boost": 0.85,
    "edge_boost": 0.75,
    "temporal_strength": 0.42,
    "micro_limit": 0.025,
    "luma_boost_mult": 0.85,
    "saturation_boost_mult": 0.75,
    "motion_threshold": 0.10,
    "taa_strength": 0.60,
    "dlaa_strength": 0.60,
    "tone_strength": 0.16,
    "edge_sharp_strength": 0.08,
    "motion_sensitivity": 0.10,
    "jitter_scale": 0.25,
}


MOTION_SUPPRESSION = {
    "Detail": {
        "detail": 0.05,
        "edge": 0.08,
        "micro": 0.18,
    },
    "Default": {
        "detail": 0.20,
        "edge": 0.15,
        "micro": 0.20,
    },
}


PRESET_MODEL_WEIGHT = {
    "Balanced": 1.00,
    "Detail": 1.38,
    "Smooth": 0.90,
    "Auto": 1.10,
}


TEXTURE_PRESETS = {
    "Detail": {
        "enabled": True,
        "strength": 1.25,
        "limit": 0.040,
        "blur_kernel": 7,
        "dark_base": 0.40,
        "motion_suppression": 0.80,
        "highlight_suppression": 0.55,
        "line_suppression": 0.55,
        "edge_threshold": 0.060,
        "edge_slope": 18.0,
    },
    "Auto": {
        "enabled": True,
        "strength": 0.85,
        "limit": 0.030,
        "blur_kernel": 7,
        "dark_base": 0.35,
        "motion_suppression": 0.90,
        "highlight_suppression": 0.60,
        "line_suppression": 0.60,
        "edge_threshold": 0.060,
        "edge_slope": 18.0,
    },
    "Balanced": {
        "enabled": True,
        "strength": 0.60,
        "limit": 0.020,
        "blur_kernel": 5,
        "dark_base": 0.30,
        "motion_suppression": 0.95,
        "highlight_suppression": 0.65,
        "line_suppression": 0.65,
        "edge_threshold": 0.065,
        "edge_slope": 18.0,
    },
    "Smooth": {
        "enabled": False,
        "strength": 0.25,
        "limit": 0.015,
        "blur_kernel": 5,
        "dark_base": 0.70,
        "motion_suppression": 0.95,
        "highlight_suppression": 0.75,
        "line_suppression": 0.65,
        "edge_threshold": 0.070,
        "edge_slope": 18.0,
    },
}