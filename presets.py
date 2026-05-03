# Video TAA + DLAA presets
# MSXYZ


def frame_preset(
    detail_boost=1.00,
    edge_boost=1.00,
    temporal_strength=0.35,
    micro_limit=0.040,
    luma_boost_mult=1.00,
    saturation_boost_mult=1.00,
    motion_threshold=0.08,
    taa_strength=0.45,
    dlaa_strength=0.65,
    tone_strength=0.12,
    edge_sharp_strength=0.14,
    motion_sensitivity=0.08,
    jitter_scale=0.20,
):
    return {
        "detail_boost": detail_boost,
        "edge_boost": edge_boost,
        "temporal_strength": temporal_strength,
        "micro_limit": micro_limit,
        "luma_boost_mult": luma_boost_mult,
        "saturation_boost_mult": saturation_boost_mult,
        "motion_threshold": motion_threshold,
        "taa_strength": taa_strength,
        "dlaa_strength": dlaa_strength,
        "tone_strength": tone_strength,
        "edge_sharp_strength": edge_sharp_strength,
        "motion_sensitivity": motion_sensitivity,
        "jitter_scale": jitter_scale,
    }


PRESETS = {
    "Balanced": frame_preset(),
    
    "Performance": frame_preset(
        detail_boost=1.05,
        edge_boost=1.05,
        temporal_strength=0.24,
        micro_limit=0.026,
        luma_boost_mult=0.96,
        saturation_boost_mult=0.92,
        motion_threshold=0.085,
        taa_strength=0.42,
        dlaa_strength=0.85,
        tone_strength=0.10,
        edge_sharp_strength=0.16,
        motion_sensitivity=0.10,
        jitter_scale=0.16,
    ),

    "Detail": frame_preset(
        detail_boost=1.44,
        edge_boost=1.36,
        temporal_strength=0.00,
        micro_limit=0.050,
        luma_boost_mult=1.08,
        saturation_boost_mult=1.03,
        motion_threshold=0.050,
        taa_strength=0.00,
        dlaa_strength=1.25,
        tone_strength=0.06,
        motion_sensitivity=0.060,
        jitter_scale=0.0,
    ),

    "Smooth": frame_preset(
        detail_boost=0.85,
        edge_boost=0.75,
        temporal_strength=0.45,
        micro_limit=0.025,
        luma_boost_mult=0.80,
        saturation_boost_mult=0.70,
        motion_threshold=0.10,
        taa_strength=0.65,
        tone_strength=0.16,
        edge_sharp_strength=0.08,
        motion_sensitivity=0.10,
        jitter_scale=0.25,
    ),

    "Photo": frame_preset(
        detail_boost=0.92,
        edge_boost=0.24,
        temporal_strength=0.08,
        micro_limit=0.022,
        luma_boost_mult=0.98,
        saturation_boost_mult=0.98,
        taa_strength=0.06,
        dlaa_strength=0.90,
        tone_strength=0.07,
        edge_sharp_strength=0.028,
        motion_sensitivity=0.50,
        jitter_scale=0.0,
    ),
}


AUTO_STATIC = frame_preset(
    detail_boost=1.12,
    edge_boost=1.15,
    temporal_strength=0.28,
    micro_limit=0.045,
    luma_boost_mult=1.15,
    saturation_boost_mult=1.05,
    motion_threshold=0.07,
    taa_strength=0.35,
    dlaa_strength=0.70,
    tone_strength=0.10,
    edge_sharp_strength=0.18,
    jitter_scale=0.15,
)


AUTO_BALANCED = frame_preset()


AUTO_MOTION = frame_preset(
    detail_boost=0.85,
    edge_boost=0.75,
    temporal_strength=0.42,
    micro_limit=0.025,
    luma_boost_mult=0.85,
    saturation_boost_mult=0.75,
    motion_threshold=0.10,
    taa_strength=0.60,
    dlaa_strength=0.60,
    tone_strength=0.16,
    edge_sharp_strength=0.08,
    motion_sensitivity=0.10,
    jitter_scale=0.25,
)

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
    "Performance": 0.95,
    "Balanced": 1.00,
    "Detail": 1.38,
    "Smooth": 0.90,
    "Photo": 0.85,
    "Auto": 1.10,
}


def texture_preset(
    strength,
    limit,
    *,
    enabled=True,
    blur_kernel=5,
    dark_base=0.30,
    motion_suppression=0.95,
    highlight_suppression=0.65,
    line_suppression=0.65,
    edge_threshold=0.065,
    edge_slope=18.0,
):
    return {
        "enabled": enabled,
        "strength": strength,
        "limit": limit,
        "blur_kernel": blur_kernel,
        "dark_base": dark_base,
        "motion_suppression": motion_suppression,
        "highlight_suppression": highlight_suppression,
        "line_suppression": line_suppression,
        "edge_threshold": edge_threshold,
        "edge_slope": edge_slope,
    }


TEXTURE_PRESETS = {
    "Detail": texture_preset(
        strength=1.25,
        limit=0.040,
        blur_kernel=7,
        dark_base=0.40,
        motion_suppression=0.80,
        highlight_suppression=0.55,
        line_suppression=0.55,
        edge_threshold=0.060,
    ),
    "Auto": texture_preset(
        strength=0.85,
        limit=0.030,
        blur_kernel=7,
        dark_base=0.35,
        motion_suppression=0.90,
        highlight_suppression=0.60,
        line_suppression=0.60,
        edge_threshold=0.060,
    ),
    "Performance": texture_preset(
        enabled=False,
        strength=0.40,
        limit=0.015,
    ),
    "Balanced": texture_preset(
        strength=0.60,
        limit=0.020,
    ),
    "Smooth": texture_preset(
        enabled=False,
        strength=2.25,
        limit=0.015,
        dark_base=0.70,
        highlight_suppression=0.75,
        edge_threshold=0.070,
    ),
    "Photo": texture_preset(
        strength=0.36,
        limit=0.013,
        dark_base=0.24,
        highlight_suppression=0.65,
        line_suppression=0.80,
        edge_threshold=0.070,
        edge_slope=12.0,
    ),
}