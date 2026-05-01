from pathlib import Path
import importlib.util
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_package_init():
    package_name = "comfyui_msxyz_smoke"

    spec = importlib.util.spec_from_file_location(
        package_name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)

    return module


def main():
    package = load_package_init()

    assert "VideoTAADLAA" in package.NODE_CLASS_MAPPINGS
    assert "VideoTAADLAA" in package.NODE_DISPLAY_NAME_MAPPINGS

    from VideoTAADLAA import VideoTAADLAA
    from model import DLAANet
    from utils import clamp01

    node = VideoTAADLAA()

    input_types = VideoTAADLAA.INPUT_TYPES()
    assert "required" in input_types
    assert "optional" in input_types
    assert "images" in input_types["required"]
    assert "preset" in input_types["required"]

    preset, dlaa, texture, motion = node._normalize_run_inputs(
        "Sharp",
        1.2,
        1.1,
        1.0,
    )

    assert preset == "Detail"
    assert dlaa == 1.2
    assert texture == 1.1
    assert motion == 1.0

    preset, _, _, _ = node._normalize_run_inputs(
        "Cinematic",
        1.0,
        1.0,
        1.0,
    )

    assert preset == "Smooth"

    assert clamp01(-1.0) == 0.0
    assert clamp01(0.5) == 0.5
    assert clamp01(2.0) == 1.0

    net = DLAANet()
    x = torch.rand(1, 3, 16, 16)

    luma, edge = node._luma_edge(x, net)

    assert luma.shape == (1, 1, 16, 16)
    assert edge.shape == (1, 1, 16, 16)
    assert torch.isfinite(luma).all()
    assert torch.isfinite(edge).all()

    print("[OK] Smoke test passed.")


if __name__ == "__main__":
    main()