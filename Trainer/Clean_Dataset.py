# Dataset cleaner for DLAA / anti-aliasing training
# Moves weak / broken / duplicate images into _rejected instead of deleting them.


import argparse
import json
import shutil
import warnings
from pathlib import Path

from PIL import Image, ImageStat, ImageOps, ImageFilter


warnings.filterwarnings("ignore", category=DeprecationWarning)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTS


def safe_open(path: Path):
    try:
        img = Image.open(path)
        img.verify()
        img = Image.open(path).convert("RGB")
        return img, None
    except Exception as e:
        return None, f"corrupt:{e}"


def luma(img: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img)


def brightness_score(gray: Image.Image) -> float:
    stat = ImageStat.Stat(gray.resize((256, 256), Image.BILINEAR))
    return float(stat.mean[0])


def contrast_score(gray: Image.Image) -> float:
    stat = ImageStat.Stat(gray.resize((256, 256), Image.BILINEAR))
    return float(stat.stddev[0])


def detail_score(gray: Image.Image) -> float:
    small = gray.resize((256, 256), Image.BILINEAR)
    px = small.load()
    w, h = small.size

    total = 0.0
    count = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            c = px[x, y]
            lap = abs(
                4 * c
                - px[x - 1, y]
                - px[x + 1, y]
                - px[x, y - 1]
                - px[x, y + 1]
            )
            total += lap
            count += 1

    return total / max(count, 1)


def edge_density_score(gray: Image.Image, threshold: int = 18) -> float:
    small = gray.resize((256, 256), Image.BILINEAR)
    px = small.load()
    w, h = small.size

    hits = 0
    count = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = abs(px[x + 1, y] - px[x - 1, y])
            gy = abs(px[x, y + 1] - px[x, y - 1])
            if gx + gy >= threshold:
                hits += 1
            count += 1

    return hits / max(count, 1)


def thin_detail_score(gray: Image.Image) -> float:
    small = gray.resize((256, 256), Image.BILINEAR)
    blur = small.filter(ImageFilter.GaussianBlur(radius=1.0))

    a = list(small.getdata())
    b = list(blur.getdata())

    total = sum(abs(x - y) for x, y in zip(a, b))
    return total / max(len(a), 1)


def document_page_score(gray: Image.Image):
    small = gray.resize((256, 256), Image.BILINEAR)
    vals = list(small.getdata())

    white_ratio = sum(v > 235 for v in vals) / len(vals)
    dark_ratio = sum(v < 80 for v in vals) / len(vals)
    mid_ratio = sum(80 <= v <= 220 for v in vals) / len(vals)

    return white_ratio, dark_ratio, mid_ratio


def avg_hash(gray: Image.Image, size: int = 8) -> str:
    small = gray.resize((size, size), Image.BILINEAR)
    vals = list(small.getdata())
    mean = sum(vals) / len(vals)
    return "".join("1" if v >= mean else "0" for v in vals)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def reject_reasons(path: Path, img: Image.Image, args, seen_hashes) -> list[str]:
    reasons = []
    w, h = img.size

    if w < args.min_width or h < args.min_height:
        reasons.append(f"too_small:{w}x{h}")

    aspect = max(w / max(h, 1), h / max(w, 1))
    if aspect > args.max_aspect:
        reasons.append(f"bad_aspect:{aspect:.2f}")

    gray = luma(img)

    bright = brightness_score(gray)
    contrast = contrast_score(gray)
    detail = detail_score(gray)
    edge_density = edge_density_score(gray, threshold=args.edge_threshold)
    thin_detail = thin_detail_score(gray)
    white_ratio, dark_ratio, mid_ratio = document_page_score(gray)

    if bright < args.min_brightness:
        reasons.append(f"too_dark:{bright:.1f}")
    elif bright > args.max_brightness:
        reasons.append(f"too_bright:{bright:.1f}")

    if contrast < args.min_contrast:
        reasons.append(f"low_contrast:{contrast:.1f}")

    looks_like_document = (
        white_ratio >= args.doc_white_ratio
        and args.doc_dark_min <= dark_ratio <= args.doc_dark_max
        and mid_ratio <= args.doc_mid_max
    )

    if looks_like_document:
        reasons.append(
            f"document_page:white={white_ratio:.2f},dark={dark_ratio:.2f},mid={mid_ratio:.2f}"
        )

    useful_for_aa = (
        detail >= args.min_detail
        and edge_density >= args.min_edge_density
        and thin_detail >= args.min_thin_detail
    )

    # Keep strong edge/detail images
    edge_exception = (
        edge_density >= args.edge_keep_density
        and detail >= args.edge_keep_detail
    )

    # Keep HD portraits
    portrait_like_exception = (
        w >= args.portrait_min_width
        and h >= args.portrait_min_height
        and contrast >= args.portrait_min_contrast
        and thin_detail >= args.portrait_min_thin_detail
    )

    if not useful_for_aa and not edge_exception and not portrait_like_exception:
        reasons.append(
            f"not_useful_for_aa:"
            f"detail={detail:.2f},"
            f"edge={edge_density:.3f},"
            f"thin={thin_detail:.2f}"
        )

    ah = avg_hash(gray)

    for old_hash, old_path in seen_hashes:
        if hamming(ah, old_hash) <= args.duplicate_hamming:
            reasons.append(f"duplicate_of:{old_path.name}")
            break

    seen_hashes.append((ah, path))
    return reasons


def unique_destination(path: Path, rejected_dir: Path) -> Path:
    target = rejected_dir / path.name

    if not target.exists():
        return target

    stem = path.stem
    suffix = path.suffix
    i = 1

    while True:
        candidate = rejected_dir / f"{stem}_{i:03d}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def clean_dataset(args):
    dataset_dir = Path(args.dataset)
    rejected_dir = dataset_dir / args.rejected_dir
    rejected_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in dataset_dir.iterdir() if is_image_file(p)])

    if not files:
        print(f"No images found in: {dataset_dir}")
        return

    report = []
    seen_hashes = []
    kept = 0
    rejected = 0

    print("==============================================================")
    print("DLAA Dataset Cleaner")
    print("Mode: anti-aliasing / thin-detail focused")
    print("==============================================================")
    print(f"Dataset : {dataset_dir}")
    print(f"Images  : {len(files)}")
    print(f"Action  : {'DRY RUN' if args.dry_run else f'MOVE TO {args.rejected_dir}'}")
    print("==============================================================")

    for path in files:
        img, error = safe_open(path)

        if error:
            reasons = [error]
        else:
            reasons = reject_reasons(path, img, args, seen_hashes)

        if reasons:
            rejected += 1
            report.append(
                {
                    "file": path.name,
                    "action": "reject",
                    "reasons": reasons,
                }
            )

            print(f"[REJECT] {path.name} -> {', '.join(reasons)}")

            if not args.dry_run:
                dst = unique_destination(path, rejected_dir)
                shutil.move(str(path), str(dst))
        else:
            kept += 1
            report.append(
                {
                    "file": path.name,
                    "action": "keep",
                    "reasons": [],
                }
            )

    report_path = dataset_dir / "clean_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("==============================================================")
    print(f"Kept     : {kept}")
    print(f"Rejected : {rejected}")
    print(f"Report   : {report_path}")

    if args.dry_run:
        print("Dry run only. Nothing was moved.")
    else:
        print(f"Rejected images moved to: {rejected_dir}")

    print("==============================================================")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean image dataset for DLAA / anti-aliasing training"
    )

    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--rejected-dir", type=str, default="_rejected")

    parser.add_argument("--min-width", type=int, default=512)
    parser.add_argument("--min-height", type=int, default=512)
    parser.add_argument("--max-aspect", type=float, default=3.0)

    parser.add_argument("--min-brightness", type=float, default=10.0)
    parser.add_argument("--max-brightness", type=float, default=248.0)
    parser.add_argument("--min-contrast", type=float, default=10.0)

    # Soft defaults: keeps portraits, fur, hair, wires and useful textures.
    parser.add_argument("--min-detail", type=float, default=2.6)
    parser.add_argument("--min-edge-density", type=float, default=0.025)
    parser.add_argument("--min-thin-detail", type=float, default=1.8)
    parser.add_argument("--edge-threshold", type=int, default=18)

    parser.add_argument("--edge-keep-density", type=float, default=0.055)
    parser.add_argument("--edge-keep-detail", type=float, default=2.6)

    parser.add_argument("--portrait-min-width", type=int, default=768)
    parser.add_argument("--portrait-min-height", type=int, default=768)
    parser.add_argument("--portrait-min-contrast", type=float, default=8.0)
    parser.add_argument("--portrait-min-thin-detail", type=float, default=1.4)
    
    parser.add_argument("--doc-white-ratio", type=float, default=0.72)
    parser.add_argument("--doc-dark-min", type=float, default=0.015)
    parser.add_argument("--doc-dark-max", type=float, default=0.22)
    parser.add_argument("--doc-mid-max", type=float, default=0.22)
    parser.add_argument("--duplicate-hamming", type=int, default=2)

    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    clean_dataset(parse_args())