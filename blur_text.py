#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blur_text.py — Batch redaction (blur/pixelate/paint/inpaint) of detected text in images.

Features:
  • Works with WEBP for PaddleOCR (passes NumPy array to bypass path suffix checks)
  • 'both' backend -> unions PaddleOCR (polygons) + Tesseract (boxes)
  • Upsampling (--upsample) to catch tiny/far text
  • Multi-rotation (--rotations 0,90,270) to detect vertical banners/signs
  • Tiling (--tile-size, --tile-overlap) for high-res local detection
  • Area filters (--min-area, --max-area-frac) to drop specks/huge regions
  • NMS (--nms-iou) to reduce overlaps/duplicates
  • Recognition-gating (--gate-with-rec tesseract) to keep only true text
  • Debug overlays (--debug-overlay)

Examples:
  python blur_text.py
  python blur_text.py --backend both --upsample 2.0 --rotations 0,90,270 --tile-size 1280 --tile-overlap 0.2 \
      --paddle-min-score 0.35 --tess-min-conf 55 --pad 6 --debug-overlay --recursive
"""

import os
import glob
import argparse
from typing import List, Tuple, Union
import numpy as np
import cv2

# ---------- Optional imports (handled gracefully if missing) ----------
_PADDLE_OK = True
try:
    import paddleocr  # noqa: F401  (presence check; modules imported later)
except Exception:
    _PADDLE_OK = False

_TESS_OK = True
try:
    import pytesseract  # type: ignore
except Exception:
    _TESS_OK = False


# -------------------------- Utilities --------------------------

def ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def list_images(input_folder: str, recursive: bool) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff")
    paths: List[str] = []
    for ext in exts:
        pattern = os.path.join(input_folder, "**", ext) if recursive else os.path.join(input_folder, ext)
        paths.extend(glob.glob(pattern, recursive=recursive))
    return sorted(list(dict.fromkeys(paths)))  # de-dup, stable order


def make_output_path(output_folder: str, image_path: str, method: str) -> str:
    base = os.path.basename(image_path)
    name, _ = os.path.splitext(base)
    out_name = f"{name}.redacted.{method}.png"
    return os.path.join(output_folder, out_name)


def build_mask_from_polygons(shape: Tuple[int, int, int], polygons: List[np.ndarray], pad: int) -> np.ndarray:
    """
    shape: (H, W, C)
    polygons: list of (N, 2) int32 arrays
    pad: pixels to dilate (expands mask around text)
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygons:
        cv2.fillPoly(mask, polygons, 255)
        if pad > 0:
            k = ensure_odd(pad * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def redact_with_method(img: np.ndarray, mask: np.ndarray, method: str,
                       blur_kernel: int, blur_sigma: float,
                       pixel_block: int, inpaint_radius: int) -> np.ndarray:
    """Apply redaction method within mask."""
    if mask.max() == 0:
        return img.copy()

    m = (mask > 0)[:, :, None]

    if method == "black":
        return np.where(m, np.zeros_like(img), img)

    if method == "white":
        return np.where(m, np.full_like(img, 255), img)

    if method == "blur":
        k = ensure_odd(blur_kernel)
        blurred = cv2.GaussianBlur(img, (k, k), blur_sigma)
        return np.where(m, blurred, img)

    if method == "pixelate":
        h, w = img.shape[:2]
        fx = max(1, pixel_block)
        small = cv2.resize(img, (max(1, w // fx), max(1, h // fx)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return np.where(m, pixelated, img)

    if method == "inpaint":
        inpainted = cv2.inpaint(img, mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
        return np.where(m, inpainted, img)

    return img.copy()


def save_debug_overlay(debug_dir: str, image_path: str, img: np.ndarray,
                       polygons: List[np.ndarray], mask: np.ndarray) -> None:
    os.makedirs(debug_dir, exist_ok=True)
    overlay = img.copy()
    for poly in polygons:
        try:
            cv2.polylines(overlay, [poly.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        except Exception:
            pass
    red = np.zeros_like(img); red[:, :, 2] = 255
    overlay = np.where(mask[:, :, None] > 0, (0.6 * overlay + 0.4 * red).astype(overlay.dtype), overlay)
    name, _ = os.path.splitext(os.path.basename(image_path))
    cv2.imwrite(os.path.join(debug_dir, f"{name}.overlay.png"), overlay)


# -------------------------- Geometry (rotations/tiling) --------------------------

def parse_rotations(rot_str: str) -> List[int]:
    vals = []
    for t in rot_str.split(","):
        t = t.strip()
        if not t:
            continue
        a = int(float(t)) % 360
        if a % 90 != 0:
            raise ValueError("Only 90° multiples supported in --rotations.")
        if a not in vals:
            vals.append(a)
    return vals or [0]


def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    if angle == 0:
        return img
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("Angle must be 0, 90, 180, or 270.")


def map_poly_from_rotated(poly: np.ndarray, angle: int, H: int, W: int) -> np.ndarray:
    """
    Map polygon from rotated image coords back to original (pre-rotation) coords.
    Original image size: HxW.
    """
    p = poly.astype(np.float32).copy()
    if angle == 0:
        return p.astype(np.int32)
    if angle == 90:
        # x' = H-1 - y, y' = x  -> inverse: x = y', y = H-1 - x'
        xprime, yprime = p[:, 0], p[:, 1]
        x = yprime
        y = (H - 1) - xprime
        return np.stack([x, y], axis=1).astype(np.int32)
    if angle == 180:
        xprime, yprime = p[:, 0], p[:, 1]
        x = (W - 1) - xprime
        y = (H - 1) - yprime
        return np.stack([x, y], axis=1).astype(np.int32)
    if angle == 270:
        # x' = y, y' = W-1 - x  -> inverse: x = W-1 - y', y = x'
        xprime, yprime = p[:, 0], p[:, 1]
        x = (W - 1) - yprime
        y = xprime
        return np.stack([x, y], axis=1).astype(np.int32)
    raise ValueError("Angle must be 0, 90, 180, or 270.")


def clip_poly(poly: np.ndarray, w: int, h: int) -> np.ndarray:
    p = poly.copy()
    p[:, 0] = np.clip(p[:, 0], 0, w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    return p


def tiles_for_image(w: int, h: int, tile: int, overlap_frac: float):
    """Yield (x0, y0, x1, y1) windows over an image."""
    if tile <= 0:
        yield (0, 0, w, h)
        return
    step = max(1, int(tile * (1.0 - float(overlap_frac))))
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            x1 = min(x0 + tile, w)
            y1 = min(y0 + tile, h)
            yield (x0, y0, x1, y1)
            if x1 == w:
                break
        if y1 == h:
            break


# -------------------------- Detection backends --------------------------

def detect_polygons_paddle(img_or_path: Union[str, np.ndarray], min_score: float) -> List[np.ndarray]:
    """PaddleOCR TextDetection polygons; accepts file path or ndarray (bypasses WEBP suffix issue)."""
    if not _PADDLE_OK:
        raise RuntimeError("PaddleOCR not available. Install: pip install paddleocr (requires paddlepaddle>=3.0).")
    try:
        from paddleocr import TextDetection  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PaddleOCR TextDetection unavailable: {e}")

    if not hasattr(detect_polygons_paddle, "_det"):
        detect_polygons_paddle._det = TextDetection(model_name=None)  # default PP-OCRv5_server_det
    det = detect_polygons_paddle._det  # type: ignore[attr-defined]

    # Choose ndarray for unsupported suffixes
    if isinstance(img_or_path, str):
        ext = os.path.splitext(img_or_path)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".pdf"):
            input_obj: Union[str, np.ndarray] = img_or_path
        else:
            arr = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
            if arr is None:
                raise RuntimeError(f"Failed to read image: {img_or_path}")
            input_obj = arr
    else:
        input_obj = img_or_path

    kwargs = {}
    if min_score > 0:
        kwargs["box_thresh"] = float(min_score)

    output = det.predict(input_obj, batch_size=1, **kwargs)

    polygons: List[np.ndarray] = []
    for res in output:
        js = getattr(res, "json", res)
        data = js.get("res", js) if isinstance(js, dict) else {}
        dt_polys = data.get("dt_polys", [])
        dt_scores = data.get("dt_scores", [])
        dt_polys_iter = dt_polys.tolist() if isinstance(dt_polys, np.ndarray) else dt_polys
        for i, poly in enumerate(dt_polys_iter or []):
            score = float(dt_scores[i]) if i < len(dt_scores) else 1.0
            if score >= min_score:
                polygons.append(np.array(poly, dtype=np.int32))
    return polygons


def detect_polygons_tesseract(img: np.ndarray, min_conf: float,
                              psm: int = 11, lang: str = "eng") -> List[np.ndarray]:
    """Tesseract OCR boxes returned as 4-pt polygons."""
    if not _TESS_OK:
        raise RuntimeError("pytesseract is not installed, or Tesseract binary is not on PATH.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cfg = f"--oem 1 --psm {psm} -l {lang}"
    data = pytesseract.image_to_data(th, config=cfg, output_type=pytesseract.Output.DICT)

    polys: List[np.ndarray] = []
    n = len(data.get("level", []))
    for i in range(n):
        conf_str = str(data["conf"][i]) if "conf" in data else "-1"
        try:
            conf_val = float(conf_str)
        except Exception:
            conf_val = -1.0
        if conf_val < min_conf:
            continue
        x, y = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])
        if w <= 0 or h <= 0:
            continue
        poly = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        polys.append(poly)
    return polys


def detect_on(img: np.ndarray, backend: str,
              paddle_min_score: float, tess_min_conf: float, tess_psm: int, tess_lang: str) -> List[np.ndarray]:
    """Run the selected backend(s) on an image region."""
    polys: List[np.ndarray] = []
    if backend in ("paddle", "both"):
        try:
            polys.extend(detect_polygons_paddle(img, min_score=paddle_min_score))
        except Exception:
            pass
    if backend in ("tesseract", "both"):
        try:
            polys.extend(detect_polygons_tesseract(img, min_conf=tess_min_conf, psm=tess_psm, lang=tess_lang))
        except Exception:
            pass
    return polys


# -------------------------- NMS + recognition-gating --------------------------

def _aabb(poly: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1 = poly[:, 0].min(), poly[:, 1].min()
    x2, y2 = poly[:, 0].max(), poly[:, 1].max()
    return int(x1), int(y1), int(x2), int(y2)


def _iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    iw = max(0, x2 - x1 + 1); ih = max(0, y2 - y1 + 1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1)
    a2 = (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1)
    return inter / float(a1 + a2 - inter)


def nms_polys(polys: List[np.ndarray], iou_thr: float = 0.3) -> List[np.ndarray]:
    """AABB NMS (keeps smaller boxes when overlapping)."""
    if not polys:
        return polys
    boxes = [_aabb(p) for p in polys]
    areas = [(b[2] - b[0] + 1) * (b[3] - b[1] + 1) for b in boxes]
    order = np.argsort(areas)  # small to large
    kept = []
    for i in order:
        bi = boxes[i]
        if any(_iou(bi, boxes[j]) >= iou_thr for j in kept):
            continue
        kept.append(i)
    return [polys[i] for i in kept]


def _best_tess_conf_and_text(crop: np.ndarray, psm: int, lang: str) -> Tuple[float, str]:
    """Try 0/90/270 to handle vertical text; return (best_conf, concatenated_text)."""
    if not _TESS_OK:
        return -1.0, ""
    best = (-1.0, "")
    for ang in (0, 90, 270):
        if ang == 0:
            c = crop
        elif ang == 90:
            c = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        else:
            c = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cfg = f"--oem 1 --psm {psm} -l {lang}"
        data = pytesseract.image_to_data(c, config=cfg, output_type=pytesseract.Output.DICT)
        confs = []
        texts = []
        for conf, txt in zip(data.get("conf", []), data.get("text", [])):
            try:
                confs.append(float(conf))
            except Exception:
                continue
            texts.append((txt or "").strip())
        if confs:
            m = max(confs)
            t = "".join([t for t in texts if t])
            if m > best[0]:
                best = (m, t)
    return best


def gate_polys_with_tesseract(img: np.ndarray, polys: List[np.ndarray],
                              min_conf: float = 70.0, psm: int = 6, lang: str = "eng",
                              min_text_chars: int = 2, pad: int = 2) -> List[np.ndarray]:
    """Keep only polygons whose crop reads as text with decent confidence."""
    if not _TESS_OK or not polys:
        return polys
    H, W = img.shape[:2]
    keep: List[np.ndarray] = []
    for p in polys:
        x1, y1, x2, y2 = _aabb(p)
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = img[y1:y2 + 1, x1:x2 + 1]
        conf, text = _best_tess_conf_and_text(crop, psm=psm, lang=lang)
        if conf >= min_conf and sum(ch.isalnum() for ch in text) >= min_text_chars:
            keep.append(p)
    return keep


# -------------------------- Main --------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch text redaction using PaddleOCR, Tesseract, or both.")
    parser.add_argument("--input", "-i", default="input_images", help="Input folder.")
    parser.add_argument("--output", "-o", default="output_images_retry", help="Output folder.")
    parser.add_argument("--recursive", action="store_true", help="Search input folder recursively.")
    parser.add_argument("--backend", choices=["paddle", "tesseract", "both"], default="paddle",
                        help="Detection backend.")

    # Redaction behavior
    parser.add_argument("--method", choices=["blur", "pixelate", "black", "white", "inpaint"], default="blur")
    parser.add_argument("--pad", type=int, default=2, help="Extra pixels around detected regions (mask dilation).")

    # Blur/pixel/inpaint
    parser.add_argument("--blur-kernel", type=int, default=23)
    parser.add_argument("--blur-sigma", type=float, default=30.0)
    parser.add_argument("--pixel", type=int, default=12)
    parser.add_argument("--inpaint-radius", type=int, default=3)

    # Detector thresholds
    parser.add_argument("--paddle-min-score", type=float, default=0.6,
                        help="Min detection score to keep a polygon (Paddle).")
    parser.add_argument("--tess-min-conf", type=float, default=60.0,
                        help="Min confidence [0..100] (Tesseract).")
    parser.add_argument("--tess-psm", type=int, default=11, help="Tesseract PSM (11 = sparse text).")
    parser.add_argument("--tess-lang", type=str, default="eng", help="Tesseract language code(s).")

    # Upsample + rotations + tiling + filtering
    parser.add_argument("--upsample", type=float, default=1.0, help="Resize factor before detection.")
    parser.add_argument("--rotations", type=str, default="0",
                        help="Comma-separated 90° multiples to run (e.g., '0,90,270').")
    parser.add_argument("--tile-size", type=int, default=0, help="Tile size in px (0=disable).")
    parser.add_argument("--tile-overlap", type=float, default=0.15, help="Tile overlap fraction [0..0.5].")
    parser.add_argument("--min-area", type=int, default=60, help="Drop detections smaller than this area (px).")
    parser.add_argument("--max-area-frac", type=float, default=0.25,
                        help="Drop detections bigger than this fraction of the image area.")

    # NMS + recognition gating
    parser.add_argument("--nms-iou", type=float, default=0.3,
                        help="IoU threshold for dropping overlapping boxes (AABB-based).")
    parser.add_argument("--gate-with-rec", choices=["none", "tesseract"], default="tesseract",
                        help="Confirm candidates via OCR to cut false positives.")
    parser.add_argument("--rec-min-conf", type=float, default=70.0, help="Min OCR confidence for gating.")
    parser.add_argument("--rec-psm", type=int, default=6, help="Tesseract PSM for gating (6 = block).")
    parser.add_argument("--rec-lang", type=str, default="eng+spa", help="Recognition languages for gating.")
    parser.add_argument("--min-text-chars", type=int, default=2, help="Min alphanumeric chars required for gating.")

    # Debug
    parser.add_argument("--debug-overlay", action="store_true", help="Save an overlay showing detected regions.")
    parser.add_argument("--debug-dir", type=str, default="debug_overlays", help="Folder for debug overlays.")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    images = list_images(args.input, recursive=args.recursive)
    if not images:
        print(f"No images found in '{args.input}'.")
        return

    # Optional: Windows Tesseract auto-path
    if args.backend in ("tesseract", "both") and os.name == "nt" and _TESS_OK:
        default_win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_win):
            pytesseract.pytesseract.tesseract_cmd = default_win  # type: ignore

    rotations = parse_rotations(args.rotations)
    print(f"Found {len(images)} image(s) to process with backend='{args.backend}', method='{args.method}'.")
    print(f"Rotations: {rotations} | Tile: {args.tile_size} overlap {args.tile_overlap} | Upsample: {args.upsample}")

    processed = 0
    failed = 0

    for image_path in images:
        try:
            print(f"\nProcessing: {image_path}")
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if img is None:
                print("  ! Failed to load image. Skipping.")
                failed += 1
                continue

            base_h, base_w = img.shape[:2]

            # Upsample to catch small text
            proc = img
            scale = float(args.upsample)
            if scale != 1.0:
                proc = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            Hs, Ws = proc.shape[:2]

            # Collect polygons (in upsampled/original orientation coords)
            polys_proc: List[np.ndarray] = []

            for ang in rotations:
                rot = rotate_image(proc, ang)
                rH, rW = rot.shape[:2]

                for (x0, y0, x1, y1) in tiles_for_image(rW, rH, args.tile_size, args.tile_overlap):
                    crop = rot[y0:y1, x0:x1]
                    local_polys = detect_on(
                        crop, args.backend,
                        args.paddle_min_score, args.tess_min_conf, args.tess_psm, args.tess_lang
                    )
                    # Shift into rotated coords
                    for p in local_polys:
                        p2 = p.copy()
                        p2[:, 0] += x0
                        p2[:, 1] += y0
                        # Map back to pre-rotation (proc) coords
                        p3 = map_poly_from_rotated(p2, ang, Hs, Ws)
                        polys_proc.append(p3)

            # Map to original size if upsampled, clip, and area-filter
            polys_final: List[np.ndarray] = []
            area_max = args.max_area_frac * (base_w * base_h) if args.max_area_frac > 0 else float("inf")
            for p in polys_proc:
                if scale != 1.0:
                    p = np.round(p.astype(np.float32) / scale).astype(np.int32)
                p = clip_poly(p, base_w, base_h)
                a = abs(cv2.contourArea(p))
                if a < args.min_area or a > area_max:
                    continue
                polys_final.append(p)

            # NMS to reduce overlaps/duplicates
            if args.nms_iou > 0:
                polys_final = nms_polys(polys_final, iou_thr=args.nms_iou)

            # Recognition gating to keep only true text
            if args.gate_with_rec != "none":
                polys_final = gate_polys_with_tesseract(
                    img, polys_final,
                    min_conf=args.rec_min_conf,
                    psm=args.rec_psm,
                    lang=args.rec_lang,
                    min_text_chars=args.min_text_chars,
                    pad=max(2, args.pad // 2),
                )

            print(f"  Detected regions: {len(polys_final)}")

            # Mask + redact
            mask = build_mask_from_polygons(img.shape, polys_final, pad=args.pad)
            out = redact_with_method(
                img, mask, method=args.method,
                blur_kernel=args.blur_kernel, blur_sigma=args.blur_sigma,
                pixel_block=args.pixel, inpaint_radius=args.inpaint_radius
            )

            if args.debug_overlay:
                try:
                    save_debug_overlay(args.debug_dir, image_path, img, polys_final, mask)
                except Exception as e:
                    print(f"  ! Debug overlay error: {e}")

            out_path = make_output_path(args.output, image_path, method=args.method)
            ok = cv2.imwrite(out_path, out)
            if not ok:
                raise RuntimeError("cv2.imwrite returned False (save failed).")
            print(f"  Saved -> {out_path}")
            processed += 1

        except Exception as e:
            print(f"  ! Error: {e}")
            failed += 1

    print(f"\nDone. Processed: {processed}, Failed: {failed}, Output folder: {args.output}")


if __name__ == "__main__":
    main()
