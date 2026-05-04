# best_preprocess_tesseract.py
# Pick the best preprocessing for ONE invoice/receipt image using:
# - Tesseract OCR confidence + amount of recognized text
# - Penalty for washed-out (too many near-white pixels)
# - Bonus for text-like edges (stroke visibility)
#
# Usage:
#   python best_preprocess_tesseract.py --image path/to/invoice.jpg --outdir out --debug_save_all
#
# Install:
#   pip install opencv-python numpy pytesseract
#   (and install the Tesseract binary on your OS)

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Optional

import cv2
import numpy as np
import pytesseract


# -------------------- Preprocessing actions --------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gamma_correction(gray: np.ndarray, gamma: float) -> np.ndarray:
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)

def linear_contrast(gray: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    # new = alpha*old + beta
    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

def clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return c.apply(gray)

def denoise(gray: np.ndarray, h: int = 10) -> np.ndarray:
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)

def sharpen(gray: np.ndarray, amount: float = 1.0) -> np.ndarray:
    g = gray.astype(np.float32)
    blur = cv2.GaussianBlur(g, (0, 0), 1.0)
    out = cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)
    return np.clip(out, 0, 255).astype(np.uint8)

def otsu(gray: np.ndarray) -> np.ndarray:
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def adaptive(gray: np.ndarray, block_size: int = 31, C: int = 10) -> np.ndarray:
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )

def build_actions() -> List[Tuple[str, Callable[[np.ndarray], np.ndarray]]]:
    # Keep this small first (5–12). Expand later.
    return [
        ("id", lambda g: g),
        ("gamma_darken_0.8", lambda g: gamma_correction(g, 0.8)),
        ("gamma_brighten_1.2", lambda g: gamma_correction(g, 1.2)),
        ("contrast_up", lambda g: linear_contrast(g, alpha=1.4, beta=-10)),
        ("clahe_2", lambda g: clahe(g, clip_limit=2.0, tile=8)),
        ("denoise_then_contrast", lambda g: linear_contrast(denoise(g, h=10), alpha=1.3, beta=-5)),
        ("sharpen_then_contrast", lambda g: linear_contrast(sharpen(g, amount=1.0), alpha=1.2, beta=-5)),
        ("clahe_then_otsu", lambda g: otsu(clahe(g, 2.0, 8))),
        ("clahe_then_adaptive", lambda g: adaptive(clahe(g, 2.0, 8), block_size=31, C=8)),
    ]


# -------------------- OCR + scoring --------------------

@dataclass
class OCRResult:
    text: str
    mean_conf: float
    num_words: int

def run_tesseract(gray_or_bin: np.ndarray, psm: int = 6) -> OCRResult:
    """
    Uses pytesseract.image_to_data to get word-level confidences.
    psm=6 is a decent default for block text; try 4/6/11 depending on your docs.
    """
    img = gray_or_bin
    if img.ndim != 2:
        img = to_gray(img)

    config = f"--psm {psm} -l ara+eng"

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)

    texts = data.get("text", [])
    confs = data.get("conf", [])

    words = []
    conf_vals = []
    for t, c in zip(texts, confs):
        t = (t or "").strip()
        try:
            c_val = float(c)
        except Exception:
            c_val = -1.0
        # Tesseract uses -1 for non-words/unknown sometimes
        if t and c_val >= 0:
            words.append(t)
            conf_vals.append(c_val)

    text = " ".join(words)
    mean_conf = float(np.mean(conf_vals)) if conf_vals else 0.0
    return OCRResult(text=text, mean_conf=mean_conf, num_words=len(words))

def saturation_ratios(gray: np.ndarray) -> Tuple[float, float]:
    g = np.clip(gray, 0, 255).astype(np.uint8)
    white_ratio = float((g > 245).mean())
    black_ratio = float((g < 10).mean())
    return white_ratio, black_ratio

def edge_ratio(gray: np.ndarray) -> float:
    g = np.clip(gray, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(g, cv2.CV_16S, ksize=3)
    mag = np.abs(lap).astype(np.uint8)
    edges = mag > 20  # heuristic threshold; tune if needed
    return float(edges.mean())

def compute_score(processed: np.ndarray, psm: int = 6) -> Tuple[float, Dict[str, float], OCRResult]:
    g = to_gray(processed)

    w_ratio, b_ratio = saturation_ratios(g)
    e_ratio = edge_ratio(g)

    ocr = run_tesseract(g, psm=psm)

    # OCR score: confidence (0..100) scaled, times some amount of text.
    # cap words to avoid one verbose doc dominating
    ocr_score = (ocr.mean_conf / 100.0) * min(ocr.num_words, 200)

    # penalties/bonuses
    washout_penalty = 8.0 * w_ratio   # strong: your main failure
    crush_penalty = 3.0 * b_ratio
    edge_bonus = 6.0 * e_ratio        # moderate help

    total = ocr_score - washout_penalty - crush_penalty + edge_bonus

    parts = {
        "ocr_score": float(ocr_score),
        "mean_conf": float(ocr.mean_conf),
        "num_words": float(ocr.num_words),
        "white_ratio": float(w_ratio),
        "black_ratio": float(b_ratio),
        "edge_ratio": float(e_ratio),
        "washout_penalty": float(washout_penalty),
        "crush_penalty": float(crush_penalty),
        "edge_bonus": float(edge_bonus),
    }
    return float(total), parts, ocr


# -------------------- Main selection loop --------------------

def pick_best(img_bgr: np.ndarray, psm: int, debug_save_all: bool, outdir: str) -> Dict:
    gray = to_gray(img_bgr)
    actions = build_actions()

    best = {
        "action_id": -1,
        "action_name": "",
        "score": -1e9,
        "parts": {},
        "processed": None,
        "ocr": None,
    }

    os.makedirs(outdir, exist_ok=True)

    for i, (name, fn) in enumerate(actions):
        try:
            proc = fn(gray)
        except Exception as e:
            print(f"[WARN] action {i} {name} failed: {e}")
            continue

        score, parts, ocr = compute_score(proc, psm=psm)

        if debug_save_all:
            cv2.imwrite(os.path.join(outdir, f"try_{i:02d}_{name}.png"), proc)
            with open(os.path.join(outdir, f"try_{i:02d}_{name}.txt"), "w", encoding="utf-8") as f:
                f.write(f"score={score}\n")
                for k, v in parts.items():
                    f.write(f"{k}={v}\n")
                f.write("\n--- OCR TEXT (first 2000 chars) ---\n")
                f.write((ocr.text or "")[:2000])

        if score > best["score"]:
            best.update(
                {
                    "action_id": i,
                    "action_name": name,
                    "score": score,
                    "parts": parts,
                    "processed": proc,
                    "ocr": ocr,
                }
            )

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="images\\receipts_original\\data11.png")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--psm", type=int, default=6, help="Tesseract page segmentation mode (try 4/6/11)")
    ap.add_argument("--debug_save_all", action="store_true", help="Save every attempt + OCR text")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    best = pick_best(img, psm=args.psm, debug_save_all=args.debug_save_all, outdir=args.outdir)

    if best["processed"] is None:
        raise SystemExit("No preprocessing succeeded.")

    print(f"BEST: id={best['action_id']} name={best['action_name']} score={best['score']:.4f}")
    print("Breakdown:")
    for k, v in best["parts"].items():
        print(f"  {k}: {v:.4f}")

    # Save best image + text
    best_img_path = os.path.join(args.outdir, f"best_{best['action_id']:02d}_{best['action_name']}.png")
    cv2.imwrite(best_img_path, best["processed"])

    best_txt_path = os.path.join(args.outdir, f"best_{best['action_id']:02d}_{best['action_name']}.txt")
    with open(best_txt_path, "w", encoding="utf-8") as f:
        #f.write(f"best_action_id={best['action_id']}\n")
        #f.write(f"best_action_name={best['action_name']}\n")
        #f.write(f"score={best['score']}\n\n")
        f.write("--- OCR TEXT ---\n")
        #f.write(best["ocr"].text if best["ocr"] else "")

    #print(f"\nSaved best image: {best_img_path}")
    print(f"Saved best OCR text: {best_txt_path}")


if __name__ == "__main__":
    main()
