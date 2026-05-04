# extract_text.py
#
# End-to-end pipeline for pre-extracted OCR text files:
# 1) Loop over .txt files in a folder you choose
# 2) Read receipt OCR text directly from each file
# 3) Extract: vendor name, date, invoice number, total (via Groq or Ollama)
# 4) Write ordered JSON (sorted by filename)
#
# Install:
#   pip install opencv-python numpy pytesseract joblib
# Also install the Tesseract binary + language data (ara, eng).
#
# NOTE:
# - Vendor/date/invoice/total extraction is heuristic (rule-based).
# - It will return null when it can't confidently find a field (this is normal).

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import joblib
import numpy as np
import pytesseract
import requests
import time

# ===================== USER CONFIG =====================

INPUT_TXT_DIR = Path(r"D:\UNI\252\opencv project\output3")  # change later to your OCR text folder
OUT_JSON = Path(r"extracted_receipts.json")
LLM_PROVIDER = "groq"  # "groq" or "ollama"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:14b"   # or whatever you pulled in Ollama

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# qwen/qwen3-32b rate limits from your screenshot.
GROQ_LIMIT_RPM = 30
GROQ_LIMIT_RPD = 1000
GROQ_LIMIT_TPM = 30000
GROQ_LIMIT_TPD = 500000
GROQ_MAX_OUTPUT_TOKENS = 256


LLM_TIMEOUT = 100


TESS_LANG = "ara+eng"
TESS_PSM = 6  # try 6 or 11 if needed

# =======================================================


# -------------------- Preprocessing (same idea as your labeling script) --------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gamma_correction(gray: np.ndarray, gamma: float) -> np.ndarray:
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)

def linear_contrast(gray: np.ndarray, alpha: float, beta: float) -> np.ndarray:
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
    # IMPORTANT: keep the same order you used during label generation.
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


# -------------------- Features + model prediction --------------------

def extract_features(img_bgr: np.ndarray) -> np.ndarray:
    g = to_gray(img_bgr)
    g = np.clip(g, 0, 255).astype(np.uint8)

    mean = float(np.mean(g))
    std = float(np.std(g))

    p5, p25, p50, p75, p95 = np.percentile(g, [5, 25, 50, 75, 95]).astype(float)

    white_ratio = float((g > 245).mean())
    black_ratio = float((g < 10).mean())

    lap = cv2.Laplacian(g, cv2.CV_16S, ksize=3)
    mag = np.abs(lap).astype(np.uint8)
    edge_ratio = float((mag > 20).mean())

    h, w = g.shape
    tile = 48
    tile_stds = []
    for yy in range(0, h, tile):
        for xx in range(0, w, tile):
            patch = g[yy:min(yy + tile, h), xx:min(xx + tile, w)]
            if patch.size >= 400:
                tile_stds.append(float(np.std(patch)))
    local_std_mean = float(np.mean(tile_stds)) if tile_stds else 0.0
    local_std_p90 = float(np.percentile(tile_stds, 90)) if tile_stds else 0.0

    return np.array(
        [
            mean, std,
            p5, p25, p50, p75, p95,
            white_ratio, black_ratio,
            edge_ratio,
            local_std_mean, local_std_p90,
        ],
        dtype=np.float32,
    )

def predict_best_action_id(img_bgr: np.ndarray, payload: dict) -> int:
    t1 = float(payload["thresholds"]["t1"])
    t4 = float(payload["thresholds"]["t4"])

    clf_1 = payload["models"]["is_1"]
    clf_4 = payload["models"]["is_4"]
    clf_other = payload["models"]["other"]

    x = extract_features(img_bgr).reshape(1, -1)

    p_is_1 = float(clf_1.predict_proba(x)[0][1])
    if p_is_1 >= t1:
        return 1

    p_is_4 = float(clf_4.predict_proba(x)[0][1])
    if p_is_4 >= t4:
        return 4

    return int(clf_other.predict(x)[0])


# -------------------- OCR --------------------

def run_tesseract(gray_or_bin: np.ndarray, lang: str, psm: int) -> str:
    img = gray_or_bin
    if img.ndim != 2:
        img = to_gray(img)

    config = f"--psm {psm} -l {lang}"
    # image_to_string is enough here; you can switch to image_to_data later for confidence-based extraction
    text = pytesseract.image_to_string(img, config=config)
    return text or ""


# -------------------- Field extraction (heuristics) --------------------

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_digits(s: str) -> str:
    return s.translate(ARABIC_DIGITS)

def clean_lines(text: str) -> List[str]:
    # Keep useful lines, remove super short garbage
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # collapse spaces
        line = re.sub(r"\s+", " ", line)
        # drop lines that are almost entirely punctuation
        if sum(ch.isalnum() for ch in line) < 3:
            continue
        lines.append(line)
    return lines

def extract_vendor(lines: List[str]) -> Optional[str]:
    # Heuristic: first "name-like" line not containing common invoice keywords
    bad_keywords = [
        "فاتورة", "ضريبية", "مبسطة", "رقم", "الرقم", "الضريبة", "ضريبة",
        "invoice", "tax", "vat", "subtotal", "total", "thank", "شكرا", "شكراً",
        "date", "time", "printed", "created"
    ]
    for line in lines[:12]:  # vendor is usually near top
        low = line.lower()
        if any(k in low for k in bad_keywords):
            continue
        # ignore mostly numeric lines
        if sum(ch.isdigit() for ch in line) > max(4, len(line) // 2):
            continue
        # keep if it has letters (arabic or latin)
        if re.search(r"[A-Za-z\u0600-\u06FF]", line):
            return line
    return None

def extract_date(text: str) -> Optional[str]:
    t = normalize_digits(text)

    # YYYY/MM/DD or YYYY-MM-DD
    m = re.search(r"\b(20\d{2})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    # DD/MM/YYYY or DD-MM-YYYY
    m = re.search(r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](20\d{2})\b", t)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{y:04d}-{mo:02d}-{d:02d}"

    return None

def extract_invoice_number(text: str) -> Optional[str]:
    t = normalize_digits(text)

    # Keyword-based (Arabic/English)
    patterns = [
        r"(?:رقم\s*الفاتورة|فاتورة\s*#|فاتورة#)\s*[:\-]?\s*([A-Za-z0-9\-\/]{4,})",
        r"(?:invoice\s*(?:no|number)?|invoice#|inv\.?\s*no|inv|inv[#\-])\s*[:\-]?\s*([A-Za-z0-9\-\/]{4,})"

    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback: something like 003603-200-26
    m = re.search(r"\b\d{3,}-\d{2,}-\d{2,}\b", t)
    if m:
        return m.group(0)

    return None

def parse_amount(s: str) -> Optional[float]:
    s = normalize_digits(s)
    # keep digits + dot + comma
    s = re.sub(r"[^0-9\.,]", "", s)
    if not s:
        return None
    # If comma used as thousands separator, remove it.
    # If comma used as decimal, replace with dot (simple heuristic).
    if s.count(",") > 0 and s.count(".") == 0:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

def extract_total(text: str) -> Optional[float]:
    t = normalize_digits(text)

    # Search for total keywords; take the LAST reasonable match
    total_keywords = [
        r"total", r"subtotal", r"sub",  r"amount\s*due", r"grand\s*total",
        r"اجمالي", r"إجمالي", r"المجموع", r"المتجموع", r"احمالي", r"الإجمالي"
    ]

    # Match: keyword ... number
    matches: List[float] = []
    for kw in total_keywords:
        # allow SAR markers (رس / ر س / SAR) around the number
        pat = rf"(?:{kw})\s*[:\-]?\s*(?:ر\.?\s*س|رس|sar|sr)?\s*([0-9][0-9\.,]{{1,}})"
        for m in re.finditer(pat, t, flags=re.IGNORECASE):
            amt = parse_amount(m.group(1))
            if amt is not None and 0.5 <= amt <= 100000:
                matches.append(amt)

    if matches:
        return matches[-1]

    return None

def strip_bidi_marks(s: str) -> str:
    if not isinstance(s, str):
        return s
    return s.replace("\u200e", "").replace("\u200f", "")


def _null_fields() -> dict:
    return {"vendor": None, "date": None, "invoice_number": None, "total": None}


def _extract_json_fields(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return _null_fields()

    try:
        obj = json.loads(raw[start:end + 1])
    except Exception:
        return _null_fields()

    return {
        "vendor": obj.get("vendor", None),
        "date": obj.get("date", None),
        "invoice_number": obj.get("invoice_number", None),
        "total": obj.get("total", None),
    }


def _build_extraction_prompt(top_text: str, bottom_text: str) -> str:
    return f"""
  You are a receipt information extraction system.

Extract fields from OCR text that may contain Arabic and English.

IMPORTANT:
The input is one continuous OCR string. Treat the first part of the string as the beginning/header section, and the last part as the ending/payment section.

STRICT RULES:
- Return JSON ONLY.
- Do not explain anything.
- Do not guess.
- If a field is unclear or not explicitly present, return null.
- OCR may contain spelling mistakes, broken words, Arabic text, English text, and mixed digits.
- Interpret Arabic digits ٠١٢٣٤٥٦٧٨٩ as 0123456789.

FIELD RULES:

vendor:
- Must come only from the beginning/header part of the string.
- Prefer store/company/brand names.
- Do not use item names as vendor.
- Do not use payment method, address, phone, or website as vendor.

date:
- May appear anywhere in the string.
- Prefer dates from the beginning/header section.
- Convert to YYYY-MM-DD.
- If the year is missing or unclear, return null.
- Do not confuse time with date.

invoice_number:
- Extract only if clearly labeled as invoice, receipt, check, bill, order, فاتورة, رقم فاتورة, رقم الإيصال, or similar.
- Do not use phone numbers, VAT numbers, CR numbers, or payment card numbers.

total:
- Must come only from the ending/payment section of the string.
- Must be explicitly present in the text.
- Prefer values near keywords such as:
  Total, Grand Total, Amount Due, Net Total, Balance Due, الإجمالي, المجموع, المبلغ المستحق, الصافي, MADA, VISA, CASH.
- Do not calculate the total from item prices.
- Do not infer a total if it is not explicitly written.
- If multiple totals exist, choose the final payable amount.
- If subtotal/VAT are present and clearly reliable, the total must be consistent with them.
- If OCR makes subtotal, VAT, or item prices unreliable, rely only on a clearly labeled final total.
- Ignore tax-only values such as VAT, Tax, ضريبة unless they are clearly the final payable amount.

Return exactly this JSON schema:
{{
  "vendor": string or null,
  "date": "YYYY-MM-DD" or null,
  "invoice_number": string or null,
  "total": number or null
}}



    TOP_SECTION:
    {top_text}

    BOTTOM_SECTION:
    {bottom_text}
    """


def _estimate_tokens(text: str) -> int:
    # Lightweight approximation for budgeting without external tokenizers.
    return max(1, len(text) // 4)


def _new_groq_rate_state() -> dict:
    return {
        "day": date.today().isoformat(),
        "day_requests": 0,
        "day_tokens": 0,
        "minute_requests": deque(),
        "minute_tokens": deque(),
    }


GROQ_RATE_STATE = _new_groq_rate_state()


def _reset_groq_day_if_needed(state: dict) -> None:
    today = date.today().isoformat()
    if state["day"] != today:
        state["day"] = today
        state["day_requests"] = 0
        state["day_tokens"] = 0


def _prune_groq_minute_windows(state: dict, now: float) -> None:
    minute_requests: Deque[float] = state["minute_requests"]
    minute_tokens: Deque[Tuple[float, int]] = state["minute_tokens"]

    while minute_requests and now - minute_requests[0] >= 60:
        minute_requests.popleft()
    while minute_tokens and now - minute_tokens[0][0] >= 60:
        minute_tokens.popleft()


def _groq_minute_tokens_used(state: dict) -> int:
    return int(sum(tokens for _, tokens in state["minute_tokens"]))


def _enforce_groq_limits_before_request(estimated_prompt_tokens: int) -> None:
    while True:
        now = time.time()
        _reset_groq_day_if_needed(GROQ_RATE_STATE)
        _prune_groq_minute_windows(GROQ_RATE_STATE, now)

        day_requests = GROQ_RATE_STATE["day_requests"]
        day_tokens = GROQ_RATE_STATE["day_tokens"]
        minute_requests = len(GROQ_RATE_STATE["minute_requests"])
        minute_tokens = _groq_minute_tokens_used(GROQ_RATE_STATE)

        planned_tokens = estimated_prompt_tokens + GROQ_MAX_OUTPUT_TOKENS

        if day_requests >= GROQ_LIMIT_RPD:
            raise RuntimeError("Groq daily request limit reached (RPD).")
        if day_tokens + planned_tokens > GROQ_LIMIT_TPD:
            raise RuntimeError("Groq daily token limit reached (TPD).")

        requests_ok = minute_requests < GROQ_LIMIT_RPM
        tokens_ok = minute_tokens + planned_tokens <= GROQ_LIMIT_TPM

        if requests_ok and tokens_ok:
            return

        waits: List[float] = []
        if not requests_ok and GROQ_RATE_STATE["minute_requests"]:
            oldest_req = GROQ_RATE_STATE["minute_requests"][0]
            waits.append(max(0.05, 60 - (now - oldest_req) + 0.05))

        if not tokens_ok and GROQ_RATE_STATE["minute_tokens"]:
            oldest_tok_ts = GROQ_RATE_STATE["minute_tokens"][0][0]
            waits.append(max(0.05, 60 - (now - oldest_tok_ts) + 0.05))

        sleep_for = min(waits) if waits else 1.0
        print(f"[INFO] Waiting {sleep_for:.2f}s to respect Groq rate limits...")
        time.sleep(sleep_for)


def _register_groq_request() -> None:
    now = time.time()
    _reset_groq_day_if_needed(GROQ_RATE_STATE)
    _prune_groq_minute_windows(GROQ_RATE_STATE, now)
    GROQ_RATE_STATE["minute_requests"].append(now)
    GROQ_RATE_STATE["day_requests"] += 1


def _register_groq_tokens(total_tokens: int) -> None:
    now = time.time()
    _reset_groq_day_if_needed(GROQ_RATE_STATE)
    _prune_groq_minute_windows(GROQ_RATE_STATE, now)
    GROQ_RATE_STATE["minute_tokens"].append((now, int(total_tokens)))
    GROQ_RATE_STATE["day_tokens"] += int(total_tokens)


def _print_groq_usage_status() -> None:
    now = time.time()
    _reset_groq_day_if_needed(GROQ_RATE_STATE)
    _prune_groq_minute_windows(GROQ_RATE_STATE, now)

    minute_requests = len(GROQ_RATE_STATE["minute_requests"])
    minute_tokens = _groq_minute_tokens_used(GROQ_RATE_STATE)
    day_requests = GROQ_RATE_STATE["day_requests"]
    day_tokens = GROQ_RATE_STATE["day_tokens"]

    print(
        "[RATE] Groq usage "
        f"RPM {minute_requests}/{GROQ_LIMIT_RPM} | "
        f"TPM {minute_tokens}/{GROQ_LIMIT_TPM} | "
        f"RPD {day_requests}/{GROQ_LIMIT_RPD} | "
        f"TPD {day_tokens}/{GROQ_LIMIT_TPD}"
    )


def llm_extract_fields_ollama(top_text: str, bottom_text: str) -> dict:
    # Normalize digits helps Arabic OCR a bit
    top_text = strip_bidi_marks(normalize_digits(top_text))
    bottom_text = strip_bidi_marks(normalize_digits(bottom_text))

    prompt = _build_extraction_prompt(top_text, bottom_text)

    r = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        },
        timeout=LLM_TIMEOUT
    )
    r.raise_for_status()
    raw = r.json().get("response", "").strip()
    return _extract_json_fields(raw)


def llm_extract_fields_groq(top_text: str, bottom_text: str) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is empty. Set it in your environment first.")

    top_text = strip_bidi_marks(normalize_digits(top_text))
    bottom_text = strip_bidi_marks(normalize_digits(bottom_text))
    prompt = _build_extraction_prompt(top_text, bottom_text)

    estimated_prompt_tokens = _estimate_tokens(prompt)
    _enforce_groq_limits_before_request(estimated_prompt_tokens)
    _register_groq_request()

    r = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "temperature": 0,
            "max_tokens": GROQ_MAX_OUTPUT_TOKENS,
            "messages": [
                {
                    "role": "system",
                    "content": "Extract structured receipt fields and return JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=LLM_TIMEOUT,
    )
    r.raise_for_status()
    body = r.json()
    raw = (
        body
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )

    usage = body.get("usage", {})
    total_tokens = int(usage.get("total_tokens") or 0)
    if total_tokens <= 0:
        total_tokens = estimated_prompt_tokens + _estimate_tokens(raw)
    _register_groq_tokens(total_tokens)
    _print_groq_usage_status()

    return _extract_json_fields(raw)


def llm_extract_fields(top_text: str, bottom_text: str) -> dict:
    provider = LLM_PROVIDER.strip().lower()
    if provider == "groq":
        return llm_extract_fields_groq(top_text=top_text, bottom_text=bottom_text)
    if provider == "ollama":
        return llm_extract_fields_ollama(top_text=top_text, bottom_text=bottom_text)
    raise ValueError(f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}. Use 'groq' or 'ollama'.")


# -------------------- Main loop --------------------

def is_text_file(p: Path) -> bool:
    return p.suffix.lower() == ".txt"


def main():
    if not INPUT_TXT_DIR.exists():
        raise SystemExit(f"INPUT_TXT_DIR not found: {INPUT_TXT_DIR.resolve()}")
    if LLM_PROVIDER.strip().lower() == "groq" and not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is empty. Set it in your environment and rerun.")

    text_paths = sorted([p for p in INPUT_TXT_DIR.iterdir() if p.is_file() and is_text_file(p)])

    results = []
    for p in text_paths:
        try:
            raw_text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            print(f"[WARN] Could not read text file: {p} ({exc})")
            continue

        if not raw_text.strip():
            print(f"[WARN] Empty text file: {p}")
            continue

        # Many OCR dumps are pipe-separated; convert pipes to line breaks first.
        normalized_text = raw_text.replace("|", "\n")
        lines = clean_lines(normalized_text)
        top_text = "\n".join(lines[:25])      # vendor, header info
        bottom_text = "\n".join(lines[-30:])  # totals, payment info

        try:
            llm_out = llm_extract_fields(top_text=top_text, bottom_text=bottom_text)
        except Exception as exc:
            print(f"[WARN] LLM extraction failed for {p.name}: {exc}")
            llm_out = _null_fields()

        vendor = llm_out["vendor"]
        date_iso = llm_out["date"]
        invoice_no = llm_out["invoice_number"]
        total = llm_out["total"]

        results.append({
            "file": p.name,
            #"text_path": str(p),
            "vendor": vendor,
            "date": date_iso,
            "invoice_number": invoice_no,
            "total": total,
        })

        print(f"[OK] {p.name} | vendor={bool(vendor), vendor} date={bool(date_iso), date_iso} inv={bool(invoice_no), invoice_no} total={total}")
        time.sleep(1)

    # Ordered JSON (already ordered by filename)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved JSON: {OUT_JSON.resolve()}")
    print(f"Processed {len(results)} text files.")


if __name__ == "__main__":
    main()

