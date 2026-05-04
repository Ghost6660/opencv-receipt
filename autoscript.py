import os
import cv2
import natsort

from preprocessing import pick_best


# -------- CONFIG --------
INPUT_DIR = "images/cckfupm_receipts"   # folder with ~30 invoices
OUTPUT_ROOT = "CCKFUPM_RECEIPTS_OCR_TEXT"                  # single output folder (txt only)
PSM = 6                                  # tesseract psm
DEBUG_SAVE_ALL = False                   # keep False for clean outputs
# ------------------------


def is_image(fname: str) -> bool:
    return fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    images = natsort.natsorted(f for f in os.listdir(INPUT_DIR) if is_image(f))

    if not images:
        print("No images found.")
        return

    print(f"Found {len(images)} images")

    for idx, fname in enumerate(images, start=1):
        image_path = os.path.join(INPUT_DIR, fname)
        print(f"\n[{idx}/{len(images)}] Processing {fname}")

        img = cv2.imread(image_path)
        if img is None:
            print("  ❌ Could not read image, skipping")
            continue

        best = pick_best(
            img_bgr=img,
            psm=PSM,
            debug_save_all=DEBUG_SAVE_ALL,
            outdir=OUTPUT_ROOT
        )

        if best["processed"] is None:
            print("  ❌ No preprocessing succeeded")
            continue

        # Save OCR text + metadata
        base_name = os.path.splitext(fname)[0]
        best_txt_path = os.path.join(
            OUTPUT_ROOT,
            f"{base_name}_best_{best['action_id']:02d}_{best['action_name']}.txt"
        )
        with open(best_txt_path, "w", encoding="utf-8") as f:
           # f.write(f"source_image={fname}\n")
            #f.write(f"best_action_id={best['action_id']}\n")
            #f.write(f"best_action_name={best['action_name']}\n")
            #f.write(f"score={best['score']}\n\n")
            #f.write("--- OCR TEXT ---\n")
            f.write(best["ocr"].text if best["ocr"] else "")

        print(f"  ✅ Saved {best_txt_path}")


if __name__ == "__main__":
    main()
