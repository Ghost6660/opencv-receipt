import os
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2 as cv
import numpy as np
from PIL import Image
from wia_scan import *

try:
    import pythoncom
except ImportError:
    pythoncom = None


TARGET_DEVICE_UID = r"SWD\Escl\00000000-0000-1000-8000-00115695e2ce"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
SCAN_LOCK = threading.Lock()

# ---------- filename / numbering ----------

def get_next_filename(folder: str) -> str:
    """Return next dataX filename (without extension)."""
    os.makedirs(folder, exist_ok=True)

    existing = [
        f for f in os.listdir(folder)
        if f.startswith("data") and f.lower().endswith(".png")
    ]
    nums = []
    for f in existing:
        try:
            nums.append(int(f[4:-4]))  # "data" + number + ".png"
        except:
            pass
    return f"data{max(nums) + 1}" if nums else "data1"


# ---------- document detection + warp ----------

def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts.astype("float32"))
    (tl, tr, br, bl) = rect

    # Compute width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    return cv.warpPerspective(image_bgr, M, (maxWidth, maxHeight))

def detect_document_contour(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    edges = cv.Canny(binary, 50, 150)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    doc_contour = None
    for contour in contours:
        peri = cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            doc_contour = approx
            break

    if doc_contour is None:
        print("No document detected")
        return image_bgr
    
    H, W = image_bgr.shape[:2]
    img_area = H * W

    min_doc_area = 0.05 * img_area   # receipt must be at least 5% of image
    max_doc_area = 0.90 * img_area   # but not basically the whole scan/page

    warped = four_point_transform(image_bgr.copy(), doc_contour.reshape(4, 2))
    print(f"Document detected and transformed: {warped.shape}")
    area = cv.contourArea(approx)
    print("area ratio:", cv.contourArea(approx) / (H*W), "points:", len(approx))
    if len(approx) == 4 and (min_doc_area < area < max_doc_area):
        return approx


def contour_to_4x2(doc) -> np.ndarray | None:
    """Return points as (4,2) if doc is a 4-corner contour, else None."""
    if doc is None:
        return None
    doc = np.asarray(doc)

    # OpenCV approx format: (4,1,2)
    if doc.ndim == 3 and doc.shape[0] == 4 and doc.shape[1] == 1 and doc.shape[2] == 2:
        return doc.reshape(4, 2)

    # Already in (4,2)
    if doc.ndim == 2 and doc.shape == (4, 2):
        return doc

    # Not valid
    return None


def process_receipt(pil_img: Image.Image) -> np.ndarray:
    """
    Returns processed (cropped + aligned) image in BGR (OpenCV format).
    If no document detected, returns the original.
    """
    pil_img = pil_img.convert("RGB")
    img_rgb = np.array(pil_img)
    img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)

    doc = detect_document_contour(img_bgr)   # must return approx or None
    pts = contour_to_4x2(doc)

    warped = img_bgr
    if pts is None:
        print("Not a 4-point contour; skipping warp.")
        warped = img_bgr
    else:
        warped = four_point_transform(img_bgr.copy(), pts)

    return warped


def get_local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return "127.0.0.1"


def run_scan_once() -> dict:
    com_started = False
    if pythoncom is not None:
        pythoncom.CoInitialize()
        com_started = True

    try:
        device = connect_to_device_by_uid(TARGET_DEVICE_UID, quiet=True)

        pil_scan = scan_side(device=device, quiet=True)  # Pillow Image

        out_folder = r"images\cckfupm_receipts"
        next_name = get_next_filename(out_folder)
        out_path = os.path.join(out_folder, f"{next_name}.png")

        processed_bgr = process_receipt(pil_scan)
        cv.imwrite(out_path, processed_bgr, [cv.IMWRITE_PNG_COMPRESSION, 3])
        print(f"Saved cropped+aligned image as: {out_path}")

        return {
            "ok": True,
            "saved_path": out_path,
            "device_uid": TARGET_DEVICE_UID,
        }
    finally:
        if com_started:
            pythoncom.CoUninitialize()

# ---------- scan -> process -> save ----------

class ScanRequestHandler(BaseHTTPRequestHandler):
    def _send_html(self, status_code: int, html: str) -> None:
        data = html.encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status_code: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            ip = get_local_ip()
            self._send_html(
                200,
                f"""<!doctype html>
<html>
<head>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Receipt Scanner</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; max-width: 720px; margin: 0 auto; }}
    .btn {{ display: inline-block; padding: 16px 22px; font-size: 18px; background: #111; color: #fff; text-decoration: none; border-radius: 10px; }}
    code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Receipt Scanner</h1>
  <p>Open this page from your phone on the same Wi-Fi network.</p>
  <p>Scan endpoint: <code>http://{ip}:{SERVER_PORT}/scan</code></p>
  <p><a class=\"btn\" href=\"/scan\">Scan now</a></p>
</body>
</html>""",
            )
            return

        if self.path.startswith("/scan"):
            with SCAN_LOCK:
                try:
                    result = run_scan_once()
                    self._send_json(200, result)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
            return

        self._send_html(404, "<h1>404</h1><p>Not found.</p>")

    def do_POST(self):
        if self.path == "/scan":
            with SCAN_LOCK:
                try:
                    result = run_scan_once()
                    self._send_json(200, result)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc)})
            return
        self._send_json(404, {"ok": False, "error": "Not found"})

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((SERVER_HOST, SERVER_PORT), ScanRequestHandler)
    ip = get_local_ip()
    print(f"HTTP server running on http://{ip}:{SERVER_PORT}")
    print(f"Open http://{ip}:{SERVER_PORT}/scan on your phone to scan immediately.")
    server.serve_forever()

if __name__ == "__main__":
    main()

