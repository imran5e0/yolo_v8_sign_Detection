import os
import io
import json
import logging
from pathlib import Path

import fitz                        # PyMuPDF
import pytesseract
import numpy as np
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO

# —— CONFIG —— #
MODEL_PATH    = "runs/signature_detect/train2/weights/best.pt"
INPUT_PDF     = "./testing/test1.pdf"
DEBUG_DIR     = Path("debug")
OUTPUT_DIR    = Path("output")
OCR_CONF      = 60       # only mask words with conf ≥ this
AREA_THRESH   = 500      # minimum ink‐blob area to consider
CONF_THRESH   = 0.25     # YOLO confidence threshold

DEBUG_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# —— HELPERS —— #
def pdf_to_pil_pages(pdf_path):
    """Load PDF → [PIL page images]."""
    data = open(pdf_path, "rb").read()
    doc  = fitz.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        mat = fitz.Matrix(2,2)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(img)
    return pages

def mask_printed_text(img: Image.Image):
    """White-out high-confidence OCR text."""
    logger.info(" Masking printed text…")
    gray = img.convert("L")
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    draw = ImageDraw.Draw(img)
    for i, conf in enumerate(data["conf"]):
        try:
            if int(conf) >= OCR_CONF:
                x,y,w,h = (data["left"][i], data["top"][i],
                           data["width"][i], data["height"][i])
                draw.rectangle([x,y,x+w,y+h], fill="white")
        except ValueError:
            continue
    return img

def propose_regions(img: Image.Image):
    """Find ink-dense blobs via threshold+contours → [x1,y1,x2,y2]."""
    logger.info(" Proposing ink‐dense regions…")
    gray = np.array(img.convert("L"))
    # threshold: ink (dark) → white on black background
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # find blobs
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if w*h >= AREA_THRESH:
            boxes.append((x,y,x+w,y+h))
    return boxes

def detect_in_regions(pages):
    """For each page and each proposed region, run YOLO to confirm signature."""
    model = YOLO(MODEL_PATH)
    all_dets = []
    for pi, page in enumerate(pages, start=1):
        logger.info(f" Page {pi}: running region proposals")
        dbg = page.copy()
        draw_dbg = ImageDraw.Draw(dbg)
        regions = propose_regions(page)
        sig_count = 0

        for (x1,y1,x2,y2) in regions:
            # crop and run YOLO
            roi = page.crop((x1,y1,x2,y2))
            res = model(roi, conf=CONF_THRESH)[0]
            # if any detection of class 'signature'
            for box in res.boxes:
                cls_id = int(box.cls.cpu())
                if model.names[cls_id] != "signature": 
                    continue
                # rescale box coords from ROI → page
                bx1,by1,bx2,by2 = box.xyxy[0].tolist()
                abs_box = (x1+bx1, y1+by1, x1+bx2, y1+by2)
                # crop final
                final_crop = page.crop(abs_box)
                out_name = f"page{pi}_sig{sig_count}.png"
                out_path = OUTPUT_DIR/out_name
                final_crop.save(out_path)
                all_dets.append({
                    "page": pi,
                    "box": list(map(float, abs_box)),
                    "file": str(out_path)
                })
                # draw final box in green
                draw_dbg.rectangle(abs_box, outline="green", width=3)
                sig_count += 1

        # draw all proposals in red
        for (x1,y1,x2,y2) in regions:
            draw_dbg.rectangle((x1,y1,x2,y2), outline="red", width=1)
        dbg.save(DEBUG_DIR/f"debug_page{pi}.png")
        logger.info(f" Saved debug image: debug_page{pi}.png  ({sig_count} signatures)")

    # write summary
    with open(OUTPUT_DIR/"detections.json", "w") as f:
        json.dump(all_dets, f, indent=2)
    logger.info(f" Wrote summary → {OUTPUT_DIR/'detections.json'}")
    return all_dets

if __name__ == "__main__":
    logger.info(f" Loading PDF → {INPUT_PDF}")
    pages = pdf_to_pil_pages(INPUT_PDF)
    logger.info(f" PDF has {len(pages)} pages")

    masked = []
    for i, pg in enumerate(pages, start=1):
        m = mask_printed_text(pg.copy())
        m.save(DEBUG_DIR/f"masked_page{i}.png")
        masked.append(m)

    detections = detect_in_regions(masked)
    logger.info(f" Done: {len(detections)} signatures extracted.")
