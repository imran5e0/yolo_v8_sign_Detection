import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import fitz  # PyMuPDF
import logging

# ——— CONFIG ———
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "runs/detect/train/weights/best.pt")  # your fine-tuned signature model
OUTPUT_DIR  = os.path.join(BASE_DIR, "signatures")
CONF_THRESH = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ——— LOGGING ———
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signature-extractor")

# ——— APP & MODEL INIT ———
app   = FastAPI(title="Signature Extraction Service")
model = YOLO(MODEL_PATH)

def pdf_bytes_to_pil(data: bytes):
    """Convert PDF bytes → list of PIL Images via PyMuPDF."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        mat = fitz.Matrix(2, 2)  # 2× resolution
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(img)
    return pages

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # 1) Read bytes
    data = await file.read()

    # 2) Try as image first, else PDF
    pages = []
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        pages = [img]
        logger.info("Decoded upload as image.")
    except UnidentifiedImageError:
        try:
            pages = pdf_bytes_to_pil(data)
            logger.info(f"Rendered PDF into {len(pages)} pages.")
        except Exception as e:
            raise HTTPException(400, f"Failed to parse PDF: {e}")

    detections = []

    # 3) Run inference & crop
    for page_idx, page_img in enumerate(pages, start=1):
        results = model(page_img, conf=CONF_THRESH)[0]
        logger.info(f"Page {page_idx}: {len(results.boxes)} total boxes")

        for det_idx, box in enumerate(results.boxes):
            cls_id = int(box.cls.cpu())
            name   = model.names.get(cls_id, str(cls_id))
            if name != "signature":
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            crop = page_img.crop((x1, y1, x2, y2))

            base     = f"page{page_idx}_sig{det_idx}"
            png_path = os.path.join(OUTPUT_DIR, base + ".png")
        

            crop.save(png_path)
            
            logger.info(f"Saved signature crop: {png_path}")

            detections.append({
                "page": page_idx,
                "box":  [x1, y1, x2, y2],
                "png":  png_path,
            })

    return JSONResponse({"signatures": detections})
