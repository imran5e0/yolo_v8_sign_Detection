import io
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import fitz      # PyMuPDF
import pytesseract

def pdf_bytes_to_pil(data: bytes, zoom: float = 2.0):
    """Convert PDF bytes → list of PIL Images."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    mat = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        pages.append(img)
    return pages

def mask_printed_text(pil_img: Image.Image, inflate: int = 2):
    """
    Use Tesseract OCR to find printed text and paint over it white.
    inflate: how many px to pad each box by.
    """
    # run OCR
    data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    img = np.array(pil_img)
    for x,y,w,h in zip(data['left'], data['top'], data['width'], data['height']):
        x0 = max(0, x-inflate)
        y0 = max(0, y-inflate)
        x1 = min(img.shape[1], x+w+inflate)
        y1 = min(img.shape[0], y+h+inflate)
        cv2.rectangle(img, (x0,y0), (x1,y1), (255,255,255), -1)
    return Image.fromarray(img)

def load_image_or_pdf(path: str):
    """Read file on disk, return list of PIL pages/images."""
    with open(path, 'rb') as f:
        data = f.read()
    # try image
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return [img]
    except UnidentifiedImageError:
        return pdf_bytes_to_pil(data)
