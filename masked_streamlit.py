import os
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from signature_utils import load_image_or_pdf, mask_printed_text

# CONFIG
MODEL_PATH  = "runs/signature_detect/exp1/weights/best.pt"  # your trained model
OUTPUT_DIR  = "sign_crops"
CONF_THRESH = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# FASTAPI SETUP
app = FastAPI(title="Signature Extraction API")
model = YOLO(MODEL_PATH)

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    # 1) read & vector→pages
    if file.content_type not in ("application/pdf","image/png","image/jpeg"):
        raise HTTPException(400, "Only PDF/PNG/JPEG accepted")
    img_list = load_image_or_pdf(await file.read())
    detections = []

    # 2) per page
    for pi, page in enumerate(img_list, start=1):
        # mask printed text
        clean = mask_printed_text(page)
        # detect
        res = model(clean, conf=CONF_THRESH)[0]
        for di, box in enumerate(res.boxes):
            cls = int(box.cls.cpu())
            if model.names[cls] != "signature":
                continue
            x1,y1,x2,y2 = map(float, box.xyxy[0].tolist())
            crop = clean.crop((x1,y1,x2,y2))
            fname = f"page{pi}_sig{di}.png"
            outp = os.path.join(OUTPUT_DIR, fname)
            crop.save(outp)

            detections.append({
                "page": pi,
                "box":  [x1, y1, x2, y2],
                "file": outp
            })

    return JSONResponse({"signatures": detections})
