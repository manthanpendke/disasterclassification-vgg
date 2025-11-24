# yolo_api.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional
from pydantic import BaseModel
from ultralytics import YOLO
from PIL import Image
import io
import os
import numpy as np

# CONFIG
API_KEY = os.environ.get("YOLO_API_KEY", "change_me")  # set secure value in deployment
YOLO_MODEL_PATH = "yolov8n.pt"  # ensure present in server root

app = FastAPI(title="YOLO Detection API")

# load model once
if not os.path.exists(YOLO_MODEL_PATH):
    raise SystemExit(f"Missing {YOLO_MODEL_PATH} in server root. Upload model and restart.")
_model = YOLO(YOLO_MODEL_PATH)


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    score: float


class DetectionResponse(BaseModel):
    width: int
    height: int
    boxes: list[Box]


def _check_api_key(api_key: Optional[str]):
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.post("/detect", response_model=DetectionResponse)
async def detect_image(file: UploadFile = File(...), x_api_key: Optional[str] = Header(None)):
    _check_api_key(x_api_key)
    contents = await file.read()
    try:
        pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Run YOLO - prefer model(pil) API
    try:
        results = _model(pil, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO error: {e}")

    if not results:
        return JSONResponse(content={"width": pil.width, "height": pil.height, "boxes": []})

    r = results[0]
    boxes_out = []
    # ultralytics v8: r.boxes.xyxy, r.boxes.cls, r.boxes.conf
    boxes = getattr(r, "boxes", None)
    if boxes is None or boxes.data is None:
        return JSONResponse(content={"width": pil.width, "height": pil.height, "boxes": []})

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()
    conf = boxes.conf.cpu().numpy()

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = [float(v) for v in xyxy[i][:4]]
        boxes_out.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "class_id": int(cls[i]),
            "score": float(conf[i])
        })

    return JSONResponse(content={"width": pil.width, "height": pil.height, "boxes": boxes_out})
