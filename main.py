from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import tempfile
import shutil
import os

app = FastAPI()

MODEL_PATH = "best1.pt"
model = YOLO(MODEL_PATH)

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    # 업로드 파일을 임시로 저장
    suffix = os.path.splitext(image.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(image.file, tmp)
        tmp_path = tmp.name

    try:
        results = model(tmp_path)
        r = results[0]

        # 예시: 박스/클래스 카운트 요약
        names = r.names  # class_id -> name
        summary = {}

        boxes = []

        if r.boxes is not None:
            for i, box in enumerate(r.boxes.xyxy.tolist()):
                cls_id = int(r.boxes.cls[i])
                conf = float(r.boxes.conf[i])

                boxes.append({
                    "id": i,
                    "label": r.names[cls_id],  # 예: "pipe"
                    "confidence": conf,
                    "bbox": {
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                    },
                    "blur": True  # 프론트에서 블러 처리용
                })

        return {
            "summary": summary,
            "num_detections": len(boxes),
            "boxes": boxes
        }
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass