import io
import base64
import logging
from typing import Optional, Tuple, Dict, Any, Literal

import numpy as np
import cv2
from PIL import Image, ImageOps

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- ML stacks ---
try:
    from keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

import torch
import torchvision.transforms as transforms
import timm
from huggingface_hub import hf_hub_download

# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("age-backend")

# ----------------------------------------------------
# FastAPI app
# ----------------------------------------------------
app = FastAPI(title="Age Predictor API (Base64 Image)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------
# Data model
# ----------------------------------------------------
class ImageInput(BaseModel):
    image: str

# ----------------------------------------------------
# Load models
# ----------------------------------------------------
KERAS_MODEL = None
PYTORCH_MODEL = None
PT_TRANSFORM = None

def load_keras_model():
    global KERAS_MODEL
    if keras_load_model is None:
        logger.info("Keras not available.")
        return
    try:
        KERAS_MODEL = keras_load_model("best_model_finetuned.h5")
        logger.info("✅ Keras model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load Keras model: {e}")
        KERAS_MODEL = None

def load_pytorch_model():
    global PYTORCH_MODEL, PT_TRANSFORM
    try:
        repo_id = "sai9390/age_25epochs"
        filename = "best_effnetv2s_20to50.pt"
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        PYTORCH_MODEL = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=1)

        checkpoint = torch.load(model_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = {k.replace("model.", "").replace("module.", ""): v
                          for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint

        PYTORCH_MODEL.load_state_dict(state_dict, strict=False)
        PYTORCH_MODEL.eval()

        PT_TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        logger.info("✅ PyTorch model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load PyTorch model: {e}")
        PYTORCH_MODEL = None
        PT_TRANSFORM = None

load_keras_model()
load_pytorch_model()

# ----------------------------------------------------
# Face detection
# ----------------------------------------------------
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def strip_base64_prefix(b64: str) -> str:
    if "," in b64 and b64.lower().startswith("data:image"):
        return b64.split(",", 1)[1]
    return b64

def decode_base64_to_pil(b64: str) -> Optional[Image.Image]:
    try:
        img_bytes = base64.b64decode(b64, validate=False)
        pil = Image.open(io.BytesIO(img_bytes))
        return ImageOps.exif_transpose(pil).convert("RGB")
    except Exception as e:
        logger.error(f"❌ Error decoding base64: {e}")
        return None

def detect_largest_face_bgr(bgr_img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return bgr_img[y:y+h, x:x+w], (int(x), int(y), int(w), int(h))

def preprocess_for_pytorch(face_bgr: np.ndarray) -> torch.Tensor:
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face)
    return PT_TRANSFORM(pil).unsqueeze(0)

def predict_age(face_bgr: np.ndarray) -> Optional[float]:
    try:
        if KERAS_MODEL is not None:
            face = cv2.resize(face_bgr, (224, 224))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = np.asarray(face, dtype=np.float32) / 255.0
            face = np.expand_dims(face, axis=0)
            pred = KERAS_MODEL.predict(face, verbose=0)
            return float(pred[0][0])
        if PYTORCH_MODEL is not None:
            with torch.no_grad():
                out = PYTORCH_MODEL(preprocess_for_pytorch(face_bgr))
                return float(out.item())
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None
    return None

def pack_face_box(box: Optional[Tuple[int, int, int, int]]) -> Optional[Dict[str, int]]:
    if not box:
        return None
    x, y, w, h = box
    return {"x": x, "y": y, "w": w, "h": h}

# ----------------------------------------------------
# Quality check (always allow prediction, just warn)
# ----------------------------------------------------
def quality_check(bgr_img: np.ndarray, face_box: Tuple[int, int, int, int]) -> Tuple[str, str]:
    H, W = bgr_img.shape[:2]
    x, y, w, h = face_box
    gray_face = cv2.cvtColor(bgr_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    lap_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray_face))

    # Default
    status, msg = "ok", "OK"

    if lap_var < 50:
        status, msg = "warn", "Face too blurry or moving"
    elif lap_var < 120:
        status, msg = "warn", "Face slightly blurry"

    if mean_brightness < 35:
        status, msg = "warn", "Image too dark"
    elif mean_brightness > 235:
        status, msg = "warn", "Image too bright"

    return status, msg

# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "keras_model_loaded": KERAS_MODEL is not None,
        "pytorch_model_loaded": PYTORCH_MODEL is not None,
    }

@app.post("/predict/image")
async def predict_image(data: ImageInput):
    if KERAS_MODEL is None and PYTORCH_MODEL is None:
        raise HTTPException(status_code=500, detail="No model loaded")

    b64 = strip_base64_prefix(data.image)
    pil_img = decode_base64_to_pil(b64)
    if pil_img is None:
        return {"error": "Invalid image"}

    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]

    face_bgr, box = detect_largest_face_bgr(bgr)
    if face_bgr is None:
        return {"error": "NO FACE DETECTED", "image_size": {"width": W, "height": H}}

    status, qmsg = quality_check(bgr, box)

    age = predict_age(face_bgr)
    if age is None:
        return {"error": "Prediction failed", "image_size": {"width": W, "height": H}}

    return {
        "age": round(age, 2),
        "quality": status,
        "message": qmsg,
        "face_box": pack_face_box(box),
        "image_size": {"width": W, "height": H},
    }


# ----------------------------------------------------
# Run:
# uvicorn main:app --host 0.0.0.0 --port 8000
# ----------------------------------------------------
