from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List, Tuple, Any

import base64, io, json, logging
import numpy as np
import librosa
import soundfile as sf
import joblib
from huggingface_hub import hf_hub_download

# --- For image prediction ---
import cv2
from PIL import Image, ImageOps

try:
    from keras.models import load_model as keras_load_model
except Exception:
    keras_load_model = None

import torch
import torchvision.transforms as transforms
import timm


# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("age-backend")

# ----------------------------------------------------
# FastAPI app
# ----------------------------------------------------
app = FastAPI(title="Age Prediction API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== VOICE MODEL PART =====================
REPO_ID = "sai9390/voice-age-band-svm"
MODEL_FILENAME = "model.joblib"
LABELMAP_FILENAME = "label_map.json"

SR = 16000
CLIP_SEC = 3
N_MFCC = 40

clf = None
label_names = None

class AudioRequest(BaseModel):
    audio: str  # base64 or data URL string

class VoiceResponse(BaseModel):
    age_prediction: str
    probs: Dict[str, float]
    message: Optional[str] = "Voice prediction successful"
    warnings: Optional[List[str]] = None


def _download_assets():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    label_path = hf_hub_download(repo_id=REPO_ID, filename=LABELMAP_FILENAME)
    return model_path, label_path

def _load_model():
    global clf, label_names
    model_path, label_path = _download_assets()
    clf = joblib.load(model_path)
    with open(label_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    label_names = label_map.get("label_names") or label_map.get("labels")
    if not isinstance(label_names, list) or not label_names:
        raise RuntimeError("Invalid label map.")

def _ensure_loaded():
    if clf is None or label_names is None:
        _load_model()

def _decode_audio_with_soundfile(b64_or_dataurl: str) -> np.ndarray:
    if not b64_or_dataurl:
        raise ValueError("Empty audio payload.")
    if b64_or_dataurl.startswith("data:audio"):
        b64_or_dataurl = b64_or_dataurl.split(",", 1)[1]

    audio_bytes = base64.b64decode(b64_or_dataurl)
    if len(audio_bytes) < 100:
        raise ValueError("Audio payload too small.")

    with io.BytesIO(audio_bytes) as bio:
        y, sr = sf.read(bio, dtype="float32")

    if y.size == 0:
        raise ValueError("Decoded audio has zero samples.")

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    return y.astype(np.float32)

def _pad_or_trim(y: np.ndarray, n: int) -> np.ndarray:
    return np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]

def _extract_features(y: np.ndarray) -> np.ndarray:
    y = _pad_or_trim(y, SR * CLIP_SEC)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    M = np.concatenate([mfcc, d1, d2], axis=0)
    feat = np.concatenate([M.mean(axis=1), M.std(axis=1)], axis=0)
    return feat.astype(np.float32)

def _analyze_signal(y: np.ndarray, sr: int = SR) -> List[str]:
    warnings: List[str] = []
    if y.size == 0:
        warnings.append("No audio been found.")
        return warnings

    max_abs = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    rms_db = 20.0 * np.log10(rms)
    intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
    voiced_dur = float(np.sum([(e - s) for s, e in intervals]) / sr)

    try:
        zcr = float(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256).mean())
    except Exception:
        zcr = 0.0
    try:
        flat = float(librosa.feature.spectral_flatness(y=y).mean())
    except Exception:
        flat = 0.0

    if max_abs < 1e-4 or voiced_dur < 0.2:
        warnings.append("No audio been found.")
    if rms_db < -38.0:
        warnings.append("Voice is too low.")
    clip_ratio = float(np.mean(np.abs(y) > 0.98))
    if rms_db > -8.0 or clip_ratio > 0.01:
        warnings.append("Voice is too high / clipping.")
    if len(intervals) > 6 or (zcr > 0.15 and flat > 0.30):
        warnings.append("Voice is too noisy (multiple speakers or background noise).")

    uniq = []
    seen = set()
    for w in warnings:
        if w not in seen:
            uniq.append(w)
            seen.add(w)
    return uniq

def _predict_from_array(y: np.ndarray) -> VoiceResponse:
    x = _extract_features(y)[None, ...]
    proba = clf.predict_proba(x)[0]
    idx = int(np.argmax(proba))
    probs = {label_names[i]: float(proba[i]) for i in range(len(label_names))}
    warnings = _analyze_signal(y)

    return VoiceResponse(
        age_prediction=label_names[idx],
        probs=probs,
        message="Voice prediction successful",
        warnings=warnings if warnings else None
    )

@app.post("/predict/voice", response_model=VoiceResponse)
def predict_voice(req: AudioRequest):
    _ensure_loaded()
    if not req.audio:
        raise HTTPException(status_code=400, detail="No audio been found.")
    try:
        y = _decode_audio_with_soundfile(req.audio)
        max_abs = float(np.max(np.abs(y))) if y.size else 0.0
        intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
        voiced_dur = float(np.sum([(e - s) for s, e in intervals]) / SR)
        if max_abs < 1e-4 or voiced_dur < 0.2:
            raise HTTPException(status_code=400, detail="No audio been found.")
        return _predict_from_array(y)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


# ===================== IMAGE MODEL PART =====================
class ImageInput(BaseModel):
    image: str

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

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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

def quality_check(bgr_img: np.ndarray, face_box: Tuple[int, int, int, int]) -> Tuple[str, str]:
    H, W = bgr_img.shape[:2]
    x, y, w, h = face_box
    gray_face = cv2.cvtColor(bgr_img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    mean_brightness = float(np.mean(gray_face))
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


# ===================== HEALTH ROUTES =====================
@app.get("/")
def root():
    return {"message": "Age Prediction API running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "voice_model_loaded": clf is not None,
        "voice_labels": label_names,
        "keras_model_loaded": KERAS_MODEL is not None,
        "pytorch_model_loaded": PYTORCH_MODEL is not None,
    }

# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
