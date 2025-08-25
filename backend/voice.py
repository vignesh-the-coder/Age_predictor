# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import Dict, Optional, List
# import base64, io, json, logging

# import numpy as np
# import librosa
# import soundfile as sf
# import joblib
# from huggingface_hub import hf_hub_download

# # ---------------------------
# # App & CORS
# # ---------------------------
# app = FastAPI(title="Voice Age Prediction API", version="1.1.0")
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # tighten for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# logger = logging.getLogger("uvicorn")
# logger.setLevel(logging.INFO)

# # ---------------------------
# # Config
# # ---------------------------
# REPO_ID = "sai9390/voice-age-band-svm"
# MODEL_FILENAME = "model.joblib"
# LABELMAP_FILENAME = "label_map.json"

# SR = 16000
# CLIP_SEC = 3
# N_MFCC = 40

# clf = None
# label_names = None

# # ---------------------------
# # Schemas
# # ---------------------------
# class AudioRequest(BaseModel):
#   audio: str  # base64 or data URL string

# class VoiceResponse(BaseModel):
#   age_prediction: str
#   probs: Dict[str, float]
#   message: Optional[str] = "Voice prediction successful"
#   warnings: Optional[List[str]] = None

# # ---------------------------
# # Model loading
# # ---------------------------
# def _download_assets():
#   model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
#   label_path = hf_hub_download(repo_id=REPO_ID, filename=LABELMAP_FILENAME)
#   return model_path, label_path

# def _load_model():
#   global clf, label_names
#   model_path, label_path = _download_assets()
#   clf = joblib.load(model_path)
#   with open(label_path, "r", encoding="utf-8") as f:
#     label_map = json.load(f)
#   label_names = label_map.get("label_names") or label_map.get("labels")
#   if not isinstance(label_names, list) or not label_names:
#     raise RuntimeError("Invalid label map.")

# def _ensure_loaded():
#   if clf is None or label_names is None:
#     _load_model()

# # ---------------------------
# # Audio utils
# # ---------------------------
# def _decode_audio_with_soundfile(b64_or_dataurl: str) -> np.ndarray:
#   """Decode base64 WAV/OGG/FLAC/AIFF into float32 mono @16k."""
#   if not b64_or_dataurl:
#     raise ValueError("Empty audio payload.")

#   if b64_or_dataurl.startswith("data:audio"):
#     b64_or_dataurl = b64_or_dataurl.split(",", 1)[1]

#   audio_bytes = base64.b64decode(b64_or_dataurl)
#   if len(audio_bytes) < 100:
#     raise ValueError("Audio payload too small.")

#   with io.BytesIO(audio_bytes) as bio:
#     y, sr = sf.read(bio, dtype="float32")

#   if y.size == 0:
#     raise ValueError("Decoded audio has zero samples.")

#   # if stereo → mono
#   if y.ndim > 1:
#     y = np.mean(y, axis=1)

#   # resample to 16k if needed
#   if sr != SR:
#     y = librosa.resample(y, orig_sr=sr, target_sr=SR)

#   return y.astype(np.float32)

# def _pad_or_trim(y: np.ndarray, n: int) -> np.ndarray:
#   return np.pad(y, (0, n - len(y))) if len(y) < n else y[:n]

# def _extract_features(y: np.ndarray) -> np.ndarray:
#   y = _pad_or_trim(y, SR * CLIP_SEC)
#   mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
#   d1 = librosa.feature.delta(mfcc, order=1)
#   d2 = librosa.feature.delta(mfcc, order=2)
#   M = np.concatenate([mfcc, d1, d2], axis=0)
#   feat = np.concatenate([M.mean(axis=1), M.std(axis=1)], axis=0)
#   return feat.astype(np.float32)

# # ---------------------------
# # Signal quality checks → warnings
# # ---------------------------
# def _analyze_signal(y: np.ndarray, sr: int = SR) -> List[str]:
#   warnings: List[str] = []

#   if y.size == 0:
#     warnings.append("No audio been found.")
#     return warnings

#   max_abs = float(np.max(np.abs(y)))
#   rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
#   rms_db = 20.0 * np.log10(rms)

#   # Voice activity / silence check
#   intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
#   voiced_dur = float(np.sum([(e - s) for s, e in intervals]) / sr)

#   # Zero-crossing rate & spectral flatness help flag noise / clumsy
#   try:
#     zcr = float(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256).mean())
#   except Exception:
#     zcr = 0.0
#   try:
#     flat = float(librosa.feature.spectral_flatness(y=y).mean())
#   except Exception:
#     flat = 0.0

#   # 1) No audio (hard fail handled upstream)
#   if max_abs < 1e-4 or voiced_dur < 0.2:
#     # We'll raise HTTP 400 in the endpoint for "no audio"; keep for completeness here.
#     warnings.append("No audio been found.")

#   # 2) Too low volume
#   if rms_db < -38.0:
#     warnings.append("Voice is too low.")

#   # 3) Too high / clipping
#   clip_ratio = float(np.mean(np.abs(y) > 0.98))
#   if rms_db > -8.0 or clip_ratio > 0.01:
#     warnings.append("Voice is too high / clipping.")

#   # 4) Noisy / multi-speaker / clumsy
#   # Heuristics: many disjoint voiced segments, high ZCR & high flatness
#   if len(intervals) > 6 or (zcr > 0.15 and flat > 0.30):
#     warnings.append("Voice is too noisy (multiple speakers or background noise).")

#   # Deduplicate while preserving order
#   seen = set()
#   uniq = []
#   for w in warnings:
#     if w not in seen:
#       uniq.append(w); seen.add(w)
#   return uniq

# # ---------------------------
# # Predict
# # ---------------------------
# def _predict_from_array(y: np.ndarray) -> VoiceResponse:
#   x = _extract_features(y)[None, ...]
#   proba = clf.predict_proba(x)[0]
#   idx = int(np.argmax(proba))
#   probs = {label_names[i]: float(proba[i]) for i in range(len(label_names))}
#   warnings = _analyze_signal(y)

#   return VoiceResponse(
#     age_prediction=label_names[idx],
#     probs=probs,
#     message="Voice prediction successful",
#     warnings=warnings if warnings else None
#   )

# # ---------------------------
# # Endpoints
# # ---------------------------
# @app.post("/predict/voice", response_model=VoiceResponse)
# def predict_voice(req: AudioRequest):
#   _ensure_loaded()
#   if not req.audio:
#     raise HTTPException(status_code=400, detail="No audio been found.")

#   try:
#     y = _decode_audio_with_soundfile(req.audio)

#     # Hard "no audio" condition → 400
#     max_abs = float(np.max(np.abs(y))) if y.size else 0.0
#     intervals = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=512)
#     voiced_dur = float(np.sum([(e - s) for s, e in intervals]) / SR)
#     if max_abs < 1e-4 or voiced_dur < 0.2:
#       raise HTTPException(status_code=400, detail="No audio been found.")

#     return _predict_from_array(y)

#   except HTTPException:
#     raise
#   except Exception as e:
#     raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# @app.get("/")
# def root():
#   return {"message": "Voice Age Prediction API running"}

# @app.get("/health")
# def health():
#   return {"status": "ok", "model_loaded": clf is not None, "labels": label_names}
# #uvicorn voice:app --host 0.0.0.0 --port 8000