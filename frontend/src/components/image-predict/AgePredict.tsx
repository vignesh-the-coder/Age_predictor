"use client";

import React, { useEffect, useRef, useState } from "react";
import { Upload, Camera, StopCircle, PlayCircle, Loader2 } from "lucide-react";

/* -------- Types -------- */
interface RealTimeEntry {
  age: number;
  time: string;
}
interface FaceBox {
  x: number;
  y: number;
  w: number;
  h: number;
}
interface ImageSize {
  width: number;
  height: number;
}
type Quality = "ok" | "warn";

/* -------- Component -------- */
export default function AgePredictionApp() {
  const [image, setImage] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [quality, setQuality] = useState<Quality | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [realTimePredictions, setRealTimePredictions] = useState<RealTimeEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [isRealTimeMode, setIsRealTimeMode] = useState(false);

  const [box, setBox] = useState<FaceBox | null>(null);
  const [serverImgSize, setServerImgSize] = useState<ImageSize | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  /* -------- Webcam -------- */
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      streamRef.current = stream;
      setIsWebcamActive(true);
    } catch (err: any) {
      setError("Failed to access webcam: " + err.message);
    }
  };

  useEffect(() => {
    if (isWebcamActive && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(() => {});
    }
  }, [isWebcamActive]);

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsWebcamActive(false);
    stopRealTime();
  };

  /* -------- Upload -------- */
  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const base64 = await toBase64(file);
    const cleanBase64 = base64.replace(/^data:image\/[a-z]+;base64,/, "");
    setImage(base64);
    await sendToBackend(cleanBase64);
  };

  /* -------- Capture frame -------- */
  const captureImage = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.8);
    const cleanBase64 = dataUrl.split(",")[1];
    setImage(dataUrl);
    await sendToBackend(cleanBase64);
  };

  /* -------- Send to backend -------- */
  const sendToBackend = async (base64: string) => {
    setIsProcessing(true);
    setError(null);
    setPrediction(null);
    setQuality(null);
    setMessage(null);
    setBox(null);
    setServerImgSize(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64 }),
      });
      const data = await res.json();

      // Common: server also returns image_size; save for scaling
      if (data.image_size) {
        setServerImgSize(data.image_size);
      }

      if (data.error) {
        setError(data.error);
        if (data.face_box) setBox(data.face_box);
        return;
      }

      // success / warn flow
      const roundedAge = Math.round(data.age * 10) / 10;
      setPrediction(roundedAge);
      setQuality(data.quality as Quality);
      setMessage(data.message || null);
      if (data.face_box) setBox(data.face_box);

      if (isRealTimeMode) {
        const newEntry: RealTimeEntry = {
          age: roundedAge,
          time: new Date().toLocaleTimeString(),
        };
        setRealTimePredictions((prev) => [newEntry, ...prev].slice(0, 5));
      }
    } catch (err: any) {
      setError("Prediction failed: " + err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  /* -------- Real-time -------- */
  const startRealTime = () => {
    if (!isWebcamActive) return;
    setIsRealTimeMode(true);
    if (intervalRef.current) clearInterval(intervalRef.current);
    intervalRef.current = setInterval(() => {
      captureImage();
    }, 3000);
  };

  const stopRealTime = () => {
    setIsRealTimeMode(false);
    if (intervalRef.current) clearInterval(intervalRef.current);
  };

  /* -------- Helpers for box scaling -------- */
  const getScaledBoxStyle = (): React.CSSProperties | null => {
    if (!box || !imgRef.current || !serverImgSize) return null;

    // server original size (used by detector)
    const sW = serverImgSize.width;
    const sH = serverImgSize.height;

    // rendered size in the page
    const rect = imgRef.current.getBoundingClientRect();
    const rW = rect.width;
    const rH = rect.height;

    const scaleX = rW / sW;
    const scaleY = rH / sH;

    return {
      position: "absolute",
      top: box.y * scaleY,
      left: box.x * scaleX,
      width: box.w * scaleX,
      height: box.h * scaleY,
      boxSizing: "border-box",
      borderWidth: 4,
      borderStyle: "solid",
      borderColor: error
        ? "red"
        : quality === "warn"
        ? "orange"
        : "green",
      pointerEvents: "none",
    };
  };

  /* -------- UI -------- */
  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 flex flex-col items-center p-6">
      <h1 className="text-3xl font-bold mb-6 text-indigo-700">Age Prediction AI</h1>

      {/* Upload Section */}
      <div className="bg-white shadow-md rounded-xl p-6 mb-6 w-full max-w-md text-center">
        <h2 className="text-lg font-semibold mb-4">Upload Image</h2>
        <label className="flex flex-col items-center justify-center border-2 border-dashed border-indigo-300 rounded-lg p-6 cursor-pointer hover:bg-indigo-50">
          <Upload className="w-10 h-10 text-indigo-500 mb-2" />
          <span className="text-indigo-600 font-medium">Click to Upload</span>
          <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
        </label>
      </div>

      {/* Webcam Section */}
      <div className="bg-white shadow-md rounded-xl p-6 mb-6 w-full max-w-md text-center">
        <h2 className="text-lg font-semibold mb-4">Webcam</h2>
        {!isWebcamActive ? (
          <button
            onClick={startWebcam}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg flex items-center gap-2 mx-auto"
          >
            <Camera className="w-5 h-5" /> Start Webcam
          </button>
        ) : (
          <>
            <video
              ref={videoRef}
              autoPlay
              muted
              playsInline
              className="w-full rounded-lg border mb-4"
            />
            <canvas ref={canvasRef} className="hidden" />
            <div className="flex gap-2 justify-center">
              <button onClick={captureImage} className="px-4 py-2 bg-green-500 text-white rounded-lg">
                Capture
              </button>
              {!isRealTimeMode ? (
                <button
                  onClick={startRealTime}
                  className="px-4 py-2 bg-blue-500 text-white rounded-lg flex items-center gap-2"
                >
                  <PlayCircle className="w-5 h-5" /> Real-Time
                </button>
              ) : (
                <button
                  onClick={stopRealTime}
                  className="px-4 py-2 bg-red-500 text-white rounded-lg flex items-center gap-2"
                >
                  <StopCircle className="w-5 h-5" /> Stop
                </button>
              )}
              <button
                onClick={stopWebcam}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg"
              >
                Stop Webcam
              </button>
            </div>
          </>
        )}
      </div>

      {/* Results */}
      <div className="bg-white shadow-md rounded-xl p-6 w-full max-w-md text-center">
        <h2 className="text-lg font-semibold mb-4">Result</h2>
        {isProcessing && (
          <div className="flex justify-center items-center gap-2 text-indigo-600">
            <Loader2 className="w-5 h-5 animate-spin" /> Processing...
          </div>
        )}

        {prediction !== null && !isRealTimeMode && (
          <p className="text-xl font-bold text-green-600">{prediction} years old</p>
        )}

        {/* Show quality warning (still predicted) */}
        {quality === "warn" && !error && (
          <p className="mt-2 text-amber-600 font-medium">Note: {message}</p>
        )}

        {/* Show error when we didn't predict */}
        {error && <p className="text-red-500">{error}</p>}

        {isRealTimeMode && realTimePredictions.length > 0 && (
          <div className="text-left">
            <h3 className="font-medium mb-2">Recent Predictions:</h3>
            <ul className="space-y-1">
              {realTimePredictions.map((p, idx) => (
                <li key={idx} className="text-gray-700">
                  {p.age} yrs <span className="text-sm">({p.time})</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Preview with bounding box (scaled) */}
      {image && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold mb-2">Preview</h2>
          <div className="relative inline-block">
            <img
              ref={imgRef}
              src={image}
              alt="preview"
              className="w-64 rounded-lg shadow-md border"
            />
            {box && serverImgSize && (
              <div style={getScaledBoxStyle()!} />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* -------- Helper -------- */
function toBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = (error) => reject(error);
  });
}
