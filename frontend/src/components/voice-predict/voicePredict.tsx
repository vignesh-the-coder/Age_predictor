import React, { useState, useRef, useEffect } from "react";

interface VoiceResponse {
  age_prediction: string;
  probs: Record<string, number>;
  message?: string;
  warnings?: string[];
}

export default function VoicePredictionApp() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioFileName, setAudioFileName] = useState<string | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [prediction, setPrediction] = useState<VoiceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const recordingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ---- Utility: Convert any Blob the browser can decode → WAV Blob ----
 async function convertToWav(file: Blob): Promise<Blob> {
  const arrayBuffer = await file.arrayBuffer();
  const audioCtx = new AudioContext();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  const numChannels = audioBuffer.numberOfChannels;
  const sampleRate = audioBuffer.sampleRate;
  const numFrames = audioBuffer.length;

  const buffer = new ArrayBuffer(44 + numFrames * numChannels * 2);
  const view = new DataView(buffer);

  function writeString(view: DataView, offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  let offset = 0;
  writeString(view, offset, "RIFF"); offset += 4;
  view.setUint32(offset, 36 + numFrames * numChannels * 2, true); offset += 4;
  writeString(view, offset, "WAVE"); offset += 4;
  writeString(view, offset, "fmt "); offset += 4;
  view.setUint32(offset, 16, true); offset += 4;
  view.setUint16(offset, 1, true); offset += 2;
  view.setUint16(offset, numChannels, true); offset += 2;
  view.setUint32(offset, sampleRate, true); offset += 4;
  view.setUint32(offset, sampleRate * numChannels * 2, true); offset += 4;
  view.setUint16(offset, numChannels * 2, true); offset += 2;
  view.setUint16(offset, 16, true); offset += 2;
  writeString(view, offset, "data"); offset += 4;
  view.setUint32(offset, numFrames * numChannels * 2, true); offset += 4;

  for (let i = 0; i < numFrames; i++) {
    for (let ch = 0; ch < numChannels; ch++) {
      const sample = audioBuffer.getChannelData(ch)[i];
      const s = Math.max(-1, Math.min(1, sample));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      offset += 2;
    }
  }

  return new Blob([buffer], { type: "audio/wav" });
}


  // ---- Utility: Blob → Base64 (no data URL prefix) ----
  const blobToBase64 = (blob: Blob): Promise<string> =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve((reader.result as string).split(",")[1] ?? "");
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

  // ---- Upload audio file ----
  const handleAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const allowedTypes = [
      "audio/wav", "audio/mpeg", "audio/mp3", "audio/m4a",
      "audio/webm", "audio/ogg", "audio/flac",
    ];

    if (
      !allowedTypes.includes(file.type) &&
      !/\.(wav|mp3|m4a|webm|ogg|flac)$/i.test(file.name)
    ) {
      setError("Please upload a valid audio file (WAV, MP3, M4A, WebM, OGG, FLAC).");
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError("File too large. Upload < 50MB");
      return;
    }

    setAudioBlob(file);
    setAudioUrl(URL.createObjectURL(file));
    setAudioFileName(file.name);
    setRecordingDuration(0);
    setPrediction(null);
    setError(null);
  };

  // ---- Start recording ----
  const startRecording = async () => {
    try {
      if (!("MediaRecorder" in window)) {
        setError("MediaRecorder not supported in this browser.");
        return;
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorderRef.current.onstop = () => {
        const recordedBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        if (recordedBlob.size === 0) {
          setError("No audio been found. Please try again.");
          setIsRecording(false);
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        setAudioBlob(recordedBlob);
        setAudioUrl(URL.createObjectURL(recordedBlob));
        setAudioFileName(null);
        stream.getTracks().forEach((t) => t.stop());
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
      setRecordingDuration(0);

      recordingTimerRef.current = setInterval(() => {
        setRecordingDuration((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      setError("Mic access failed: " + (err?.message || String(err)));
    }
  };

  // ---- Stop recording ----
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }
    }
  };

  // ---- Clear audio ----
  const clearAudio = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setAudioFileName(null);
    setPrediction(null);
    setRecordingDuration(0);
    setError(null);
  };

  // ---- Predict Age ----
  const predictVoiceAge = async () => {
    if (!audioBlob) {
      setError("No audio been found. Please upload or record first.");
      return;
    }

    setIsProcessing(true);
    setError(null);
    setPrediction(null);

    try {
      const wavBlob = await convertToWav(audioBlob);
      if (!wavBlob || wavBlob.size === 0) {
        throw new Error("No audio been found after conversion.");
      }
      const audioBase64 = await blobToBase64(wavBlob);

      const response = await fetch("http://localhost:8000/predict/voice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ audio: audioBase64 }),
      });

      const payload = await response.json();

      if (!response.ok) {
        const msg = payload?.detail || "Server error.";
        throw new Error(msg);
      }

      const data = payload as VoiceResponse;
      setPrediction(data);
    } catch (err: any) {
      setError(err?.message ? err.message : "Prediction failed.");
    } finally {
      setIsProcessing(false);
    }
  };

  // ---- Helpers ----
  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
  };

  useEffect(() => {
    return () => {
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#fbf6ef] p-6 font-sans">
      <div className="max-w-2xl w-full bg-white shadow-lg rounded-xl p-6">
        <h1 className="text-3xl font-bold text-center text-sky-700 mb-6">
          Voice Age Prediction AI
        </h1>

        {/* Upload or Record */}
        {!audioBlob ? (
          <div className="space-y-4">
            <label className="flex flex-col items-center justify-center border-2 border-dashed border-sky-300 rounded-lg p-6 cursor-pointer hover:bg-sky-50 transition-colors">
              <span className="text-sky-600 font-medium">Click to Upload Audio</span>
              <span className="text-sm text-gray-500">WAV, MP3, M4A, WebM, OGG, FLAC (max 50MB)</span>
              <input type="file" accept="audio/*" className="hidden" onChange={handleAudioUpload} />
            </label>
            <div className="text-center">
              <div className="text-gray-500 mb-2">OR</div>
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="px-6 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600 transition"
                >
                  🎤 Start Recording
                </button>
              ) : (
                <div className="space-y-3">
                  <div className="text-red-500">⏺ Recording: {formatDuration(recordingDuration)}</div>
                  <button
                    onClick={stopRecording}
                    className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition"
                  >
                    ⏹ Stop Recording
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <p className="text-green-700 font-medium mb-2">
                ✅ {audioFileName ? "Audio Uploaded" : "Recording Complete"}
              </p>
              {audioFileName && <p className="text-sm text-gray-600">File: {audioFileName}</p>}
              {recordingDuration > 0 && (
                <p className="text-sm text-gray-600">Duration: {formatDuration(recordingDuration)}</p>
              )}
              {audioUrl && <audio controls src={audioUrl} className="w-full" />}
            </div>
            <div className="flex gap-2 justify-center">
              <button
                onClick={clearAudio}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition"
              >
                ❌ Remove
              </button>
            </div>
          </div>
        )}

        {/* Predict */}
        {audioBlob && (
          <div className="text-center mt-6">
            <button
              onClick={predictVoiceAge}
              disabled={isProcessing}
              className="px-6 py-3 bg-gradient-to-r from-sky-600 to-blue-600 text-white rounded-lg font-semibold hover:from-sky-700 hover:to-blue-700 transition disabled:opacity-50"
            >
              {isProcessing ? "Processing..." : "Predict Age"}
            </button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-4 bg-red-100 text-red-700 p-3 rounded">
            ⚠️ {error}
          </div>
        )}

        {/* Prediction */}
        {prediction && (
          <div className="mt-6 bg-sky-50 border border-sky-200 rounded-lg p-6 text-center">
            <h2 className="text-xl font-semibold text-sky-700">Predicted Age</h2>
            <p className="text-3xl font-bold text-sky-600 mt-2">{prediction.age_prediction}</p>

            {prediction.warnings && prediction.warnings.length > 0 && (
              <div className="mt-4 bg-yellow-50 border border-yellow-200 text-yellow-800 p-3 rounded text-sm text-left">
                <p className="font-semibold mb-1">Warnings:</p>
                <ul className="list-disc list-inside space-y-1">
                  {prediction.warnings.map((w, i) => (
                    <li key={i}>{w}</li>
                  ))}
                </ul>
              </div>
            )}

            <div className="mt-4">
              <h3 className="font-semibold text-gray-700">Probabilities</h3>
              <ul className="text-sm text-gray-600 mt-2 space-y-1">
                {Object.entries(prediction.probs).map(([label, prob]) => (
                  <li key={label}>
                    {label}: {(prob * 100).toFixed(2)}%
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
