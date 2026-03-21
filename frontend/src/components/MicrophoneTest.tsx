"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";

type TestState = "idle" | "requesting" | "listening" | "pass" | "fail";

export default function MicrophoneTest() {
  const [state, setState] = useState<TestState>("idle");
  const [volume, setVolume] = useState(0);
  const [peakVolume, setPeakVolume] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState("");

  const streamRef = useRef<MediaStream | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const rafRef = useRef<number | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const peakRef = useRef(0);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      cleanup();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cleanup = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close();
    }
    streamRef.current = null;
    analyserRef.current = null;
    audioCtxRef.current = null;
    rafRef.current = null;
  }, []);

  const startTest = useCallback(async () => {
    // Reset
    if (audioUrl) URL.revokeObjectURL(audioUrl);
    setAudioUrl(null);
    setVolume(0);
    setPeakVolume(0);
    peakRef.current = 0;
    setErrorMsg("");
    setState("requesting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Set up audio analysis
      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      // Set up recording
      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        const url = URL.createObjectURL(blob);
        setAudioUrl(url);
      };

      recorder.start(100);
      setState("listening");

      // Read volume levels
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const readVolume = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteTimeDomainData(dataArray);

        // Calculate RMS
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const val = (dataArray[i] - 128) / 128;
          sum += val * val;
        }
        const rms = Math.sqrt(sum / dataArray.length);
        const normalized = Math.min(1, rms * 4); // Scale up for visibility

        setVolume(normalized);
        if (normalized > peakRef.current) {
          peakRef.current = normalized;
          setPeakVolume(normalized);
        }

        rafRef.current = requestAnimationFrame(readVolume);
      };
      rafRef.current = requestAnimationFrame(readVolume);

      // Stop after 4 seconds
      setTimeout(() => {
        if (rafRef.current) cancelAnimationFrame(rafRef.current);
        if (
          recorderRef.current &&
          recorderRef.current.state !== "inactive"
        ) {
          recorderRef.current.stop();
        }

        const detected = peakRef.current > 0.05;
        setState(detected ? "pass" : "fail");
        if (!detected) {
          setErrorMsg("לא זוהה קול. בדקי שהמיקרופון מחובר ולא מושתק.");
        }

        // Stop stream
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop());
        }
        if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
          audioCtxRef.current.close();
        }
      }, 4000);
    } catch (err) {
      cleanup();
      setState("fail");
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setErrorMsg(
          "הגישה למיקרופון נחסמה. לחצי על סמל המנעול בשורת הכתובת ואפשרי גישה למיקרופון."
        );
      } else {
        setErrorMsg("לא הצלחנו לגשת למיקרופון. בדקי שמיקרופון מחובר למחשב.");
      }
    }
  }, [audioUrl, cleanup]);

  return (
    <Card>
      <div className="flex items-center gap-3 mb-3">
        <span className="text-2xl">🎤</span>
        <h3 className="font-semibold text-gray-700">בדיקת מיקרופון</h3>
        {state === "pass" && (
          <span className="text-sm bg-green-100 text-green-700 px-2 py-0.5 rounded-full font-medium">
            תקין
          </span>
        )}
        {state === "fail" && (
          <span className="text-sm bg-red-100 text-red-700 px-2 py-0.5 rounded-full font-medium">
            בעיה
          </span>
        )}
      </div>

      {/* Idle / Ready */}
      {state === "idle" && (
        <div>
          <p className="text-sm text-gray-500 mb-3">
            בדקי שהמיקרופון עובד לפני שמתחילים לתרגל
          </p>
          <Button onClick={startTest} size="sm">
            התחלת בדיקה
          </Button>
        </div>
      )}

      {/* Requesting permission */}
      {state === "requesting" && (
        <p className="text-sm text-gray-500">מבקשת גישה למיקרופון...</p>
      )}

      {/* Listening — show volume meter */}
      {state === "listening" && (
        <div className="space-y-3">
          <p className="text-sm text-blue-600 font-medium animate-pulse">
            מקשיבה... דברי משהו!
          </p>
          {/* Volume bar */}
          <div className="w-full bg-gray-100 rounded-full h-5 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-75 bg-gradient-to-l from-blue-500 to-blue-300"
              style={{ width: `${Math.max(2, volume * 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-400">
            <span>שקט</span>
            <span>חזק</span>
          </div>
        </div>
      )}

      {/* Pass */}
      {state === "pass" && (
        <div className="space-y-3">
          <p className="text-sm text-green-700">
            המיקרופון עובד מצוין! האזיני להקלטה:
          </p>
          {audioUrl && (
            <audio controls src={audioUrl} className="w-full" />
          )}
          <Button onClick={startTest} variant="secondary" size="sm">
            בדיקה חוזרת
          </Button>
        </div>
      )}

      {/* Fail */}
      {state === "fail" && (
        <div className="space-y-3">
          <p className="text-sm text-red-600">{errorMsg}</p>
          <div className="bg-red-50 rounded-lg p-3 text-sm text-gray-600 space-y-1">
            <p className="font-medium text-gray-700">טיפים לפתרון:</p>
            <ul className="list-disc mr-4 space-y-1">
              <li>בדקי שהמיקרופון מחובר למחשב</li>
              <li>בדקי שהמיקרופון לא מושתק</li>
              <li>לחצי על סמל המנעול ליד שורת הכתובת ואפשרי גישה למיקרופון</li>
              <li>נסי לרענן את הדף</li>
            </ul>
          </div>
          <Button onClick={startTest} size="sm">
            נסי שוב
          </Button>
        </div>
      )}
    </Card>
  );
}
