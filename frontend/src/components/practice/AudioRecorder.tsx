"use client";

import { useEffect } from "react";
import Button from "@/components/ui/Button";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";

interface AudioRecorderProps {
  onRecordingComplete: (blob: Blob) => void;
  onRecordingStateChange?: (isRecording: boolean) => void;
  disabled?: boolean;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function AudioRecorder({
  onRecordingComplete,
  onRecordingStateChange,
  disabled = false,
}: AudioRecorderProps) {
  const {
    isRecording,
    recordingDuration,
    audioBlob,
    audioUrl,
    startRecording,
    stopRecording,
    resetRecording,
  } = useAudioRecorder();

  // Notify parent of recording state changes
  useEffect(() => {
    onRecordingStateChange?.(isRecording);
  }, [isRecording, onRecordingStateChange]);

  const handleStop = () => {
    stopRecording();
  };

  const handleSubmit = () => {
    if (audioBlob) {
      onRecordingComplete(audioBlob);
    }
  };

  const handleReset = () => {
    resetRecording();
  };

  if (audioBlob && audioUrl) {
    return (
      <div className="flex flex-col items-center gap-4">
        <audio src={audioUrl} controls className="w-full max-w-md" />
        <div className="flex gap-3">
          <Button variant="success" size="lg" onClick={handleSubmit}>
            שליחה לזיהוי
          </Button>
          <Button variant="secondary" onClick={handleReset}>
            הקלטה מחדש
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-4">
      {isRecording ? (
        <>
          <div className="flex items-center gap-3">
            <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse" />
            <span className="text-2xl font-mono text-red-600">
              {formatTime(recordingDuration)}
            </span>
          </div>
          <Button variant="danger" size="lg" onClick={handleStop}>
            עצירה
          </Button>
        </>
      ) : (
        <Button size="lg" onClick={startRecording} disabled={disabled}>
          התחלת הקלטה
        </Button>
      )}
    </div>
  );
}
