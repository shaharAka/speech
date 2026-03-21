"use client";

import { useParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";

import { fetchText, uploadRecording, transcribeRecording, getAudioUrl } from "@/lib/api";
import TextDisplay from "@/components/practice/TextDisplay";
import AudioRecorder from "@/components/practice/AudioRecorder";
import TranscriptionResultView from "@/components/practice/TranscriptionResult";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import Spinner from "@/components/ui/Spinner";
import type { TranscriptionResult } from "@/types/api";

type Phase = "display" | "recording" | "processing" | "results";

export default function PracticePage() {
  const params = useParams();
  const textId = Number(params.textId);

  const [phase, setPhase] = useState<Phase>("display");
  const [result, setResult] = useState<TranscriptionResult | null>(null);
  const [recordingId, setRecordingId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);

  const { data: text, isLoading } = useQuery({
    queryKey: ["text", textId],
    queryFn: () => fetchText(textId),
    enabled: !!textId,
  });

  const handleRecordingComplete = async (blob: Blob) => {
    setIsRecording(false);
    setPhase("processing");
    setError(null);

    try {
      // Upload recording
      const recording = await uploadRecording(textId, blob);
      setRecordingId(recording.id);

      // Transcribe
      const transcription = await transcribeRecording(recording.id);
      setResult(transcription);
      setPhase("results");
    } catch (e) {
      setError(e instanceof Error ? e.message : "שגיאה בעיבוד ההקלטה");
      setPhase("display");
    }
  };

  const handleRecordingStateChange = (recording: boolean) => {
    setIsRecording(recording);
  };

  const handleTryAgain = () => {
    setResult(null);
    setRecordingId(null);
    setIsRecording(false);
    setPhase("display");
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-20">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!text) {
    return (
      <Card>
        <p className="text-center text-gray-500">הטקסט לא נמצא</p>
        <div className="text-center mt-4">
          <Link href="/texts">
            <Button variant="secondary">חזרה לטקסטים</Button>
          </Link>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Text to read — with word highlighting during recording */}
      <TextDisplay
        title={text.title}
        content={text.content}
        isRecording={isRecording}
      />

      {/* Error message */}
      {error && (
        <Card className="bg-red-50 border-red-200">
          <p className="text-red-700 text-center">{error}</p>
        </Card>
      )}

      {/* Recording / Processing / Results */}
      {phase !== "processing" && phase !== "results" && (
        <Card className="text-center">
          <p className="text-gray-600 mb-4">
            {isRecording ? "מקליטה... קראי בקצב נוח" : "קראי את הטקסט בקול רם"}
          </p>
          <AudioRecorder
            onRecordingComplete={handleRecordingComplete}
            onRecordingStateChange={handleRecordingStateChange}
          />
        </Card>
      )}

      {phase === "processing" && (
        <Card className="text-center py-12">
          <div className="flex flex-col items-center gap-4">
            <Spinner size="lg" />
            <p className="text-gray-600 text-lg">מעבד את ההקלטה...</p>
          </div>
        </Card>
      )}

      {phase === "results" && result && (
        <>
          <TranscriptionResultView
            result={result}
            audioUrl={recordingId ? getAudioUrl(recordingId) : undefined}
          />
          <div className="flex justify-center gap-4">
            <Button size="lg" onClick={handleTryAgain}>
              נסי שוב
            </Button>
            <Link href="/texts">
              <Button variant="secondary" size="lg">
                טקסט אחר
              </Button>
            </Link>
          </div>
        </>
      )}
    </div>
  );
}
