import { create } from "zustand";
import type { TranscriptionResult, WordDiffEntry } from "@/types/api";

type Phase = "display" | "recording" | "processing" | "results";

interface PracticeState {
  phase: Phase;
  setPhase: (phase: Phase) => void;
  transcriptionResult: TranscriptionResult | null;
  setTranscriptionResult: (result: TranscriptionResult | null) => void;
  reset: () => void;
}

export const usePracticeStore = create<PracticeState>((set) => ({
  phase: "display",
  setPhase: (phase) => set({ phase }),
  transcriptionResult: null,
  setTranscriptionResult: (result) => set({ transcriptionResult: result }),
  reset: () =>
    set({
      phase: "display",
      transcriptionResult: null,
    }),
}));
