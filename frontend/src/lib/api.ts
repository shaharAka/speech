const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// Texts
export const fetchTexts = (params?: { difficulty?: string; limit?: number; offset?: number }) => {
  const query = new URLSearchParams();
  if (params?.difficulty) query.set("difficulty", params.difficulty);
  if (params?.limit) query.set("limit", String(params.limit));
  if (params?.offset) query.set("offset", String(params.offset));
  return request<import("@/types/api").TextListResponse>(`/texts?${query}`);
};

export const fetchText = (id: number) =>
  request<import("@/types/api").TextItem>(`/texts/${id}`);

export const createText = (data: { title: string; content: string; difficulty: string }) =>
  request<import("@/types/api").TextItem>("/texts", {
    method: "POST",
    body: JSON.stringify(data),
  });

// Round progress
export const fetchRoundProgress = () =>
  request<import("@/types/api").RoundProgress>("/texts/round-progress");

export const generateNextRound = () =>
  request<import("@/types/api").GenerateRoundResult>("/texts/generate-round", {
    method: "POST",
  });

// Recordings
export const uploadRecording = async (textId: number, audioBlob: Blob) => {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");

  const res = await fetch(`${API_URL}/api/recordings?text_id=${textId}`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<import("@/types/api").RecordingItem>;
};

export const fetchRecordings = (params?: { text_id?: number; limit?: number }) => {
  const query = new URLSearchParams();
  if (params?.text_id) query.set("text_id", String(params.text_id));
  if (params?.limit) query.set("limit", String(params.limit));
  return request<import("@/types/api").RecordingListResponse>(`/recordings?${query}`);
};

export const getAudioUrl = (recordingId: number) =>
  `${API_URL}/api/recordings/${recordingId}/audio`;

// Transcriptions
export const transcribeRecording = (recordingId: number) =>
  request<import("@/types/api").TranscriptionResult>("/transcriptions", {
    method: "POST",
    body: JSON.stringify({ recording_id: recordingId }),
  });

export const fetchStats = () =>
  request<import("@/types/api").StatsOverview>("/transcriptions/stats/overview");

// Models
export const fetchModels = () =>
  request<import("@/types/api").ModelVersion[]>("/models");

export const fetchActiveModel = () =>
  request<import("@/types/api").ModelVersion>("/models/active");

// Training
export const fetchDataStats = () =>
  request<import("@/types/api").DataStats>("/training/data-stats");

export const fetchTrainingRuns = () =>
  request<import("@/types/api").TrainingRun[]>("/training/runs");

export const startTraining = (params?: {
  num_epochs?: number;
  lora_rank?: number;
  learning_rate?: number;
}) =>
  request<import("@/types/api").TrainingRun>("/training/start", {
    method: "POST",
    body: JSON.stringify(params ?? {}),
  });

export const fetchCoachingReport = (runId: number) =>
  request<import("@/types/api").CoachingReport>(`/training/runs/${runId}/coaching-report`);

export const fetchLatestCoachingReport = () =>
  request<import("@/types/api").CoachingReport>("/training/latest-coaching-report");

// Health
export const fetchHealth = () =>
  request<{ status: string; model_loaded: boolean; model_path: string }>("/health");
