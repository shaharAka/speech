export interface TextItem {
  id: number;
  title: string;
  content: string;
  difficulty: "easy" | "medium" | "hard";
  category: string;
  word_count: number;
  is_builtin: boolean;
  round: number;
  created_at: string;
  recording_count: number;
}

export interface TextListResponse {
  items: TextItem[];
  total: number;
}

export interface RecordingItem {
  id: number;
  text_id: number;
  audio_duration_ms: number;
  sample_rate: number;
  created_at: string;
  used_in_training: boolean;
  has_transcription: boolean;
}

export interface RecordingListResponse {
  items: RecordingItem[];
  total: number;
}

export interface WordDiffEntry {
  ref_word: string | null;
  hyp_word: string | null;
  status: "correct" | "substitution" | "insertion" | "deletion";
}

export interface TranscriptionResult {
  id: number;
  recording_id: number;
  raw_text: string;
  normalized_text: string;
  reference_text: string;
  wer_score: number;
  cer_score: number;
  word_diff: WordDiffEntry[];
  model_version_id: number;
  inference_time_ms: number;
  created_at: string;
}

export interface StatsOverview {
  total_recordings: number;
  total_transcriptions: number;
  average_wer: number | null;
  best_wer: number | null;
  recent_wer_trend: number[];
  recordings_needed_for_training: number;
}

export interface ModelVersion {
  id: number;
  version_tag: string;
  display_name: string;
  base_model_name: string;
  is_active: boolean;
  is_base: boolean;
  eval_wer: number | null;
  eval_wer_improvement: number | null;
  num_training_samples: number | null;
  created_at: string;
}

export interface TrainingRun {
  id: number;
  status: string;
  base_model_version_id: number;
  result_model_version_id: number | null;
  num_samples: number;
  num_epochs: number;
  lora_rank: number;
  learning_rate: number;
  train_wer: number | null;
  eval_wer: number | null;
  training_loss: number | null;
  error_message: string | null;
  coaching_status: string | null;
  started_at: string | null;
  completed_at: string | null;
  created_at: string;
}

export interface CoachingReport {
  id: number;
  training_run_id: number;
  round_number: number;
  next_round_number: number;
  summary_text: string;
  insights: Array<{
    category: string;
    message_he: string;
    severity: "info" | "warning" | "success";
  }>;
  recommendations: Array<{
    type: string;
    detail: string;
    priority: "high" | "medium" | "low";
  }>;
  wer_trajectory: Array<{
    run_id: number;
    eval_wer: number | null;
    train_wer: number | null;
    loss: number | null;
  }>;
  difficulty_distribution: Record<string, number>;
  suggested_next_params: Record<string, number> | null;
  texts_generated: number;
  is_round1_noise: boolean;
  created_at: string;
}

export interface DataStats {
  total_recordings: number;
  usable_recordings: number;
  min_required: number;
  is_ready: boolean;
}

export interface RoundProgress {
  current_round: number;
  total_texts: number;
  practiced_texts: number;
  is_complete: boolean;
  performance_summary: {
    has_data: boolean;
    overall_wer?: number;
    most_failed_words?: [string, number][];
    common_substitutions?: [string, number][];
    wer_by_difficulty?: Record<string, number>;
  } | null;
}

export interface GenerateRoundResult {
  round: number;
  texts_created: number;
}
