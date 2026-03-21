import type { TranscriptionResult as TResult } from "@/types/api";
import Card from "@/components/ui/Card";
import WordDiff from "./WordDiff";

interface Props {
  result: TResult;
  audioUrl?: string;
}

export default function TranscriptionResultView({ result, audioUrl }: Props) {
  return (
    <Card>
      <h2 className="text-lg font-semibold mb-4">תוצאות</h2>

      <WordDiff
        diff={result.word_diff}
        werScore={result.wer_score}
        cerScore={result.cer_score}
      />

      {audioUrl && (
        <div className="mt-4 pt-4 border-t">
          <h3 className="text-sm font-medium text-gray-500 mb-2">השמעה:</h3>
          <audio src={audioUrl} controls className="w-full" />
        </div>
      )}

      <div className="mt-4 pt-4 border-t text-sm text-gray-400 text-center">
        זמן עיבוד: {result.inference_time_ms}ms
      </div>
    </Card>
  );
}
