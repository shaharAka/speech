import type { WordDiffEntry } from "@/types/api";

interface WordDiffProps {
  diff: WordDiffEntry[];
  werScore: number;
  cerScore: number;
}

const statusStyles = {
  correct: "bg-green-100 text-green-800",
  substitution: "bg-red-100 text-red-800",
  insertion: "bg-orange-100 text-orange-800",
  deletion: "bg-red-100 text-red-800 line-through",
};

export default function WordDiff({ diff, werScore, cerScore }: WordDiffProps) {
  const correctCount = diff.filter((d) => d.status === "correct").length;
  const totalRef = diff.filter((d) => d.ref_word).length;
  const accuracy = totalRef > 0 ? Math.round((correctCount / totalRef) * 100) : 0;

  return (
    <div className="space-y-6" dir="rtl">
      {/* Score summary */}
      <div className="flex gap-6 justify-center">
        <div className="text-center">
          <div className="text-3xl font-bold text-blue-600">{accuracy}%</div>
          <div className="text-sm text-gray-500">דיוק</div>
        </div>
        <div className="text-center">
          <div className="text-3xl font-bold text-gray-600">
            {Math.round(werScore * 100)}%
          </div>
          <div className="text-sm text-gray-500">WER</div>
        </div>
        <div className="text-center">
          <div className="text-3xl font-bold text-gray-600">
            {Math.round(cerScore * 100)}%
          </div>
          <div className="text-sm text-gray-500">CER</div>
        </div>
      </div>

      {/* Expected text */}
      <div>
        <h3 className="text-sm font-medium text-gray-500 mb-2">טקסט מקורי:</h3>
        <div className="flex flex-wrap gap-1.5 text-xl leading-loose">
          {diff.map((entry, i) => {
            if (entry.status === "insertion") return null; // Skip insertions in reference row
            return (
              <span
                key={`ref-${i}`}
                className={`px-1.5 py-0.5 rounded ${statusStyles[entry.status]}`}
                title={
                  entry.status === "substitution"
                    ? `זוהה: ${entry.hyp_word}`
                    : entry.status === "deletion"
                    ? "לא זוהה"
                    : ""
                }
              >
                {entry.ref_word}
              </span>
            );
          })}
        </div>
      </div>

      {/* Recognized text */}
      <div>
        <h3 className="text-sm font-medium text-gray-500 mb-2">זוהה:</h3>
        <div className="flex flex-wrap gap-1.5 text-xl leading-loose">
          {diff.map((entry, i) => {
            if (entry.status === "deletion") return null; // Skip deletions in hypothesis row
            return (
              <span
                key={`hyp-${i}`}
                className={`px-1.5 py-0.5 rounded ${
                  entry.status === "correct"
                    ? statusStyles.correct
                    : entry.status === "substitution"
                    ? "bg-orange-100 text-orange-800"
                    : statusStyles.insertion
                }`}
              >
                {entry.hyp_word}
              </span>
            );
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-sm text-gray-600 justify-center border-t pt-4">
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-green-100 border border-green-300" />
          נכון
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-red-100 border border-red-300" />
          שגיאה
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded bg-orange-100 border border-orange-300" />
          תוספת
        </span>
      </div>
    </div>
  );
}
