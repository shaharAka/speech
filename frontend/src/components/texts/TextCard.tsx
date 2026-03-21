import Link from "next/link";
import type { TextItem } from "@/types/api";
import Badge from "@/components/ui/Badge";

const difficultyLabels = {
  easy: "קל",
  medium: "בינוני",
  hard: "קשה",
};

interface TextCardProps {
  text: TextItem;
}

export default function TextCard({ text }: TextCardProps) {
  return (
    <Link href={`/practice/${text.id}`}>
      <div className={`bg-white rounded-xl shadow-sm border p-5 hover:shadow-md transition-shadow cursor-pointer ${text.recording_count > 0 ? "border-green-200 bg-green-50/30" : ""}`}>
        <div className="flex items-start justify-between mb-2">
          <div className="flex items-center gap-1.5">
            {text.recording_count > 0 && (
              <span className="text-green-500 text-sm font-bold">&#10003;</span>
            )}
            <h3 className="font-semibold text-gray-900">{text.title}</h3>
          </div>
          <Badge variant={text.difficulty}>
            {difficultyLabels[text.difficulty] || text.difficulty}
          </Badge>
        </div>
        <p className="text-gray-600 text-sm line-clamp-2 mb-3" dir="rtl">
          {text.content}
        </p>
        <div className="flex gap-4 text-xs text-gray-400">
          <span>{text.word_count} מילים</span>
          <span>{text.recording_count} הקלטות</span>
        </div>
      </div>
    </Link>
  );
}
