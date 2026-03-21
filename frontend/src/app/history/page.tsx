"use client";

import { useQuery } from "@tanstack/react-query";

import { fetchRecordings, fetchStats, getAudioUrl } from "@/lib/api";
import Card from "@/components/ui/Card";
import Spinner from "@/components/ui/Spinner";

export default function HistoryPage() {
  const { data: recordings, isLoading: loadingRec } = useQuery({
    queryKey: ["recordings"],
    queryFn: () => fetchRecordings({ limit: 50 }),
  });

  const { data: stats, isLoading: loadingStats } = useQuery({
    queryKey: ["stats"],
    queryFn: fetchStats,
  });

  if (loadingRec || loadingStats) {
    return (
      <div className="flex justify-center py-20">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">היסטוריה</h1>

      {/* Stats cards */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <Card className="text-center">
            <div className="text-3xl font-bold text-blue-600">
              {stats.total_recordings}
            </div>
            <div className="text-sm text-gray-500">הקלטות</div>
          </Card>
          <Card className="text-center">
            <div className="text-3xl font-bold text-green-600">
              {stats.average_wer != null
                ? `${Math.round((1 - stats.average_wer) * 100)}%`
                : "--"}
            </div>
            <div className="text-sm text-gray-500">דיוק ממוצע</div>
          </Card>
          <Card className="text-center">
            <div className="text-3xl font-bold text-purple-600">
              {stats.best_wer != null
                ? `${Math.round((1 - stats.best_wer) * 100)}%`
                : "--"}
            </div>
            <div className="text-sm text-gray-500">דיוק מקסימלי</div>
          </Card>
          <Card className="text-center">
            <div className="text-3xl font-bold text-orange-600">
              {stats.recordings_needed_for_training}
            </div>
            <div className="text-sm text-gray-500">הקלטות עד אימון</div>
          </Card>
        </div>
      )}

      {/* Recording list */}
      {recordings?.items.length === 0 ? (
        <Card>
          <p className="text-center text-gray-500">אין עדיין הקלטות</p>
        </Card>
      ) : (
        <div className="space-y-3">
          {recordings?.items.map((rec) => (
            <Card key={rec.id} className="flex items-center justify-between">
              <div>
                <div className="font-medium">הקלטה #{rec.id}</div>
                <div className="text-sm text-gray-500">
                  {new Date(rec.created_at).toLocaleDateString("he-IL")} &middot;{" "}
                  {(rec.audio_duration_ms / 1000).toFixed(1)} שניות
                </div>
              </div>
              <div className="flex items-center gap-3">
                {rec.has_transcription && (
                  <span className="text-green-600 text-sm">תומלל</span>
                )}
                <audio src={getAudioUrl(rec.id)} controls className="h-8" />
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
