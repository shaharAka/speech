"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";

import {
  fetchStats,
  fetchTexts,
  fetchHealth,
  fetchRoundProgress,
  generateNextRound,
} from "@/lib/api";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import Spinner from "@/components/ui/Spinner";
import MicrophoneTest from "@/components/MicrophoneTest";

export default function HomePage() {
  const queryClient = useQueryClient();
  const [generateError, setGenerateError] = useState<string | null>(null);

  const { data: stats } = useQuery({
    queryKey: ["stats"],
    queryFn: fetchStats,
  });

  const { data: texts } = useQuery({
    queryKey: ["texts-recent"],
    queryFn: () => fetchTexts({ limit: 6 }),
  });

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
  });

  const { data: roundProgress } = useQuery({
    queryKey: ["round-progress"],
    queryFn: fetchRoundProgress,
    refetchInterval: 30000,
  });

  const generateMutation = useMutation({
    mutationFn: generateNextRound,
    onSuccess: () => {
      setGenerateError(null);
      queryClient.invalidateQueries({ queryKey: ["round-progress"] });
      queryClient.invalidateQueries({ queryKey: ["texts-recent"] });
      queryClient.invalidateQueries({ queryKey: ["texts"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
    onError: (e: Error) => {
      setGenerateError(e.message);
    },
  });

  const progressPct =
    roundProgress && roundProgress.total_texts > 0
      ? Math.round(
          (roundProgress.practiced_texts / roundProgress.total_texts) * 100
        )
      : 0;

  return (
    <div className="space-y-8">
      {/* Welcome */}
      <div className="text-center py-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">אימון דיבור</h1>
        <p className="text-gray-600 text-lg">
          תרגול קריאה בקול רם עם זיהוי דיבור חכם
        </p>
      </div>

      {/* Status */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="text-center">
          <div className="text-3xl font-bold text-blue-600">
            {stats?.total_recordings ?? 0}
          </div>
          <div className="text-sm text-gray-500">הקלטות</div>
        </Card>
        <Card className="text-center">
          <div className="text-3xl font-bold text-green-600">
            {stats?.average_wer != null
              ? `${Math.round((1 - stats.average_wer) * 100)}%`
              : "--"}
          </div>
          <div className="text-sm text-gray-500">דיוק ממוצע</div>
        </Card>
        <Card className="text-center">
          <div className="text-3xl font-bold text-purple-600">
            {stats?.total_transcriptions ?? 0}
          </div>
          <div className="text-sm text-gray-500">תמלולים</div>
        </Card>
        <Card className="text-center">
          <div
            className={`text-3xl font-bold ${
              health?.model_loaded ? "text-green-600" : "text-red-600"
            }`}
          >
            {health?.model_loaded ? "V" : "X"}
          </div>
          <div className="text-sm text-gray-500">מודל</div>
        </Card>
      </div>

      {/* Microphone test */}
      <MicrophoneTest />

      {/* Round progress */}
      {roundProgress && roundProgress.total_texts > 0 && (
        <Card>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="text-lg font-semibold">
                סבב {roundProgress.current_round}
              </span>
              <span className="text-sm text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">
                {roundProgress.practiced_texts} / {roundProgress.total_texts}{" "}
                טקסטים
              </span>
            </div>
            <span className="text-sm font-medium text-blue-600">
              {progressPct}%
            </span>
          </div>

          <div className="w-full bg-gray-200 rounded-full h-3 mb-3">
            <div
              className={`h-3 rounded-full transition-all duration-500 ${
                roundProgress.is_complete ? "bg-green-500" : "bg-blue-500"
              }`}
              style={{ width: `${progressPct}%` }}
            />
          </div>

          {roundProgress.is_complete ? (
            <div className="space-y-4">
              <div className="text-center py-3">
                <p className="text-xl font-bold text-green-600 mb-1">
                  כל הכבוד! סיימת את הסבב!
                </p>
                <p className="text-sm text-gray-500">
                  המערכת תיצור טקסטים חדשים מותאמים אישית על סמך הביצועים שלך
                </p>
              </div>

              {/* Performance insights */}
              {roundProgress.performance_summary?.has_data && (
                <div className="bg-gray-50 rounded-lg p-4 text-sm space-y-2">
                  <p className="font-medium text-gray-700">תובנות מהסבב:</p>
                  {roundProgress.performance_summary.overall_wer != null && (
                    <p className="text-gray-600">
                      דיוק כללי:{" "}
                      <span className="font-semibold">
                        {Math.round(
                          (1 - roundProgress.performance_summary.overall_wer) *
                            100
                        )}
                        %
                      </span>
                    </p>
                  )}
                  {roundProgress.performance_summary.most_failed_words &&
                    roundProgress.performance_summary.most_failed_words.length >
                      0 && (
                      <p className="text-gray-600">
                        מילים לשיפור:{" "}
                        <span className="font-medium text-red-600">
                          {roundProgress.performance_summary.most_failed_words
                            .slice(0, 8)
                            .map(([w]) => w)
                            .join("، ")}
                        </span>
                      </p>
                    )}
                </div>
              )}

              <div className="text-center">
                <Button
                  size="lg"
                  onClick={() => generateMutation.mutate()}
                  disabled={generateMutation.isPending}
                >
                  {generateMutation.isPending ? (
                    <span className="flex items-center gap-2">
                      <Spinner size="sm" />
                      יוצר טקסטים חדשים...
                    </span>
                  ) : (
                    "התחלת סבב חדש"
                  )}
                </Button>
                {generateError && (
                  <p className="text-sm text-red-600 mt-2">{generateError}</p>
                )}
                {generateMutation.isSuccess && (
                  <p className="text-sm text-green-600 mt-2">
                    נוצרו {generateMutation.data?.texts_created} טקסטים חדשים
                    לסבב {generateMutation.data?.round}!
                  </p>
                )}
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-500">
              {roundProgress.total_texts - roundProgress.practiced_texts > 0
                ? `עוד ${
                    roundProgress.total_texts - roundProgress.practiced_texts
                  } טקסטים לתרגול בסבב הזה`
                : ""}
            </p>
          )}
        </Card>
      )}

      {/* Training progress bar */}
      {stats && stats.recordings_needed_for_training > 0 && (
        <Card>
          <div className="flex justify-between text-sm mb-2">
            <span className="font-medium">התקדמות לקראת אימון מודל</span>
            <span className="text-gray-500">
              {stats.total_recordings} /{" "}
              {stats.total_recordings + stats.recordings_needed_for_training}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div
              className="h-3 rounded-full bg-blue-500 transition-all"
              style={{
                width: `${Math.round(
                  (stats.total_recordings /
                    (stats.total_recordings +
                      stats.recordings_needed_for_training)) *
                    100
                )}%`,
              }}
            />
          </div>
          <p className="text-sm text-gray-500 mt-2">
            עוד {stats.recordings_needed_for_training} הקלטות עד שנוכל לאמן את
            המודל על הדיבור שלך
          </p>
        </Card>
      )}

      {/* Quick start */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold">התחלת תרגול</h2>
          <Link href="/texts">
            <Button variant="secondary" size="sm">
              כל הטקסטים
            </Button>
          </Link>
        </div>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
          {texts?.items.map((text) => (
            <Link key={text.id} href={`/practice/${text.id}`}>
              <Card className="hover:shadow-md transition-shadow cursor-pointer">
                <div className="flex items-center gap-2 mb-1">
                  {text.recording_count > 0 && (
                    <span className="text-green-500 text-sm">&#10003;</span>
                  )}
                  <h3 className="font-semibold">{text.title}</h3>
                </div>
                <p className="text-sm text-gray-500 line-clamp-1">
                  {text.content}
                </p>
              </Card>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
