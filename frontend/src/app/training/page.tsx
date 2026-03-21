"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";

import {
  fetchDataStats,
  fetchTrainingRuns,
  fetchModels,
  startTraining,
  fetchCoachingReport,
} from "@/lib/api";
import Card from "@/components/ui/Card";
import Button from "@/components/ui/Button";
import Badge from "@/components/ui/Badge";
import Spinner from "@/components/ui/Spinner";
import type { CoachingReport, TrainingRun } from "@/types/api";

function CoachingStatusBadge({ run }: { run: TrainingRun }) {
  if (!run.coaching_status) return null;
  if (run.coaching_status === "generating") {
    return <span className="text-xs text-blue-600 animate-pulse">מייצר דוח אימון...</span>;
  }
  if (run.coaching_status === "failed") {
    return <span className="text-xs text-red-500">דוח אימון נכשל</span>;
  }
  return null;
}

function CoachingReportCard({ runId }: { runId: number }) {
  const { data: report, isLoading } = useQuery({
    queryKey: ["coaching-report", runId],
    queryFn: () => fetchCoachingReport(runId),
  });

  if (isLoading) {
    return (
      <div className="mt-3 p-4 bg-blue-50 rounded-lg text-center">
        <Spinner size="sm" /> <span className="text-sm text-blue-600 mr-2">טוען דוח...</span>
      </div>
    );
  }

  if (!report) return null;

  return (
    <div className="mt-3 space-y-3" dir="rtl">
      {/* Summary */}
      <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-200">
        <h4 className="font-semibold text-blue-900 mb-2">סיכום המאמן</h4>
        <p className="text-gray-800 leading-relaxed">{report.summary_text}</p>
      </div>

      {/* Insights */}
      {report.insights.length > 0 && (
        <div className="p-4 bg-gray-50 rounded-lg">
          <h4 className="font-semibold text-gray-700 mb-2">תובנות</h4>
          <div className="space-y-2">
            {report.insights.map((insight, i) => (
              <div key={i} className="flex items-start gap-2">
                <span
                  className={`mt-0.5 inline-block w-2 h-2 rounded-full flex-shrink-0 ${
                    insight.severity === "success"
                      ? "bg-green-500"
                      : insight.severity === "warning"
                      ? "bg-yellow-500"
                      : "bg-blue-500"
                  }`}
                />
                <span className="text-sm text-gray-700">{insight.message_he}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {report.recommendations.length > 0 && (
        <div className="p-4 bg-gray-50 rounded-lg">
          <h4 className="font-semibold text-gray-700 mb-2">המלצות</h4>
          <div className="space-y-2">
            {report.recommendations.map((rec, i) => (
              <div key={i} className="flex items-start gap-2">
                <Badge
                  variant={
                    rec.priority === "high" ? "hard" : rec.priority === "medium" ? "medium" : "easy"
                  }
                >
                  {rec.priority === "high" ? "חשוב" : rec.priority === "medium" ? "מומלץ" : "טיפ"}
                </Badge>
                <span className="text-sm text-gray-700">{rec.detail}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Next round info */}
      <div className="p-3 bg-green-50 rounded-lg border border-green-200 text-center">
        <span className="text-green-700 font-medium">
          נוצרו {report.texts_generated} טקסטים לסבב {report.next_round_number}
        </span>
        {report.is_round1_noise && (
          <p className="text-xs text-green-600 mt-1">
            סבב 1 הוא בסיס — ניתוח משמעותי יתחיל מסבב 2
          </p>
        )}
      </div>
    </div>
  );
}

export default function TrainingPage() {
  const queryClient = useQueryClient();
  const [trainingError, setTrainingError] = useState<string | null>(null);
  const [expandedRun, setExpandedRun] = useState<number | null>(null);

  const { data: dataStats, isLoading: loadingStats } = useQuery({
    queryKey: ["data-stats"],
    queryFn: fetchDataStats,
  });

  const { data: runs, isLoading: loadingRuns } = useQuery({
    queryKey: ["training-runs"],
    queryFn: fetchTrainingRuns,
    refetchInterval: (query) => {
      const data = query.state.data;
      return data?.some(
        (r) =>
          r.status === "pending" ||
          r.status === "running" ||
          r.coaching_status === "generating"
      )
        ? 10000
        : false;
    },
  });

  const { data: models, isLoading: loadingModels } = useQuery({
    queryKey: ["models"],
    queryFn: fetchModels,
  });

  const trainMutation = useMutation({
    mutationFn: () => startTraining(),
    onSuccess: () => {
      setTrainingError(null);
      queryClient.invalidateQueries({ queryKey: ["training-runs"] });
      queryClient.invalidateQueries({ queryKey: ["data-stats"] });
    },
    onError: (err: Error) => {
      setTrainingError(err.message);
    },
  });

  if (loadingStats || loadingRuns || loadingModels) {
    return (
      <div className="flex justify-center py-20">
        <Spinner size="lg" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">אימון מודל</h1>

      {/* Data readiness */}
      {dataStats && (
        <Card>
          <h2 className="text-lg font-semibold mb-4">מוכנות נתונים</h2>
          <div className="mb-4">
            <div className="flex justify-between text-sm mb-1">
              <span>הקלטות: {dataStats.total_recordings}</span>
              <span>מינימום: {dataStats.min_required}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3">
              <div
                className={`h-3 rounded-full transition-all ${
                  dataStats.is_ready ? "bg-green-500" : "bg-blue-500"
                }`}
                style={{
                  width: `${Math.min(
                    100,
                    (dataStats.total_recordings / dataStats.min_required) * 100
                  )}%`,
                }}
              />
            </div>
          </div>
          {trainingError && (
            <p className="text-red-600 text-sm mb-2">{trainingError}</p>
          )}
          {dataStats.is_ready ? (
            <div className="flex items-center gap-3">
              <span className="text-green-600 font-medium">
                מספיק נתונים לאימון!
              </span>
              <Button
                onClick={() => trainMutation.mutate()}
                disabled={trainMutation.isPending}
              >
                {trainMutation.isPending ? "מתחיל אימון..." : "התחלת אימון"}
              </Button>
            </div>
          ) : (
            <p className="text-gray-600">
              נדרשות עוד{" "}
              {dataStats.min_required - dataStats.total_recordings} הקלטות
              כדי להתחיל אימון
            </p>
          )}
        </Card>
      )}

      {/* Model versions */}
      <Card>
        <h2 className="text-lg font-semibold mb-4">גרסאות מודל</h2>
        {models?.length === 0 ? (
          <p className="text-gray-500">אין גרסאות מודל</p>
        ) : (
          <div className="space-y-3">
            {models?.map((model) => (
              <div
                key={model.id}
                className="flex items-center justify-between p-3 rounded-lg bg-gray-50"
              >
                <div>
                  <div className="font-medium">{model.display_name}</div>
                  <div className="text-sm text-gray-500">
                    {model.version_tag}
                    {model.eval_wer != null &&
                      ` | WER: ${Math.round(model.eval_wer * 100)}%`}
                  </div>
                </div>
                {model.is_active ? (
                  <Badge variant="easy">פעיל</Badge>
                ) : (
                  <Button variant="secondary" size="sm">
                    הפעלה
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Training history with coaching reports */}
      <Card>
        <h2 className="text-lg font-semibold mb-4">היסטוריית אימונים</h2>
        {runs?.length === 0 ? (
          <p className="text-gray-500">לא בוצעו אימונים עדיין</p>
        ) : (
          <div className="space-y-3">
            {runs?.map((run) => (
              <div key={run.id} className="rounded-lg bg-gray-50 overflow-hidden">
                <div
                  className="flex items-center justify-between p-3 cursor-pointer hover:bg-gray-100 transition-colors"
                  onClick={() =>
                    setExpandedRun(expandedRun === run.id ? null : run.id)
                  }
                >
                  <div>
                    <div className="font-medium">אימון #{run.id}</div>
                    <div className="text-sm text-gray-500">
                      {run.num_samples} דגימות | {run.num_epochs} אפוקים
                      {run.eval_wer != null &&
                        ` | WER: ${Math.round(run.eval_wer * 100)}%`}
                    </div>
                    <CoachingStatusBadge run={run} />
                  </div>
                  <div className="flex items-center gap-2">
                    {run.coaching_status === "completed" && (
                      <span className="text-xs text-blue-600">דוח מאמן</span>
                    )}
                    <Badge
                      variant={
                        run.status === "completed"
                          ? "easy"
                          : run.status === "failed"
                          ? "hard"
                          : "medium"
                      }
                    >
                      {run.status}
                    </Badge>
                  </div>
                </div>

                {/* Expanded: coaching report */}
                {expandedRun === run.id && run.coaching_status === "completed" && (
                  <div className="px-3 pb-3">
                    <CoachingReportCard runId={run.id} />
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
