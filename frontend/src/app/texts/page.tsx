"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";

import { fetchTexts, fetchRoundProgress } from "@/lib/api";
import TextCard from "@/components/texts/TextCard";
import Button from "@/components/ui/Button";
import Spinner from "@/components/ui/Spinner";

export default function TextsPage() {
  const [difficulty, setDifficulty] = useState<string | undefined>();

  const { data, isLoading } = useQuery({
    queryKey: ["texts", difficulty],
    queryFn: () => fetchTexts({ difficulty, limit: 100 }),
  });

  const { data: roundProgress } = useQuery({
    queryKey: ["round-progress"],
    queryFn: fetchRoundProgress,
  });

  // Sort: un-practiced first, then practiced
  const sortedItems = data?.items
    ? [...data.items].sort((a, b) => {
        const aPracticed = a.recording_count > 0 ? 1 : 0;
        const bPracticed = b.recording_count > 0 ? 1 : 0;
        return aPracticed - bPracticed;
      })
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">טקסטים לתרגול</h1>
          {roundProgress && (
            <span className="text-sm bg-blue-100 text-blue-700 px-2.5 py-1 rounded-full font-medium">
              סבב {roundProgress.current_round}
            </span>
          )}
        </div>
        <Link href="/texts/new">
          <Button>הוספת טקסט</Button>
        </Link>
      </div>

      {/* Progress summary */}
      {roundProgress && roundProgress.total_texts > 0 && (
        <div className="flex items-center gap-3 text-sm text-gray-500">
          <span>
            {roundProgress.practiced_texts} / {roundProgress.total_texts} תורגלו
          </span>
          <div className="flex-1 bg-gray-200 rounded-full h-1.5 max-w-xs">
            <div
              className={`h-1.5 rounded-full transition-all ${
                roundProgress.is_complete ? "bg-green-500" : "bg-blue-500"
              }`}
              style={{
                width: `${Math.round(
                  (roundProgress.practiced_texts / roundProgress.total_texts) *
                    100
                )}%`,
              }}
            />
          </div>
        </div>
      )}

      {/* Difficulty filter */}
      <div className="flex gap-2">
        {[
          { value: undefined, label: "הכל" },
          { value: "easy", label: "קל" },
          { value: "medium", label: "בינוני" },
          { value: "hard", label: "קשה" },
        ].map((opt) => (
          <button
            key={opt.value ?? "all"}
            onClick={() => setDifficulty(opt.value)}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              difficulty === opt.value
                ? "bg-blue-600 text-white"
                : "bg-gray-100 text-gray-700 hover:bg-gray-200"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {/* Text list */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <Spinner />
        </div>
      ) : sortedItems.length === 0 ? (
        <p className="text-center text-gray-500 py-12">אין טקסטים עדיין</p>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {sortedItems.map((text) => (
            <TextCard key={text.id} text={text} />
          ))}
        </div>
      )}
    </div>
  );
}
