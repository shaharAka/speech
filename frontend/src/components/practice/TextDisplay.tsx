"use client";

import { useState, useEffect, useRef, useCallback } from "react";

const SPEED_PRESETS = [
  { label: "🐢", wpm: 35, title: "לאט מאוד" },
  { label: "🚶", wpm: 50, title: "לאט" },
  { label: "🏃", wpm: 70, title: "רגיל" },
  { label: "⚡", wpm: 90, title: "מהיר" },
];

interface TextDisplayProps {
  title: string;
  content: string;
  isRecording?: boolean;
  /** Words per minute reading pace. Default 50 (slow, comfortable for practice) */
  wordsPerMinute?: number;
}

/**
 * Merge standalone punctuation tokens with the preceding word so they
 * don't consume their own highlight slot during reading pace.
 * e.g. ["שלום", ",", "עולם", "."] → ["שלום,", "עולם."]
 */
function mergeWordsWithPunctuation(tokens: string[]): string[] {
  const merged: string[] = [];
  for (const token of tokens) {
    // Token is purely punctuation / symbols (Unicode-aware)
    if (/^[.,!?;:…\-–—"'""''()[\]{}·׳״]+$/.test(token)) {
      if (merged.length > 0) {
        merged[merged.length - 1] += token;
      } else {
        merged.push(token);
      }
    } else {
      merged.push(token);
    }
  }
  return merged;
}

export default function TextDisplay({
  title,
  content,
  isRecording = false,
  wordsPerMinute = 50,
}: TextDisplayProps) {
  const words = mergeWordsWithPunctuation(content.split(/\s+/).filter(Boolean));
  const [activeWordIndex, setActiveWordIndex] = useState(-1);
  const [wpm, setWpm] = useState(wordsPerMinute);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const activeWordRef = useRef<HTMLSpanElement | null>(null);

  // Milliseconds per word based on reading pace
  const msPerWord = Math.round(60000 / wpm);
  /** Extra pause (ms) after a word that ends with a sentence-ending mark */
  const SENTENCE_PAUSE_MS = 2000;

  const stopHighlighting = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    setActiveWordIndex(-1);
  }, []);

  useEffect(() => {
    if (!isRecording) {
      stopHighlighting();
      return;
    }

    setActiveWordIndex(0);

    /** Schedule the next word advance using setTimeout so each word can
     *  have a different dwell time (longer after sentence-ending punctuation). */
    function scheduleNext(currentIdx: number) {
      if (currentIdx >= words.length - 1) return; // last word — stay

      // Check if the current word ends with a period (or other sentence ender)
      const endsWithPeriod = /[.]$/.test(words[currentIdx]);
      const delay = msPerWord + (endsWithPeriod ? SENTENCE_PAUSE_MS : 0);

      timerRef.current = setTimeout(() => {
        const next = currentIdx + 1;
        setActiveWordIndex(next);
        scheduleNext(next);
      }, delay);
    }

    scheduleNext(0);

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isRecording, msPerWord, words.length, stopHighlighting]);

  // Scroll active word into view (smooth)
  useEffect(() => {
    if (activeWordRef.current) {
      activeWordRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
        inline: "center",
      });
    }
  }, [activeWordIndex]);

  return (
    <div className="bg-white rounded-xl shadow-sm border p-8">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-500">{title}</h2>
        {isRecording && activeWordIndex >= 0 && (
          <span className="text-sm text-gray-400 font-mono">
            {activeWordIndex + 1} / {words.length}
          </span>
        )}
      </div>

      {/* Speed control — show before and during recording */}
      <div className="flex items-center gap-2 mb-5" dir="rtl">
        <span className="text-xs text-gray-400 ml-1">קצב:</span>
        {SPEED_PRESETS.map((preset) => (
          <button
            key={preset.wpm}
            onClick={() => setWpm(preset.wpm)}
            title={preset.title}
            className={`
              px-2.5 py-1 rounded-full text-sm transition-all
              ${wpm === preset.wpm
                ? "bg-blue-100 text-blue-700 ring-2 ring-blue-300 font-medium"
                : "bg-gray-100 text-gray-500 hover:bg-gray-200"
              }
            `}
          >
            {preset.label}
          </button>
        ))}
      </div>

      <p
        className="text-2xl leading-relaxed text-gray-900"
        style={{ lineHeight: "2.2" }}
        dir="rtl"
      >
        {words.map((word, i) => {
          const isActive = isRecording && i === activeWordIndex;
          const isPast = isRecording && i < activeWordIndex;
          const isFuture = isRecording && i > activeWordIndex;

          return (
            <span
              key={i}
              ref={isActive ? activeWordRef : null}
              className={`
                inline-block transition-all duration-300 rounded px-1 mx-0.5
                ${isActive
                  ? "bg-blue-100 text-blue-900 font-bold scale-105 ring-2 ring-blue-300"
                  : isPast
                    ? "text-gray-400"
                    : isFuture
                      ? "text-gray-900"
                      : "text-gray-900"
                }
              `}
              style={{
                transform: isActive ? "scale(1.05)" : "scale(1)",
              }}
            >
              {word}
            </span>
          );
        })}
      </p>
    </div>
  );
}
