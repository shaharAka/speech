"use client";

import "./globals.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import Navigation from "@/components/layout/Navigation";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const [queryClient] = useState(() => new QueryClient());

  return (
    <html lang="he" dir="rtl">
      <head>
        <title>אימון דיבור - Whisper</title>
        <meta name="description" content="אימון זיהוי דיבור עם Whisper" />
      </head>
      <body className="font-heebo bg-gray-50 text-gray-900 min-h-screen">
        <QueryClientProvider client={queryClient}>
          <Navigation />
          <main className="max-w-6xl mx-auto px-4 py-8">{children}</main>
        </QueryClientProvider>
      </body>
    </html>
  );
}
