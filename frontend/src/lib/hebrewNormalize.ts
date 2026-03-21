/**
 * Strip nikud (Hebrew vowel diacritics) from text.
 * Unicode range: U+0591 to U+05C7
 */
export function stripNikud(text: string): string {
  return text.replace(/[\u0591-\u05C7]/g, "");
}

/**
 * Normalize Hebrew text for display comparison.
 */
export function normalizeHebrew(text: string): string {
  let normalized = text.normalize("NFC");
  normalized = stripNikud(normalized);
  normalized = normalized.replace(/[\u05BE\u05C0\u05C3\u05C6\u05F3\u05F4]/g, "");
  normalized = normalized.replace(/\s+/g, " ").trim();
  return normalized;
}
