// src/utils/gemini.js — DEPRECATED: All Gemini calls now go through backend /api/chat
// This file is kept only for the rule-based fallback when backend is unreachable.

// Rule-based fallback responses (offline only — used when backend is completely down)
export function getRuleBasedResponse(msg, encounter, patient) {
  const ml = msg.toLowerCase();
  const r = encounter?.risks || {};

  if (ml.includes('hf') || ml.includes('heart failure'))
    return `**Heart Failure Risk: ${r.heart_failure || '?'}%**\n\nPlease ensure the backend Docker container is running for full AI-powered analysis with medical textbook citations.\n\n📋 VisionCare 3.0 — Offline Mode`;

  return `**VisionCare 3.0 — Offline Mode**\n\nThe Medical AI backend is not reachable. Please ensure Docker is running:\n\`docker compose up --build\`\n\nOnce running, the AI will use RAG (Retrieval-Augmented Generation) from medical textbooks + Gemini 2.5 Flash.`;
}
