/**
 * How much of the eval verdict is the window itself — the card's reading of
 * `/v1/eval`'s optional `stability` block (backend:
 * agent/tradingagents_us/risk/verdict_stability.py).
 *
 * A 21-session window drops a session off the front and adds one at the back
 * every day. On 2026-10-20 that alone turned 1M GO into NO-GO (Sharpe 2.03 →
 * −0.43) on a session the book made +0.52%: the window's best day (+1.91%)
 * had left the front. The headline verdict cannot tell that apart from the
 * strategy breaking, so the card shows the last N windows next to it and,
 * when the newest move flipped the verdict, which side of the window did it.
 *
 * Pure: no React/RN, so jest covers it without a renderer.
 */

import type { EvalSessionReturn, EvalStability } from '@/api/types';

import { formatPct } from './format';

/** One dot in the strip, oldest first. */
export interface StabilityMark {
  end: string;
  passed: boolean;
}

/**
 * Which side of the window turned the verdict on the newest move.
 *
 * - `window`: the session that joined pushed the other way (the book made
 *   money and the verdict still fell, or lost money and it still rose), so
 *   the flip came from the session that left the front.
 * - `session`: the newest session pushed the verdict where it went.
 */
export type MoveKind = 'window' | 'session';

export interface LastMove {
  kind: MoveKind;
  to: 'pass' | 'fail';
  text: string;
}

/** True when the block is usable: a non-empty window list with sane counts. */
function usable(st: EvalStability | null | undefined): st is EvalStability {
  return !!st && Array.isArray(st.windows) && st.windows.length > 0 && st.total > 0;
}

/** "2026-09-21" → "21.09"; anything else passes through untouched. */
export function sessionLabel(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  return m ? `${m[3]}.${m[2]}` : iso;
}

/** Percent points (1.91) → "+1.91%", with the U+2212 minus from format.ts. */
function points(r: EvalSessionReturn): string {
  return formatPct(r.return_pct / 100, { signed: true });
}

/** Two-decimal Sharpe with the same U+2212 minus the figures use. */
function sharpe(n: number): string {
  const body = Math.abs(n).toFixed(2);
  return n < 0 ? `−${body}` : body;
}

export function stabilityMarks(st: EvalStability | null | undefined): StabilityMark[] {
  if (!usable(st)) return [];
  return st.windows.map((w) => ({ end: w.end, passed: w.passed }));
}

/** "Son 10 pencereden 9 geçti · 1 dönüş · Sharpe −0.43…3.46" */
export function stabilitySummary(st: EvalStability | null | undefined): string | null {
  if (!usable(st)) return null;
  const flips = st.flips > 0 ? `${st.flips} dönüş` : 'dönüş yok';
  return (
    `Son ${st.total} pencereden ${st.passed} geçti · ${flips} · ` +
    `Sharpe ${sharpe(st.sharpe_min)}…${sharpe(st.sharpe_max)}`
  );
}

/** Screen-reader sentence for the dot strip. */
export function stabilityA11y(st: EvalStability | null | undefined): string | null {
  if (!usable(st)) return null;
  const failed = st.total - st.passed;
  return `Pencere istikrarı: son ${st.total} pencereden ${st.passed} geçti, ${failed} kaldı, ${st.flips} dönüş`;
}

/**
 * The newest move, when it flipped the verdict. Null when the last two windows
 * agree, or when the backend could not name the two sessions that differ
 * between them — a flip with no cause is better left to the dots than guessed.
 */
export function lastMove(st: EvalStability | null | undefined): LastMove | null {
  if (!usable(st) || st.windows.length < 2) return null;
  const last = st.windows[st.windows.length - 1]!;
  const prev = st.windows[st.windows.length - 2]!;
  if (last.passed === prev.passed) return null;
  const { entered, departed } = st;
  if (!entered || !departed) return null;

  const to = last.passed ? 'pass' : 'fail';
  // Zero counts as "against": a flat session cannot have caused a flip.
  const joinedAgainst = to === 'fail' ? entered.return_pct >= 0 : entered.return_pct <= 0;
  const inn = `${sessionLabel(entered.session)} (${points(entered)})`;
  const out = `${sessionLabel(departed.session)} (${points(departed)})`;

  if (joinedAgainst) {
    return {
      kind: 'window',
      to,
      text: `Pencere kayması: son seans ${inn}; kararı önden düşen ${out} çevirdi.`,
    };
  }
  return {
    kind: 'session',
    to,
    text: `Son seans çevirdi: ${inn} eklendi, önden ${out} düştü.`,
  };
}
