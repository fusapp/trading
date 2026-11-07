/**
 * Council spend ("what did the decisions cost, and what did that buy?") —
 * pure, unit-tested, RN-free.
 *
 * Why this exists: every night the run pays for a full council on every
 * ticker, whether or not the book can act on the answer. In 2026-11 the
 * losing-streak breaker latched with the book flat, so every BUY was refused
 * before it was sized — and the councils kept running at ~$24 a night. The
 * order-flow card could say "frozen"; it could not say what frozen costs.
 *
 * GET /v1/diagnostics/spend answers from the persisted decision rows (the same
 * builder as `python -m scripts.spend_report`). Two backend rules are kept:
 *  - `cost_usd` NULL means "not measured", never "free". A bucket with an
 *    unmeasured council is a floor, and its figure carries a trailing '+'.
 *  - "Per broker order" counts orders the broker acknowledged. When none did,
 *    there is no price per order — not $0, and not the total.
 *
 * Tone is NOT decided here: whether this spend is a warning is the order-flow
 * verdict's call (`actionabilityVerdictMeta`), and a second copy of that rule
 * would drift.
 */

import { formatUsd } from './format';
import { sessionLabel } from './stability';

const EM_DASH = '—';

/** NYSE sessions per month — mirrors SESSIONS_PER_MONTH in council_spend.py. */
export const SESSIONS_PER_MONTH = 21;

export interface SpendBucketLike {
  councils: number;
  spend_usd: number;
  unmeasured: number;
  is_floor: boolean;
  broker_submits: number;
}

export interface CouncilSpendLike {
  window_days: number;
  scheduled: SpendBucketLike;
  run_days: number;
  mean_per_run_day_usd: number | null;
  monthly_run_rate_usd: number | null;
  spend_per_submit_usd: number | null;
  last_submit_session: string | null;
  since_last_submit: SpendBucketLike;
  run_days_since_last_submit: number;
}

export interface SpendFigure {
  value: string;
  hint: string;
}

export interface SpendFigures {
  perRun: SpendFigure;
  perSubmit: SpendFigure;
  sinceSubmit: SpendFigure;
  /** "Bu hızla ayda ~$514.52" — null when there is no measured mean. */
  runRate: string | null;
  /** Present only when some council in the window was never metered. */
  floorNote: string | null;
}

/** Money that may be a lower bound: "$24.50" or "$24.50+". */
export function spendUsd(value: number | null | undefined, floor = false): string {
  const body = formatUsd(value);
  return body === EM_DASH || !floor ? body : `${body}+`;
}

/**
 * The three figures of the spend row, or null when the window holds no nightly
 * run — there is nothing to price, and three dashes would read as "free".
 */
export function spendFigures(s: CouncilSpendLike | null | undefined): SpendFigures | null {
  if (!s || s.run_days <= 0) return null;

  const scheduled = s.scheduled;
  const since = s.since_last_submit;
  const n = s.run_days_since_last_submit;

  const perSubmit: SpendFigure =
    s.spend_per_submit_usd == null
      ? { value: EM_DASH, hint: "broker'a emir yok" }
      : {
          value: spendUsd(s.spend_per_submit_usd, scheduled.is_floor),
          hint: `${scheduled.broker_submits} emir`,
        };

  const sinceHint = s.last_submit_session
    ? `${n} koşu · ${sessionLabel(s.last_submit_session)} sonrası`
    : `${n} koşu · ${s.window_days} günde emir yok`;

  return {
    perRun: {
      value: spendUsd(s.mean_per_run_day_usd, scheduled.is_floor),
      hint: `${s.run_days} koşu ort.`,
    },
    perSubmit,
    sinceSubmit: {
      value: spendUsd(since.spend_usd, since.is_floor),
      hint: sinceHint,
    },
    runRate:
      s.monthly_run_rate_usd == null
        ? null
        : `Bu hızla ayda ~${spendUsd(s.monthly_run_rate_usd, scheduled.is_floor)} (koşu ort. × ${SESSIONS_PER_MONTH} seans)`,
    floorNote:
      scheduled.unmeasured > 0
        ? `${scheduled.unmeasured} council ölçülmemiş — '+' işaretli tutarlar alt sınır.`
        : null,
  };
}
