import { describe, expect, it } from '@jest/globals';

import { spendFigures, spendUsd, type CouncilSpendLike, type SpendBucketLike } from './spend';

function bucket(over: Partial<SpendBucketLike> = {}): SpendBucketLike {
  return { councils: 0, spend_usd: 0, unmeasured: 0, is_floor: false, broker_submits: 0, ...over };
}

/** The box's 30-day report from 2026-11-04: 22 runs, 2 orders, 15 runs since. */
function live(over: Partial<CouncilSpendLike> = {}): CouncilSpendLike {
  return {
    window_days: 30,
    scheduled: bucket({ councils: 242, spend_usd: 539.02, broker_submits: 2 }),
    run_days: 22,
    mean_per_run_day_usd: 24.5,
    monthly_run_rate_usd: 514.52,
    spend_per_submit_usd: 269.51,
    last_submit_session: '2026-10-13',
    since_last_submit: bucket({ councils: 165, spend_usd: 370.85 }),
    run_days_since_last_submit: 15,
    ...over,
  };
}

describe('spendUsd', () => {
  it('marks a floor with a trailing plus', () => {
    expect(spendUsd(24.5, true)).toBe('$24.50+');
    expect(spendUsd(24.5)).toBe('$24.50');
  });

  it('never puts a plus on a missing figure', () => {
    expect(spendUsd(null, true)).toBe('—');
  });
});

describe('spendFigures', () => {
  it('prices the live inert streak', () => {
    const f = spendFigures(live());
    expect(f?.perRun).toEqual({ value: '$24.50', hint: '22 koşu ort.' });
    expect(f?.perSubmit).toEqual({ value: '$269.51', hint: '2 emir' });
    expect(f?.sinceSubmit).toEqual({ value: '$370.85', hint: '15 koşu · 13.10 sonrası' });
    expect(f?.runRate).toBe('Bu hızla ayda ~$514.52 (koşu ort. × 21 seans)');
    expect(f?.floorNote).toBeNull();
  });

  it('has no price per order when nothing reached the broker — not $0', () => {
    const f = spendFigures(
      live({
        spend_per_submit_usd: null,
        last_submit_session: null,
        scheduled: bucket({ councils: 33, spend_usd: 74.0 }),
        since_last_submit: bucket({ councils: 33, spend_usd: 74.0 }),
        run_days_since_last_submit: 3,
      }),
    );
    expect(f?.perSubmit).toEqual({ value: '—', hint: "broker'a emir yok" });
    expect(f?.sinceSubmit.hint).toBe('3 koşu · 30 günde emir yok');
  });

  it('carries an unmeasured council through as a floor', () => {
    const f = spendFigures(
      live({
        scheduled: bucket({ councils: 242, spend_usd: 539.02, broker_submits: 2, unmeasured: 4, is_floor: true }),
        since_last_submit: bucket({ councils: 165, spend_usd: 370.85, unmeasured: 4, is_floor: true }),
      }),
    );
    expect(f?.perRun.value).toBe('$24.50+');
    expect(f?.perSubmit.value).toBe('$269.51+');
    expect(f?.sinceSubmit.value).toBe('$370.85+');
    expect(f?.runRate).toContain('$514.52+');
    expect(f?.floorNote).toBe("4 council ölçülmemiş — '+' işaretli tutarlar alt sınır.");
  });

  it('shows the morning after an order as zero since, not a dash', () => {
    const f = spendFigures(live({ since_last_submit: bucket(), run_days_since_last_submit: 0 }));
    expect(f?.sinceSubmit).toEqual({ value: '$0.00', hint: '0 koşu · 13.10 sonrası' });
  });

  it('draws nothing when no nightly run is in the window', () => {
    expect(spendFigures(live({ run_days: 0, mean_per_run_day_usd: null }))).toBeNull();
    expect(spendFigures(undefined)).toBeNull();
  });
});
