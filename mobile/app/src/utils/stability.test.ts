import { describe, expect, it } from '@jest/globals';

import type { EvalStability, EvalWindow } from '@/api/types';

import { lastMove, sessionLabel, stabilityA11y, stabilityMarks, stabilitySummary } from './stability';

function win(end: string, passed: boolean, sharpe = passed ? 2 : -0.4): EvalWindow {
  return { base: '2026-09-08', end, total_return_pct: passed ? 2 : 0, sharpe, max_dd_pct: -1.5, passed };
}

/** The live 1M block from 2026-10-21: nine GO windows, then NO-GO on a +0.52% day. */
function live1M(over: Partial<EvalStability> = {}): EvalStability {
  const windows = [
    '2026-10-06',
    '2026-10-07',
    '2026-10-08',
    '2026-10-09',
    '2026-10-12',
    '2026-10-13',
    '2026-10-14',
    '2026-10-15',
    '2026-10-16',
  ].map((d) => win(d, true));
  windows.push(win('2026-10-19', false, -0.43));
  return {
    window_days: 21,
    passed: 9,
    total: 10,
    flips: 1,
    sharpe_min: -0.43,
    sharpe_max: 3.46,
    windows,
    departed: { session: '2026-09-21', return_pct: 1.91 },
    entered: { session: '2026-10-19', return_pct: 0.52 },
    ...over,
  };
}

describe('sessionLabel', () => {
  it('reads an ISO session as day.month', () => {
    expect(sessionLabel('2026-09-21')).toBe('21.09');
  });
  it('passes anything else through', () => {
    expect(sessionLabel('yesterday')).toBe('yesterday');
  });
});

describe('stabilitySummary', () => {
  it('counts passes and flips and carries the Sharpe range with a true minus', () => {
    expect(stabilitySummary(live1M())).toBe('Son 10 pencereden 9 geçti · 1 dönüş · Sharpe −0.43…3.46');
  });
  it('says there was no flip rather than "0 dönüş"', () => {
    expect(stabilitySummary(live1M({ flips: 0 }))).toContain('dönüş yok');
  });
  it('is null when the block is missing, null or empty', () => {
    expect(stabilitySummary(undefined)).toBeNull();
    expect(stabilitySummary(null)).toBeNull();
    expect(stabilitySummary(live1M({ windows: [], total: 0 }))).toBeNull();
  });
});

describe('stabilityMarks', () => {
  it('keeps the backend order, oldest first, newest last', () => {
    const marks = stabilityMarks(live1M());
    expect(marks).toHaveLength(10);
    expect(marks[0]).toEqual({ end: '2026-10-06', passed: true });
    expect(marks[9]).toEqual({ end: '2026-10-19', passed: false });
  });
  it('is empty without a usable block', () => {
    expect(stabilityMarks(null)).toEqual([]);
  });
});

describe('stabilityA11y', () => {
  it('reads passes, misses and flips as one sentence', () => {
    expect(stabilityA11y(live1M())).toBe('Pencere istikrarı: son 10 pencereden 9 geçti, 1 kaldı, 1 dönüş');
  });
});

describe('lastMove', () => {
  it('names the window as the cause when the book made money and the verdict still fell (10-20)', () => {
    const move = lastMove(live1M());
    expect(move).toEqual({
      kind: 'window',
      to: 'fail',
      text: 'Pencere kayması: son seans 19.10 (+0.52%); kararı önden düşen 21.09 (+1.91%) çevirdi.',
    });
  });

  it('names the session when the newest return pushed the verdict where it went', () => {
    const move = lastMove(live1M({ entered: { session: '2026-10-19', return_pct: -2.4 } }));
    expect(move?.kind).toBe('session');
    expect(move?.to).toBe('fail');
    expect(move?.text).toBe('Son seans çevirdi: 19.10 (−2.40%) eklendi, önden 21.09 (+1.91%) düştü.');
  });

  it('reads an upward flip on a losing session as the window', () => {
    const st = live1M({
      windows: [win('2026-10-16', false), win('2026-10-19', true)],
      entered: { session: '2026-10-19', return_pct: -0.1 },
      departed: { session: '2026-09-21', return_pct: -2.2 },
    });
    expect(lastMove(st)).toMatchObject({ kind: 'window', to: 'pass' });
  });

  it('treats a flat newest session as not the cause', () => {
    const st = live1M({ entered: { session: '2026-10-19', return_pct: 0 } });
    expect(lastMove(st)?.kind).toBe('window');
  });

  it('is null when the last two windows agree', () => {
    const st = live1M({ windows: [win('2026-10-16', true), win('2026-10-19', true)] });
    expect(lastMove(st)).toBeNull();
  });

  it('stays quiet the session after a flip, when the verdict held at NO-GO (live 2026-10-23)', () => {
    const st = live1M({
      window_days: 22,
      passed: 8,
      sharpe_min: -0.65,
      sharpe_max: 3.11,
      windows: [win('2026-10-19', true, 1.69), win('2026-10-20', false, -0.65), win('2026-10-21', false, 0.86)],
      departed: { session: '2026-09-22', return_pct: -0.93 },
      entered: { session: '2026-10-21', return_pct: 0.16 },
    });
    expect(lastMove(st)).toBeNull();
    expect(stabilitySummary(st)).toBe('Son 10 pencereden 8 geçti · 1 dönüş · Sharpe −0.65…3.11');
  });

  it('reads a choppy month as half the windows, with no cause line when the newest move held (live 2026-11-03)', () => {
    const passes = [false, true, false, false, true, true, true, true, false, false];
    const ends = [
      '2026-10-20',
      '2026-10-21',
      '2026-10-22',
      '2026-10-23',
      '2026-10-26',
      '2026-10-27',
      '2026-10-28',
      '2026-10-29',
      '2026-10-30',
      '2026-11-02',
    ];
    const st = live1M({
      passed: 5,
      flips: 4,
      sharpe_min: -2.95,
      sharpe_max: 1.82,
      windows: ends.map((d, i) => win(d, passes[i]!)),
      departed: { session: '2026-10-05', return_pct: 0.8 },
      entered: { session: '2026-11-02', return_pct: -1.02 },
    });
    expect(stabilitySummary(st)).toBe('Son 10 pencereden 5 geçti · 4 dönüş · Sharpe −2.95…1.82');
    expect(stabilityA11y(st)).toBe('Pencere istikrarı: son 10 pencereden 5 geçti, 5 kaldı, 4 dönüş');
    expect(lastMove(st)).toBeNull();
  });

  it('is null when the backend could not name both sessions', () => {
    expect(lastMove(live1M({ departed: null }))).toBeNull();
    expect(lastMove(live1M({ entered: null }))).toBeNull();
  });

  it('is null with a single window or no block', () => {
    expect(lastMove(live1M({ windows: [win('2026-10-19', false)], total: 1 }))).toBeNull();
    expect(lastMove(undefined)).toBeNull();
  });
});
