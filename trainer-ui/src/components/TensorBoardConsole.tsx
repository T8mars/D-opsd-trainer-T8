'use client';

import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { Activity, LineChart, RefreshCw, SlidersHorizontal } from 'lucide-react';
import StatusPill from '@/components/StatusPill';
import { useI18n, type MessageKey } from '@/lib/i18n';
import type { TensorboardMetricKey, TensorboardRun, TensorboardScalarSeries } from '@/lib/tensorboard';

type RunsPayload = {
  ok: boolean;
  error?: string;
  runs?: TensorboardRun[];
};

type ScalarsPayload = {
  ok: boolean;
  error?: string;
  series?: TensorboardScalarSeries[];
};

const metricOptions: Array<{ key: TensorboardMetricKey; labelKey: MessageKey; color: string }> = [
  { key: 'lossTotal', labelKey: 'tensorboardMetricLossTotal', color: '#8ee8df' },
  { key: 'lossDopsd', labelKey: 'tensorboardMetricLossDopsd', color: '#7adf9b' },
  { key: 'gradNorm', labelKey: 'tensorboardMetricGradNorm', color: '#e9b85d' },
];

const runColors = ['#8ee8df', '#7adf9b', '#e9b85d', '#eb758a', '#c4ccd4', '#45d3c6'];

function formatValue(value?: number) {
  if (value == null || !Number.isFinite(value)) return '-';
  if (Math.abs(value) >= 1000) return value.toExponential(2);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(4);
}

function formatDate(value?: string) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function statusTone(status: string): 'good' | 'warn' | 'bad' | 'neutral' {
  if (status === 'completed') return 'good';
  if (status === 'running' || status === 'queued') return 'warn';
  if (status === 'failed') return 'bad';
  return 'neutral';
}

function trendLabelKey(trend?: TensorboardScalarSeries['trend']): MessageKey {
  if (trend === 'down') return 'tensorboardTrendDown';
  if (trend === 'up') return 'tensorboardTrendUp';
  if (trend === 'flat') return 'tensorboardTrendFlat';
  if (trend === 'volatile') return 'tensorboardTrendVolatile';
  return 'tensorboardTrendUnknown';
}

function metricLabelKey(metric: TensorboardMetricKey) {
  return metricOptions.find(option => option.key === metric)?.labelKey ?? 'loss';
}

function seriesColor(series: TensorboardScalarSeries, index: number) {
  const metricIndex = metricOptions.findIndex(option => option.key === series.metric);
  if (metricIndex >= 0 && index < metricOptions.length) return metricOptions[metricIndex].color;
  return runColors[index % runColors.length];
}

function buildPoints(series: TensorboardScalarSeries[], logScale: boolean) {
  const drawable = series.filter(item => item.points.length > 0);
  const steps = drawable.flatMap(item => item.points.map(point => point.step));
  const values = drawable.flatMap(item => item.points.map(point => point.value));
  const xMin = Math.min(...steps);
  const xMax = Math.max(...steps);
  const rawYMin = Math.min(...values);
  const rawYMax = Math.max(...values);
  const transformY = (value: number) => (logScale ? Math.log10(Math.max(value, 0.00000001)) : value);
  let yMin = transformY(rawYMin);
  let yMax = transformY(rawYMax);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    yMin = 0;
    yMax = 1;
  }
  if (Math.abs(yMax - yMin) < 0.0000001) {
    yMin -= 0.5;
    yMax += 0.5;
  }

  const width = 1000;
  const height = 360;
  const margin = { top: 26, right: 24, bottom: 48, left: 66 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xScale = (step: number) => margin.left + ((step - xMin) / Math.max(1, xMax - xMin)) * plotWidth;
  const yScale = (value: number) => margin.top + (1 - ((transformY(value) - yMin) / (yMax - yMin))) * plotHeight;
  const yTicks = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    const scaled = yMin + (yMax - yMin) * (1 - ratio);
    return {
      y: margin.top + ratio * plotHeight,
      label: logScale ? formatValue(10 ** scaled) : formatValue(scaled),
    };
  });
  const xTicks = Array.from({ length: 5 }, (_, index) => {
    const ratio = index / 4;
    return {
      x: margin.left + ratio * plotWidth,
      label: Math.round(xMin + (xMax - xMin) * ratio).toString(),
    };
  });

  return {
    drawable,
    width,
    height,
    margin,
    plotWidth,
    plotHeight,
    yTicks,
    xTicks,
    polylinePoints: (item: TensorboardScalarSeries) => item.points.map(point => `${xScale(point.step).toFixed(2)},${yScale(point.value).toFixed(2)}`).join(' '),
  };
}

function CurveChart({ series, logScale }: { series: TensorboardScalarSeries[]; logScale: boolean }) {
  const { t } = useI18n();
  const hasPoints = series.some(item => item.points.length > 0);
  const chart = useMemo(() => (hasPoints ? buildPoints(series, logScale) : null), [hasPoints, series, logScale]);

  if (!chart) {
    return (
      <div className="flex min-h-[360px] items-center justify-center rounded-lg border border-white/10 bg-black/20 text-sm text-ink-400">
        {t('tensorboardCurveEmpty')}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex items-center gap-2 text-sm font-medium text-white">
          <LineChart className="h-4 w-4 text-aqua-300" />
          {t('tensorboardChart')}
        </div>
        <div className="flex flex-wrap gap-3 text-xs text-ink-400">
          {chart.drawable.map((item, index) => (
            <span key={`${item.runId}-${item.metric}`} className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: seriesColor(item, index) }} />
              <span className="max-w-[14rem] truncate">{item.runName} · {t(metricLabelKey(item.metric))}</span>
            </span>
          ))}
        </div>
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${chart.width} ${chart.height}`} className="h-[360px] min-w-[760px] w-full" role="img" aria-label={t('tensorboardChart')}>
          <rect x={chart.margin.left} y={chart.margin.top} width={chart.plotWidth} height={chart.plotHeight} fill="rgba(255,255,255,0.025)" rx="8" />
          {chart.yTicks.map(tick => (
            <g key={`y-${tick.y}`}>
              <line x1={chart.margin.left} x2={chart.margin.left + chart.plotWidth} y1={tick.y} y2={tick.y} stroke="rgba(255,255,255,0.08)" />
              <text x={chart.margin.left - 10} y={tick.y + 4} textAnchor="end" className="fill-ink-400 text-[11px]">{tick.label}</text>
            </g>
          ))}
          {chart.xTicks.map(tick => (
            <g key={`x-${tick.x}`}>
              <line x1={tick.x} x2={tick.x} y1={chart.margin.top} y2={chart.margin.top + chart.plotHeight} stroke="rgba(255,255,255,0.05)" />
              <text x={tick.x} y={chart.margin.top + chart.plotHeight + 28} textAnchor="middle" className="fill-ink-400 text-[11px]">{tick.label}</text>
            </g>
          ))}
          <text x={chart.margin.left + chart.plotWidth / 2} y={chart.height - 8} textAnchor="middle" className="fill-ink-500 text-[11px]">{t('tensorboardXAxis')}</text>
          <text x={18} y={chart.margin.top + chart.plotHeight / 2} textAnchor="middle" transform={`rotate(-90 18 ${chart.margin.top + chart.plotHeight / 2})`} className="fill-ink-500 text-[11px]">{t('tensorboardYAxis')}</text>
          {chart.drawable.map((item, index) => (
            <polyline
              key={`${item.runId}-${item.metric}`}
              points={chart.polylinePoints(item)}
              fill="none"
              stroke={seriesColor(item, index)}
              strokeWidth="3"
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity="0.92"
            />
          ))}
        </svg>
      </div>
    </div>
  );
}

function ToggleButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`inline-flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition ${
        active
          ? 'border-aqua-500/35 bg-aqua-500/[0.14] text-aqua-300'
          : 'border-white/10 bg-white/[0.05] text-ink-300 hover:bg-white/[0.09]'
      }`}
    >
      {children}
    </button>
  );
}

export default function TensorBoardConsole() {
  const { t } = useI18n();
  const [runs, setRuns] = useState<TensorboardRun[]>([]);
  const [series, setSeries] = useState<TensorboardScalarSeries[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<TensorboardMetricKey[]>(['lossTotal']);
  const [smoothing, setSmoothing] = useState(0.2);
  const [logScale, setLogScale] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [limit, setLimit] = useState(1600);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadRuns = useCallback(async () => {
    const response = await fetch('/api/tensorboard/runs', { cache: 'no-store' });
    const payload = (await response.json()) as RunsPayload;
    if (!response.ok || !payload.ok) throw new Error(payload.error || t('tensorboardLoadFailed'));
    const nextRuns = payload.runs ?? [];
    setRuns(nextRuns);
    setSelectedRunIds(current => {
      const available = new Set(nextRuns.map(run => run.id));
      const kept = current.filter(id => available.has(id));
      if (kept.length) return kept;
      return nextRuns.slice(0, 2).map(run => run.id);
    });
  }, [t]);

  const loadScalars = useCallback(async () => {
    if (!selectedRunIds.length || !selectedMetrics.length) {
      setSeries([]);
      return;
    }
    const params = new URLSearchParams({
      runIds: selectedRunIds.join(','),
      metrics: selectedMetrics.join(','),
      limit: String(limit),
      stride: '1',
      smooth: String(smoothing),
    });
    const response = await fetch(`/api/tensorboard/scalars?${params.toString()}`, { cache: 'no-store' });
    const payload = (await response.json()) as ScalarsPayload;
    if (!response.ok || !payload.ok) throw new Error(payload.error || t('tensorboardLoadFailed'));
    setSeries(payload.series ?? []);
  }, [limit, selectedMetrics, selectedRunIds, smoothing, t]);

  const refreshAll = useCallback(async () => {
    setError(null);
    try {
      await loadRuns();
    } catch (refreshError: any) {
      setError(String(refreshError?.message || refreshError));
    } finally {
      setLoading(false);
    }
  }, [loadRuns]);

  useEffect(() => {
    void refreshAll();
  }, [refreshAll]);

  useEffect(() => {
    if (loading) return;
    setError(null);
    void loadScalars().catch((loadError: any) => setError(String(loadError?.message || loadError)));
  }, [loadScalars, loading]);

  useEffect(() => {
    if (!autoRefresh) return undefined;
    const interval = window.setInterval(() => {
      void refreshAll();
      void loadScalars().catch((loadError: any) => setError(String(loadError?.message || loadError)));
    }, 5000);
    return () => window.clearInterval(interval);
  }, [autoRefresh, loadScalars, refreshAll]);

  const selectedRuns = useMemo(() => runs.filter(run => selectedRunIds.includes(run.id)), [runs, selectedRunIds]);
  const visiblePointCount = useMemo(() => series.reduce((sum, item) => sum + item.points.length, 0), [series]);
  const latestStep = useMemo(() => selectedRuns.reduce((max, run) => Math.max(max, run.latestStep), 0), [selectedRuns]);
  const latestLoss = selectedRuns.find(run => run.latestLoss != null)?.latestLoss;
  const bestLoss = selectedRuns.map(run => run.bestLoss).filter((value): value is number => value != null).reduce((best, value) => Math.min(best, value), Number.POSITIVE_INFINITY);
  const eventFiles = selectedRuns.reduce((sum, run) => sum + run.eventFiles, 0);
  const primaryTrend = series.find(item => item.metric === 'lossTotal')?.trend ?? series[0]?.trend;

  function toggleRun(id: string) {
    setSelectedRunIds(current => current.includes(id) ? current.filter(item => item !== id) : [...current, id]);
  }

  function toggleMetric(metric: TensorboardMetricKey) {
    setSelectedMetrics(current => current.includes(metric) ? current.filter(item => item !== metric) : [...current, metric]);
  }

  if (loading) {
    return <div className="solid-panel rounded-lg p-4 text-sm text-ink-400">{t('tensorboardLoading')}</div>;
  }

  return (
    <div className="space-y-4">
      <section className="glass rounded-lg p-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-medium text-white">
            <Activity className="h-4 w-4 text-aqua-300" />
            {t('tensorboardWorkbench')}
            <StatusPill label={autoRefresh ? t('liveRefresh') : t('idle')} tone={autoRefresh ? 'warn' : 'neutral'} />
          </div>
          <button
            type="button"
            onClick={() => {
              void refreshAll();
              void loadScalars().catch((loadError: any) => setError(String(loadError?.message || loadError)));
            }}
            className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.06] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1]"
          >
            <RefreshCw className="h-4 w-4" />
            {t('tensorboardRefresh')}
          </button>
        </div>
      </section>

      {error ? (
        <section className="rounded-lg border border-roseSoft-500/25 bg-roseSoft-500/[0.08] p-3 text-sm text-roseSoft-500">
          {error}
        </section>
      ) : null}

      <section className="grid gap-3 md:grid-cols-5">
        {[
          [t('tensorboardLatestLoss'), formatValue(latestLoss)],
          [t('tensorboardBestLoss'), Number.isFinite(bestLoss) ? formatValue(bestLoss) : '-'],
          [t('tensorboardLatestStep'), latestStep.toLocaleString()],
          [t('tensorboardPoints'), visiblePointCount.toLocaleString()],
          [t('tensorboardEventFiles'), eventFiles.toLocaleString()],
        ].map(([label, value]) => (
          <div key={label} className="solid-panel rounded-lg p-4">
            <div className="text-xs uppercase tracking-wide text-ink-400">{label}</div>
            <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
          </div>
        ))}
      </section>

      <section className="grid gap-4 xl:grid-cols-[22rem_1fr]">
        <div className="space-y-4">
          <div className="solid-panel rounded-lg p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium text-white">
              <LineChart className="h-4 w-4 text-aqua-300" />
              {t('tensorboardRunSelection')}
            </div>
            {runs.length ? (
              <div className="max-h-[32rem] space-y-2 overflow-auto pr-1">
                {runs.map(run => (
                  <label key={run.id} className="block rounded-md border border-white/10 bg-white/[0.04] p-3 transition hover:bg-white/[0.07]">
                    <div className="flex items-start gap-3">
                      <input
                        type="checkbox"
                        checked={selectedRunIds.includes(run.id)}
                        onChange={() => toggleRun(run.id)}
                        className="mt-1 h-4 w-4 accent-[#45d3c6]"
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex min-w-0 flex-wrap items-center gap-2">
                          <span className="truncate text-sm font-medium text-white">{run.name}</span>
                          <StatusPill label={run.status} tone={statusTone(run.status)} />
                        </div>
                        <div className="mt-1 text-xs text-ink-400">{run.recipeName || run.source}</div>
                        <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
                          <div className="rounded-md bg-black/20 p-2">
                            <div className="text-ink-500">{t('tensorboardLatestStep')}</div>
                            <div className="mt-1 text-ink-100">{run.latestStep}</div>
                          </div>
                          <div className="rounded-md bg-black/20 p-2">
                            <div className="text-ink-500">{t('tensorboardLatestLoss')}</div>
                            <div className="mt-1 text-ink-100">{formatValue(run.latestLoss)}</div>
                          </div>
                          <div className="rounded-md bg-black/20 p-2">
                            <div className="text-ink-500">{t('tensorboardEvents')}</div>
                            <div className="mt-1 text-ink-100">{run.eventFiles}</div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </label>
                ))}
              </div>
            ) : (
              <div className="rounded-md border border-white/10 bg-black/20 p-4 text-sm text-ink-400">{t('tensorboardNoRuns')}</div>
            )}
          </div>

          <div className="solid-panel rounded-lg p-4">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium text-white">
              <SlidersHorizontal className="h-4 w-4 text-aqua-300" />
              {t('tensorboardMetricSelection')}
            </div>
            <div className="flex flex-wrap gap-2">
              {metricOptions.map(option => (
                <ToggleButton key={option.key} active={selectedMetrics.includes(option.key)} onClick={() => toggleMetric(option.key)}>
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: option.color }} />
                  {t(option.labelKey)}
                </ToggleButton>
              ))}
            </div>

            <div className="mt-4 space-y-4">
              <label className="block">
                <div className="mb-2 flex items-center justify-between text-xs text-ink-400">
                  <span>{t('tensorboardSmoothing')}</span>
                  <span>{smoothing.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="0.9"
                  step="0.05"
                  value={smoothing}
                  onChange={event => setSmoothing(Number(event.target.value))}
                  className="w-full accent-[#45d3c6]"
                />
              </label>

              <div className="grid gap-2 sm:grid-cols-2">
                <label className="flex items-center justify-between gap-3 rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-ink-300">
                  <span>{t('tensorboardLogScale')}</span>
                  <input type="checkbox" checked={logScale} onChange={event => setLogScale(event.target.checked)} className="h-4 w-4 accent-[#45d3c6]" />
                </label>
                <label className="flex items-center justify-between gap-3 rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-ink-300">
                  <span>{t('tensorboardAutoRefresh')}</span>
                  <input type="checkbox" checked={autoRefresh} onChange={event => setAutoRefresh(event.target.checked)} className="h-4 w-4 accent-[#45d3c6]" />
                </label>
              </div>

              <label className="block">
                <div className="mb-2 text-xs text-ink-400">{t('tensorboardLimit')}</div>
                <select
                  value={limit}
                  onChange={event => setLimit(Number(event.target.value))}
                  className="w-full rounded-md border border-white/10 bg-ink-900 px-3 py-2 text-sm text-ink-100 outline-none focus:border-aqua-500/50"
                >
                  <option value={400}>400</option>
                  <option value={800}>800</option>
                  <option value={1600}>1600</option>
                  <option value={3200}>3200</option>
                </select>
              </label>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {!selectedRunIds.length ? (
            <div className="solid-panel rounded-lg p-4 text-sm text-amberSoft-500">{t('tensorboardSelectAtLeastOneRun')}</div>
          ) : !selectedMetrics.length ? (
            <div className="solid-panel rounded-lg p-4 text-sm text-amberSoft-500">{t('tensorboardSelectAtLeastOneMetric')}</div>
          ) : (
            <CurveChart series={series} logScale={logScale} />
          )}

          <div className="solid-panel rounded-lg p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <div className="text-sm font-medium text-white">{t('tensorboardCompareRuns')}</div>
              <StatusPill label={`${t('tensorboardTrend')}: ${t(trendLabelKey(primaryTrend))}`} tone={primaryTrend === 'down' ? 'good' : primaryTrend === 'up' ? 'warn' : 'neutral'} />
            </div>
            <div className="overflow-x-auto">
              <table className="min-w-[760px] w-full text-left text-xs">
                <thead className="text-ink-500">
                  <tr>
                    <th className="px-2 py-2 font-medium">{t('tensorboardRuns')}</th>
                    <th className="px-2 py-2 font-medium">{t('tensorboardStatus')}</th>
                    <th className="px-2 py-2 font-medium">{t('tensorboardLatestStep')}</th>
                    <th className="px-2 py-2 font-medium">{t('tensorboardLatestLoss')}</th>
                    <th className="px-2 py-2 font-medium">{t('tensorboardJsonl')}</th>
                    <th className="px-2 py-2 font-medium">{t('tensorboardUpdated')}</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedRuns.map(run => (
                    <tr key={run.id} className="border-t border-white/10 text-ink-300">
                      <td className="max-w-[16rem] truncate px-2 py-2 text-white">{run.name}</td>
                      <td className="px-2 py-2"><StatusPill label={run.status} tone={statusTone(run.status)} /></td>
                      <td className="px-2 py-2 font-mono">{run.latestStep}</td>
                      <td className="px-2 py-2 font-mono">{formatValue(run.latestLoss)}</td>
                      <td className="max-w-[16rem] truncate px-2 py-2 font-mono text-ink-400">{run.lossPath}</td>
                      <td className="px-2 py-2">{formatDate(run.updatedAt)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
