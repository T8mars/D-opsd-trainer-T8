'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Cpu, RefreshCw, Thermometer, Zap } from 'lucide-react';
import StatusPill from '@/components/StatusPill';
import { useI18n } from '@/lib/i18n';
import type { TelemetryPayload } from '@/lib/system';

const emptyPayload: TelemetryPayload = {
  ok: false,
  generatedAt: '',
  gpu: { available: false, gpus: [] },
};

function formatMb(value?: number) {
  if (value == null) return '-';
  if (value >= 1024) return `${(value / 1024).toFixed(1)} GB`;
  return `${Math.round(value)} MB`;
}

function formatPercent(value?: number) {
  if (value == null) return '-';
  return `${Math.round(value)}%`;
}

function formatWatts(value?: number, limit?: number) {
  if (value == null) return '-';
  return limit == null ? `${value.toFixed(0)} W` : `${value.toFixed(0)} / ${limit.toFixed(0)} W`;
}

function memoryTone(usedPercent: number): 'good' | 'warn' | 'bad' {
  if (usedPercent >= 92) return 'bad';
  if (usedPercent >= 80) return 'warn';
  return 'good';
}

function basename(processName: string) {
  return processName.split(/[\\/]/).filter(Boolean).pop() || processName;
}

export default function GpuTelemetryPanel({ compact = false }: { compact?: boolean }) {
  const [payload, setPayload] = useState<TelemetryPayload>(emptyPayload);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { t } = useI18n();

  const loadTelemetry = useCallback(async (showSpinner = false) => {
    if (showSpinner) setRefreshing(true);
    try {
      const response = await fetch('/api/telemetry', { cache: 'no-store' });
      const nextPayload = (await response.json()) as TelemetryPayload;
      setPayload(nextPayload);
    } catch {
      setPayload(emptyPayload);
    } finally {
      setLoading(false);
      if (showSpinner) setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadTelemetry();
    const interval = window.setInterval(() => {
      void loadTelemetry();
    }, 5000);
    return () => window.clearInterval(interval);
  }, [loadTelemetry]);

  const primaryGpu = payload.gpu.gpus[0];
  const generatedLabel = useMemo(() => {
    if (!payload.generatedAt) return t('waiting');
    return new Intl.DateTimeFormat(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    }).format(new Date(payload.generatedAt));
  }, [payload.generatedAt, t]);

  if (!payload.gpu.available && !loading) {
    return (
      <section className="glass rounded-lg p-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-medium text-white">
            <Cpu className="h-4 w-4 text-aqua-300" />
            {t('gpuTelemetry')}
          </div>
          <StatusPill label={t('offline')} tone="bad" />
        </div>
        <p className="mt-3 text-sm leading-5 text-ink-400">{payload.gpu.error || t('gpuDetailMissing')}</p>
      </section>
    );
  }

  const usedPercent = primaryGpu && primaryGpu.memoryTotalMb > 0 ? (primaryGpu.memoryUsedMb / primaryGpu.memoryTotalMb) * 100 : 0;
  const tone = memoryTone(usedPercent);
  const processes = primaryGpu?.processes.slice(0, compact ? 3 : 5) ?? [];

  return (
    <section className="glass rounded-lg p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-medium text-white">
          <Cpu className="h-4 w-4 text-aqua-300" />
          {t('gpuTelemetry')}
          <StatusPill label={loading ? t('checking') : t('live')} tone={loading ? 'neutral' : tone} />
        </div>
        <button
          className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.06] px-2.5 py-1.5 text-xs text-ink-300 transition hover:bg-white/[0.1] disabled:opacity-50"
          disabled={refreshing}
          onClick={() => void loadTelemetry(true)}
        >
          <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? 'animate-spin' : ''}`} />
          {generatedLabel}
        </button>
      </div>

      {primaryGpu ? (
        <div className="space-y-4">
            <div className="min-w-0">
              <div className="truncate text-sm font-medium text-white">{primaryGpu.name}</div>
            <div className="mt-1 text-xs text-ink-400">{t('driver')} {primaryGpu.driver} · GPU {primaryGpu.index}</div>
          </div>

          <div>
            <div className="mb-1 flex items-center justify-between text-xs text-ink-400">
              <span>VRAM</span>
              <span>{formatMb(primaryGpu.memoryUsedMb)} / {formatMb(primaryGpu.memoryTotalMb)}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-black/30">
              <div
                className={`h-full rounded-full ${tone === 'bad' ? 'bg-roseSoft-500' : tone === 'warn' ? 'bg-amberSoft-500' : 'bg-aqua-300'}`}
                style={{ width: `${Math.min(100, Math.max(0, usedPercent))}%` }}
              />
            </div>
            <div className="mt-1 text-xs text-ink-400">{formatMb(primaryGpu.memoryFreeMb)} {t('free')}</div>
          </div>

          <div className="grid gap-2 sm:grid-cols-3">
            <div className="rounded-md bg-white/[0.04] p-2">
              <div className="text-xs text-ink-400">{t('util')}</div>
              <div className="mt-1 text-sm text-white">{formatPercent(primaryGpu.utilizationGpuPercent)}</div>
            </div>
            <div className="rounded-md bg-white/[0.04] p-2">
              <div className="flex items-center gap-1 text-xs text-ink-400">
                <Thermometer className="h-3.5 w-3.5" />
                {t('temp')}
              </div>
              <div className="mt-1 text-sm text-white">{primaryGpu.temperatureC == null ? '-' : `${primaryGpu.temperatureC} C`}</div>
            </div>
            <div className="rounded-md bg-white/[0.04] p-2">
              <div className="flex items-center gap-1 text-xs text-ink-400">
                <Zap className="h-3.5 w-3.5" />
                {t('power')}
              </div>
              <div className="mt-1 text-sm text-white">{formatWatts(primaryGpu.powerDrawW, primaryGpu.powerLimitW)}</div>
            </div>
          </div>

          <div className="rounded-md border border-white/10 bg-black/20 p-3">
            <div className="mb-2 flex items-center justify-between text-xs text-ink-400">
              <span>{t('processes')}</span>
              <span>{primaryGpu.processes.length}</span>
            </div>
            {processes.length ? (
              <div className="space-y-1">
                {processes.map(processInfo => (
                  <div key={`${processInfo.pid}-${processInfo.name}`} className="flex min-w-0 items-center justify-between gap-3 text-xs">
                    <span className="truncate text-ink-300">{basename(processInfo.name)}</span>
                    <span className="shrink-0 text-ink-400">{processInfo.usedMemoryMb == null ? `PID ${processInfo.pid}` : formatMb(processInfo.usedMemoryMb)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-ink-400">{t('noComputeProcesses')}</div>
            )}
          </div>
        </div>
      ) : (
        <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm text-ink-400">{t('checkingGpu')}</div>
      )}
    </section>
  );
}
