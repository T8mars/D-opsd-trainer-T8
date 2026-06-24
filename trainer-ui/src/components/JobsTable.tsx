'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Activity, AlertTriangle, Archive, Copy, FileText, FolderOpen, LineChart, Play, Plus, RefreshCw, ScrollText, Square, Trash2 } from 'lucide-react';
import StatusPill from '@/components/StatusPill';
import { useI18n, type MessageKey } from '@/lib/i18n';
import type { ArtifactEntry, JobSummary, JobStatus } from '@/lib/jobs';
import type { RecipeId } from '@/lib/recipes';

type JobsPayload = {
  ok: boolean;
  generatedAt: string;
  jobs: JobSummary[];
};

type ActionPayload = {
  ok: boolean;
  error?: string;
};

type LogLine = {
  source: 'training' | 'runner' | 'stderr';
  line: string;
};

type LogsPayload = {
  ok: boolean;
  error?: string;
  generatedAt?: string;
  truncated?: boolean;
  combined?: LogLine[];
  loading?: boolean;
  streaming?: boolean;
  streamError?: string;
};

const statusTone: Record<JobStatus, 'good' | 'warn' | 'bad' | 'neutral'> = {
  draft: 'neutral',
  queued: 'warn',
  running: 'warn',
  completed: 'good',
  failed: 'bad',
  stopped: 'neutral',
};

const statusLabelKeys: Record<JobStatus, MessageKey> = {
  draft: 'draft',
  queued: 'queued',
  running: 'running',
  completed: 'completed',
  failed: 'failed',
  stopped: 'stopped',
};

function formatLoss(value?: number) {
  if (value == null) return '-';
  return value.toFixed(4);
}

function artifactLabel(enabled: boolean, label: string) {
  return (
    <span
      className={`rounded-md border px-2 py-1 text-xs ${
        enabled ? 'border-mint-500/25 bg-mint-500/[0.08] text-mint-500' : 'border-white/10 bg-white/[0.04] text-ink-400'
      }`}
    >
      {label}
    </span>
  );
}

function formatBytes(bytes: number) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
}

function artifactUrl(job: JobSummary, artifact: ArtifactEntry) {
  return `/api/jobs/${job.id}/artifact?path=${encodeURIComponent(artifact.relativePath)}`;
}

function ArtifactBrowser({ job }: { job: JobSummary }) {
  const sampleImages = [...job.artifactItems.samples, ...job.artifactItems.sampleTrajectories]
    .filter(item => item.isImage)
    .slice(0, 3);
  const checkpoints = job.artifactItems.checkpoints.slice(0, 3);
  const hasArtifacts = sampleImages.length > 0 || checkpoints.length > 0;
  const { t } = useI18n();

  return (
    <div className="rounded-md border border-white/10 bg-black/20 p-2">
      <div className="mb-2 flex items-center justify-between gap-2 text-xs text-ink-400">
        <span className="inline-flex items-center gap-1.5">
          <Archive className="h-3.5 w-3.5" />
          {t('artifactFiles')}
        </span>
        <span>
          {job.artifactCounts.samples + job.artifactCounts.sampleTrajectories} {t('samplesShort')} · {job.artifactCounts.checkpoints} {t('ckptShort')}
        </span>
      </div>

      {hasArtifacts ? (
        <div className="space-y-2">
          {sampleImages.length > 0 ? (
            <div className="grid grid-cols-3 gap-2">
              {sampleImages.map(item => (
                <a
                  key={item.relativePath}
                  href={artifactUrl(job, item)}
                  target="_blank"
                  className="group block overflow-hidden rounded-md border border-white/10 bg-white/[0.04]"
                >
                  <img
                    src={artifactUrl(job, item)}
                    alt={item.name}
                    loading="lazy"
                    className="aspect-square w-full object-cover transition group-hover:scale-[1.03]"
                  />
                </a>
              ))}
            </div>
          ) : null}

          {checkpoints.length > 0 ? (
            <div className="space-y-1">
              {checkpoints.map(item => (
                <a
                  key={item.relativePath}
                  href={artifactUrl(job, item)}
                  target="_blank"
                  className="flex min-w-0 items-center justify-between gap-2 rounded-md bg-white/[0.04] px-2 py-1.5 text-xs text-ink-300 hover:bg-white/[0.08]"
                >
                  <span className="min-w-0 truncate">{item.name}</span>
                  <span className="shrink-0 text-ink-500">{formatBytes(item.sizeBytes)}</span>
                </a>
              ))}
            </div>
          ) : null}
        </div>
      ) : (
        <div className="flex h-16 items-center justify-center rounded-md bg-white/[0.03] text-xs text-ink-500">
          {t('noSampleCheckpointFiles')}
        </div>
      )}
    </div>
  );
}

function LossBars({ job }: { job: JobSummary }) {
  if (!job.lossHistory.length) {
    return <div className="h-10 rounded-md border border-white/10 bg-black/20" />;
  }

  const values = job.lossHistory.map(point => point.lossTotal ?? 0);
  const max = Math.max(...values, 0.001);
  return (
    <div className="flex h-10 items-end gap-1 rounded-md border border-white/10 bg-black/20 p-1.5">
      {values.slice(-16).map((value, index) => (
        <div
          key={`${job.id}-${index}`}
          className="min-w-1 flex-1 rounded-sm bg-aqua-300/80"
          style={{ height: `${Math.max(15, (value / max) * 100)}%` }}
          title={`${value.toFixed(4)}`}
        />
      ))}
    </div>
  );
}

function LossHistorySummary({ job }: { job: JobSummary }) {
  const { t } = useI18n();
  const recent = job.lossHistory.slice(-5).reverse();
  if (!recent.length) {
    return null;
  }

  return (
    <div className="rounded-md border border-white/10 bg-black/20 p-2">
      <div className="mb-2 text-xs font-medium text-ink-300">{t('recentLoss')}</div>
      <div className="grid grid-cols-[0.6fr_1fr_1fr_1fr] gap-2 text-[11px] text-ink-500">
        <span>{t('step')}</span>
        <span>{t('loss')}</span>
        <span>{t('dopsdLoss')}</span>
        <span>{t('gradNorm')}</span>
      </div>
      <div className="mt-1 space-y-1">
        {recent.map(point => (
          <div key={`${job.id}-loss-${point.step}`} className="grid grid-cols-[0.6fr_1fr_1fr_1fr] gap-2 rounded-sm bg-white/[0.035] px-1.5 py-1 font-mono text-[11px] text-ink-300">
            <span>{point.step}</span>
            <span>{formatLoss(point.lossTotal)}</span>
            <span>{formatLoss(point.lossDopsd)}</span>
            <span>{formatLoss(point.gradNorm)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function FailureDiagnostics({ job }: { job: JobSummary }) {
  if (job.status !== 'failed' && !job.failureReason && !job.errorTail.length) return null;

  const lines = (job.errorTail.length ? job.errorTail : job.runnerLogTail).slice(-4);
  const { t } = useI18n();
  return (
    <div className="mt-2 rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.08] p-2">
      <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-roseSoft-500">
        <AlertTriangle className="h-3.5 w-3.5" />
        {t('failure')}
      </div>
      {job.failureReason ? <div className="text-xs leading-5 text-roseSoft-200">{job.failureReason}</div> : null}
      {lines.length ? (
        <div className="mt-2 space-y-1 font-mono text-[11px] leading-4 text-ink-300">
          {lines.map((line, index) => (
            <div key={`${job.id}-failure-${index}`} className="truncate">
              {line}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function parseLogStreamEvent<T>(event: MessageEvent) {
  try {
    return JSON.parse(event.data) as T;
  } catch {
    return null;
  }
}

function mergeStreamedLogLines(payload: LogsPayload | undefined, lines: LogLine[], generatedAt?: string, truncated?: boolean): LogsPayload {
  const combined = [...(payload?.combined ?? []), ...lines].slice(-300);
  return {
    ...(payload ?? { ok: true }),
    ok: true,
    loading: false,
    generatedAt: generatedAt ?? payload?.generatedAt,
    truncated: truncated ?? payload?.truncated,
    combined,
    streaming: true,
    streamError: undefined,
  };
}

function LogPanel({ payload }: { payload?: LogsPayload }) {
  const lines = payload?.combined ?? [];
  const { t } = useI18n();
  return (
    <div className="mt-3 rounded-md border border-white/10 bg-black/30 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-400">
        <div className="inline-flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 text-ink-300">
            <ScrollText className="h-3.5 w-3.5" />
            {t('fullLogs')}
          </span>
          {payload?.streaming ? <StatusPill label={t('liveStream')} tone={payload.streamError ? 'bad' : 'warn'} /> : null}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {payload?.streamError ? <span className="text-roseSoft-500">{payload.streamError}</span> : null}
          {payload?.truncated ? <span>{t('last300Lines')}</span> : null}
        </div>
      </div>
      {payload?.loading ? (
        <div className="text-xs text-ink-500">{t('loadingLogs')}</div>
      ) : payload?.error ? (
        <div className="text-xs text-roseSoft-500">{payload.error}</div>
      ) : lines.length ? (
        <div className="max-h-72 overflow-auto rounded-md bg-black/35 p-2 font-mono text-[11px] leading-5 text-ink-300">
          {lines.map((entry, index) => (
            <div key={`${entry.source}-${index}`} className="grid min-w-[720px] grid-cols-[72px_1fr] gap-2">
              <span className={entry.source === 'stderr' ? 'text-roseSoft-500' : entry.source === 'runner' ? 'text-aqua-300' : 'text-ink-500'}>
                {entry.source}
              </span>
              <span className="whitespace-pre-wrap break-words">{entry.line}</span>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-xs text-ink-500">{t('noLogsWrittenYet')}</div>
      )}
    </div>
  );
}

export default function JobsTable() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [openLogIds, setOpenLogIds] = useState<string[]>([]);
  const [logsByJob, setLogsByJob] = useState<Record<string, LogsPayload>>({});
  const { t } = useI18n();

  const loadJobs = useCallback(async (showSpinner = true) => {
    if (showSpinner) setLoading(true);
    const response = await fetch('/api/jobs', { cache: 'no-store' });
    const payload = (await response.json()) as JobsPayload;
    setJobs(payload.jobs ?? []);
    if (showSpinner) setLoading(false);
  }, []);

  useEffect(() => {
    void loadJobs();
  }, [loadJobs]);

  const counts = useMemo(() => {
    return jobs.reduce(
      (acc, job) => {
        acc.total += 1;
        acc[job.status] += 1;
        return acc;
      },
      { total: 0, draft: 0, queued: 0, running: 0, completed: 0, failed: 0, stopped: 0 } as Record<JobStatus | 'total', number>,
    );
  }, [jobs]);
  const hasActiveJobs = counts.running > 0 || counts.queued > 0;
  const liveLogJobKey = useMemo(() => {
    return jobs
      .filter(job => openLogIds.includes(job.id) && (job.status === 'running' || job.status === 'queued'))
      .map(job => job.id)
      .sort()
      .join('|');
  }, [jobs, openLogIds]);

  useEffect(() => {
    if (!hasActiveJobs) return undefined;
    const interval = window.setInterval(() => {
      void loadJobs(false);
    }, 5000);
    return () => window.clearInterval(interval);
  }, [hasActiveJobs, loadJobs]);

  useEffect(() => {
    if (!liveLogJobKey) return undefined;

    const liveIds = new Set(liveLogJobKey.split('|').filter(Boolean));
    const liveJobs = jobs.filter(job => liveIds.has(job.id));
    const eventSources = liveJobs.map(job => {
      const eventSource = new EventSource(`/api/jobs/${job.id}/logs/stream`);

      setLogsByJob(current => ({
        ...current,
        [job.id]: {
          ...(current[job.id] ?? { ok: true }),
          ok: true,
          loading: current[job.id]?.loading ?? false,
          streaming: true,
          streamError: undefined,
        },
      }));

      eventSource.addEventListener('snapshot', event => {
        const payload = parseLogStreamEvent<{ generatedAt?: string; truncated?: boolean; combined?: LogLine[] }>(event);
        if (!payload) return;
        setLogsByJob(current => ({
          ...current,
          [job.id]: {
            ...(current[job.id] ?? { ok: true }),
            ok: true,
            loading: false,
            generatedAt: payload.generatedAt ?? current[job.id]?.generatedAt,
            truncated: payload.truncated,
            combined: payload.combined ?? [],
            streaming: true,
            streamError: undefined,
          },
        }));
      });

      eventSource.addEventListener('append', event => {
        const payload = parseLogStreamEvent<{ generatedAt?: string; truncated?: boolean; lines?: LogLine[] }>(event);
        if (!payload?.lines?.length) return;
        setLogsByJob(current => ({
          ...current,
          [job.id]: mergeStreamedLogLines(current[job.id], payload.lines ?? [], payload.generatedAt, payload.truncated),
        }));
      });

      eventSource.addEventListener('heartbeat', event => {
        const payload = parseLogStreamEvent<{ generatedAt?: string; truncated?: boolean }>(event);
        setLogsByJob(current => ({
          ...current,
          [job.id]: {
            ...(current[job.id] ?? { ok: true }),
            ok: true,
            loading: false,
            generatedAt: payload?.generatedAt ?? current[job.id]?.generatedAt,
            truncated: payload?.truncated ?? current[job.id]?.truncated,
            streaming: true,
            streamError: undefined,
          },
        }));
      });

      eventSource.addEventListener('error', event => {
        const payload = parseLogStreamEvent<{ error?: string }>(event as MessageEvent);
        if (!payload?.error) return;
        setLogsByJob(current => ({
          ...current,
          [job.id]: {
            ...(current[job.id] ?? { ok: false }),
            ok: false,
            loading: false,
            streaming: false,
            streamError: payload.error,
            error: payload.error,
          },
        }));
      });

      eventSource.onerror = () => {
        setLogsByJob(current => ({
          ...current,
          [job.id]: {
            ...(current[job.id] ?? { ok: false }),
            loading: false,
            streaming: true,
            streamError: t('logStreamReconnecting'),
          },
        }));
      };

      return eventSource;
    });

    return () => {
      eventSources.forEach(eventSource => eventSource.close());
      setLogsByJob(current => {
        const next = { ...current };
        for (const id of liveIds) {
          if (next[id]) next[id] = { ...next[id], streaming: false };
        }
        return next;
      });
    };
  }, [liveLogJobKey, jobs, t]);

  async function createDraft(recipeId: RecipeId) {
    setBusy(`create-${recipeId}`);
    await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ recipeId }),
    });
    await loadJobs();
    setBusy(null);
  }

  async function clone(job: JobSummary) {
    setBusy(`clone-${job.id}`);
    await fetch(`/api/jobs/${job.id}/clone`, { method: 'POST' });
    await loadJobs();
    setBusy(null);
  }

  async function start(job: JobSummary) {
    setBusy(`start-${job.id}`);
    await fetch(`/api/jobs/${job.id}/start`, { method: 'POST' });
    await loadJobs();
    setBusy(null);
  }

  async function stop(job: JobSummary) {
    setBusy(`stop-${job.id}`);
    await fetch(`/api/jobs/${job.id}/stop`, { method: 'POST' });
    await loadJobs();
    setBusy(null);
  }

  async function openOutput(job: JobSummary) {
    setBusy(`open-${job.id}`);
    try {
      const response = await fetch(`/api/jobs/${job.id}/open`, { method: 'POST' });
      const payload = (await response.json()) as ActionPayload;
      if (!response.ok || !payload.ok) {
        window.alert(payload.error || t('outputFolderCouldNotBeOpened'));
      }
    } finally {
      setBusy(null);
    }
  }

  async function toggleLogs(job: JobSummary) {
    if (openLogIds.includes(job.id)) {
      setOpenLogIds(ids => ids.filter(id => id !== job.id));
      return;
    }

    setOpenLogIds(ids => [...ids, job.id]);
    if (logsByJob[job.id]?.combined?.length) return;

    setLogsByJob(current => ({ ...current, [job.id]: { ok: true, loading: true } }));
    try {
      const response = await fetch(`/api/jobs/${job.id}/logs`, { cache: 'no-store' });
      const payload = (await response.json()) as LogsPayload;
      setLogsByJob(current => ({ ...current, [job.id]: payload.ok ? payload : { ok: false, error: payload.error || t('logsCouldNotBeLoaded') } }));
    } catch {
      setLogsByJob(current => ({ ...current, [job.id]: { ok: false, error: t('logsCouldNotBeLoaded') } }));
    }
  }

  async function remove(job: JobSummary) {
    if (!window.confirm(`${t('deleteJobConfirm')} ${job.name}?`)) return;
    setBusy(`delete-${job.id}`);
    await fetch(`/api/jobs/${job.id}`, { method: 'DELETE' });
    await loadJobs();
    setBusy(null);
  }

  return (
    <div className="space-y-4">
      <section className="grid gap-3 md:grid-cols-4">
        {[
          [t('total'), counts.total],
          [t('queued'), counts.queued],
          [t('running'), counts.running],
          [t('completed'), counts.completed],
        ].map(([label, value]) => (
          <div key={label} className="solid-panel rounded-lg p-4">
            <div className="text-xs uppercase tracking-wide text-ink-400">{label}</div>
            <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
          </div>
        ))}
      </section>

      <section className="glass rounded-lg p-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm font-medium text-white">
            <Activity className="h-4 w-4 text-aqua-300" />
            {t('jobLedger')}
            <StatusPill label={hasActiveJobs ? t('liveRefresh') : t('idle')} tone={hasActiveJobs ? 'warn' : 'neutral'} />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              className="inline-flex items-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.12] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/20 disabled:opacity-50"
              disabled={busy === 'create-flux2-klein-identity'}
              onClick={() => void createDraft('flux2-klein-identity')}
            >
              <Plus className="h-4 w-4" />
              {t('flux2Draft')}
            </button>
            <button
              className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.06] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1] disabled:opacity-50"
              disabled={busy === 'create-z-image-turbo-vlm'}
              onClick={() => void createDraft('z-image-turbo-vlm')}
            >
              <Plus className="h-4 w-4" />
              {t('zImageDraft')}
            </button>
            <button
              className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.06] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1]"
              onClick={() => void loadJobs()}
            >
              <RefreshCw className="h-4 w-4" />
              {t('refresh')}
            </button>
          </div>
        </div>
      </section>

      <section className="space-y-3">
        {loading ? (
          <div className="solid-panel rounded-lg p-4 text-sm text-ink-400">{t('loadingJobs')}</div>
        ) : jobs.length === 0 ? (
          <div className="solid-panel rounded-lg p-4 text-sm text-ink-400">{t('noJobsYet')}</div>
        ) : (
          jobs.map(job => (
            <article key={job.id} className="solid-panel rounded-lg p-4">
              <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr_0.8fr]">
                <div className="min-w-0">
                  <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h2 className="truncate text-sm font-semibold text-white">{job.name}</h2>
                      <div className="mt-1 text-xs text-ink-400">{job.recipeName} · GPU {job.gpu} · {job.launcher}</div>
                    </div>
                    <StatusPill label={t(statusLabelKeys[job.status])} tone={statusTone[job.status]} />
                  </div>

                  <div className="grid gap-2 sm:grid-cols-4">
                    <div className="rounded-md bg-white/[0.04] p-2">
                      <div className="text-xs text-ink-400">{t('step')}</div>
                      <div className="mt-1 text-sm text-white">{job.latestStep} / {job.maxTrainSteps}</div>
                    </div>
                    <div className="rounded-md bg-white/[0.04] p-2">
                      <div className="text-xs text-ink-400">{t('loss')}</div>
                      <div className="mt-1 text-sm text-white">{formatLoss(job.latestLoss)}</div>
                    </div>
                    <div className="rounded-md bg-white/[0.04] p-2">
                      <div className="text-xs text-ink-400">{t('vram')}</div>
                      <div className="mt-1 text-sm text-white">{job.lowVram ? t('low') : t('standard')}</div>
                    </div>
                    <div className="rounded-md bg-white/[0.04] p-2">
                      <div className="text-xs text-ink-400">{t('params')}</div>
                      <div className="mt-1 text-sm text-white">{job.trainableParams?.toLocaleString() ?? '-'}</div>
                    </div>
                  </div>
                  {job.runnerPid ? (
                    <div className="mt-2 rounded-md border border-white/10 bg-black/20 px-2 py-1 text-xs text-ink-400">
                      PID {job.runnerPid} · {t('runnerArtifact')} {job.runnerExitCode == null ? t('runnerActive') : `${t('runnerExit')} ${job.runnerExitCode}`}
                    </div>
                  ) : null}
                  <FailureDiagnostics job={job} />
                </div>

                <div className="min-w-0 space-y-3">
                  <div className="flex flex-wrap gap-2">
                    {artifactLabel(job.artifacts.args, t('argsArtifact'))}
                    {artifactLabel(job.artifacts.log, t('logArtifact'))}
                    {artifactLabel(job.artifacts.loss, t('lossArtifact'))}
                    {artifactLabel(job.artifacts.runnerLog, t('runnerArtifact'))}
                    {artifactLabel(job.artifacts.samples, `${t('samplesShort')} ${job.artifactCounts.samples + job.artifactCounts.sampleTrajectories}`)}
                    {artifactLabel(job.artifacts.checkpoints, `${t('ckptShort')} ${job.artifactCounts.checkpoints}`)}
                  </div>
                  <LossBars job={job} />
                  <Link
                    href="/tensorboard"
                    className="inline-flex items-center gap-2 rounded-md border border-aqua-500/25 bg-aqua-500/[0.1] px-3 py-2 text-xs text-aqua-300 transition hover:bg-aqua-500/[0.16]"
                  >
                    <LineChart className="h-3.5 w-3.5" />
                    {t('tensorboardOpenFullLoss')}
                  </Link>
                  <LossHistorySummary job={job} />
                  <ArtifactBrowser job={job} />
                  <div className="rounded-md bg-black/25 p-2 font-mono text-xs leading-5 text-ink-400">
                    {(job.logTail.length ? job.logTail : job.runnerLogTail).slice(-2).map(line => (
                      <div key={line} className="truncate">{line}</div>
                    ))}
                  </div>
                </div>

                <div className="flex min-w-0 flex-col gap-2">
                  <div className="min-w-0 rounded-md border border-white/10 bg-black/25 p-3 font-mono text-xs leading-5 text-ink-300">
                    <div className="mb-1 flex items-center gap-2 text-ink-400">
                      <FileText className="h-3.5 w-3.5" />
                      {t('command')}
                    </div>
                    <div className="break-all">{job.command}</div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {job.status === 'running' || job.status === 'queued' ? (
                      <button
                        className="inline-flex items-center gap-2 rounded-md border border-amberSoft-500/25 bg-amberSoft-500/[0.1] px-3 py-2 text-sm text-amberSoft-500 transition hover:bg-amberSoft-500/[0.16] disabled:opacity-50"
                        disabled={busy === `stop-${job.id}`}
                        onClick={() => void stop(job)}
                      >
                        <Square className="h-4 w-4" />
                        {t('stop')}
                      </button>
                    ) : job.status !== 'completed' && job.source !== 'seeded-failure' ? (
                      <button
                        className="inline-flex items-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.12] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/[0.2] disabled:opacity-50"
                        disabled={busy === `start-${job.id}`}
                        onClick={() => void start(job)}
                      >
                        <Play className="h-4 w-4" />
                        {t('start')}
                      </button>
                    ) : null}
                    <button
                      className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.05] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1] disabled:opacity-50"
                      disabled={busy === `logs-${job.id}`}
                      onClick={() => void toggleLogs(job)}
                      title={t('fullLogs')}
                    >
                      <ScrollText className="h-4 w-4" />
                      {t('logs')}
                    </button>
                    <button
                      className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.05] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1] disabled:opacity-50"
                      disabled={busy === `open-${job.id}`}
                      onClick={() => void openOutput(job)}
                      title={t('openOutputFolder')}
                    >
                      <FolderOpen className="h-4 w-4" />
                      {t('output')}
                    </button>
                    <button
                      className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.05] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1] disabled:opacity-50"
                      disabled={busy === `clone-${job.id}`}
                      onClick={() => void clone(job)}
                    >
                      <Copy className="h-4 w-4" />
                      {t('clone')}
                    </button>
                    <button
                      className="inline-flex items-center gap-2 rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.08] px-3 py-2 text-sm text-roseSoft-500 transition hover:bg-roseSoft-500/[0.14] disabled:opacity-50"
                      disabled={busy === `delete-${job.id}` || job.status === 'running' || job.status === 'queued'}
                      onClick={() => void remove(job)}
                    >
                      <Trash2 className="h-4 w-4" />
                      {t('delete')}
                    </button>
                  </div>
                  {openLogIds.includes(job.id) ? <LogPanel payload={logsByJob[job.id]} /> : null}
                </div>
              </div>
            </article>
          ))
        )}
      </section>
    </div>
  );
}
