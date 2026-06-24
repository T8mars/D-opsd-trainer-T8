'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Cpu, FolderCog, KeyRound, RefreshCw, ServerCog, ShieldCheck, SlidersHorizontal } from 'lucide-react';
import StatusPill from '@/components/StatusPill';
import { useI18n, type MessageKey } from '@/lib/i18n';
import type { ModelCacheSummary } from '@/lib/models';
import type { SafetyDefault, SettingsPathItem, SettingsSummary } from '@/lib/settings';
import type { SystemPayload } from '@/lib/system';

type SettingsPayload = {
  ok: boolean;
  generatedAt: string;
  settings: SettingsSummary | null;
  system: SystemPayload | null;
  models: ModelCacheSummary[];
  error?: string;
};

const emptyPayload: SettingsPayload = {
  ok: false,
  generatedAt: '',
  settings: null,
  system: null,
  models: [],
};

const pathLabelKeys: Record<string, MessageKey> = {
  project_root: 'pathProjectRoot',
  wsl_venv: 'pathWslVenv',
  hf_home: 'pathHfHome',
  hf_cache: 'pathHfCache',
  jobs_ledger: 'pathJobsLedger',
  runner_root: 'pathRunnerRoot',
  smoke_runs: 'pathSmokeRuns',
};

const pathPurposeKeys: Record<string, MessageKey> = {
  project_root: 'pathProjectRootPurpose',
  wsl_venv: 'pathWslVenvPurpose',
  hf_home: 'pathHfHomePurpose',
  hf_cache: 'pathHfCachePurpose',
  jobs_ledger: 'pathJobsLedgerPurpose',
  runner_root: 'pathRunnerRootPurpose',
  smoke_runs: 'pathSmokeRunsPurpose',
};

const safetyLabelKeys: Record<string, MessageKey> = {
  launcher: 'launcher',
  low_vram: 'lowVramOffload',
  block_offload: 'blockOffload',
  resolution_scale: 'trainingScale',
  use_8bit_adam: 'eightBitAdam',
  save_samples: 'saveSamples',
  save_checkpoints: 'saveCheckpoints',
};

const safetyReasonKeys: Record<string, MessageKey> = {
  launcher: 'safetyLauncherReason',
  low_vram: 'safetyLowVramReason',
  block_offload: 'safetyBlockOffloadReason',
  resolution_scale: 'safetyResolutionScaleReason',
  use_8bit_adam: 'safetyEightBitAdamReason',
  save_samples: 'safetySaveSamplesReason',
  save_checkpoints: 'safetySaveCheckpointsReason',
};

function shortTime(value: string, waitingLabel: string) {
  if (!value) return waitingLabel;
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(new Date(value));
}

function yesNoTone(value: boolean): 'good' | 'warn' {
  return value ? 'good' : 'warn';
}

function SectionTitle({ icon: Icon, title }: { icon: typeof ServerCog; title: string }) {
  return (
    <div className="mb-3 flex items-center gap-2 text-sm font-medium text-white">
      <Icon className="h-4 w-4 text-aqua-300" />
      {title}
    </div>
  );
}

export default function SettingsConsole() {
  const [payload, setPayload] = useState<SettingsPayload>(emptyPayload);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { t } = useI18n();

  const loadSettings = useCallback(async (showSpinner = false) => {
    if (showSpinner) setRefreshing(true);
    try {
      const response = await fetch('/api/settings', { cache: 'no-store' });
      const nextPayload = (await response.json()) as SettingsPayload;
      setPayload(nextPayload);
    } catch {
      setPayload({ ...emptyPayload, error: t('settingsApiUnavailable') });
    } finally {
      setLoading(false);
      if (showSpinner) setRefreshing(false);
    }
  }, [t]);

  useEffect(() => {
    void loadSettings();
  }, [loadSettings]);

  const modelCounts = useMemo(() => {
    const defaults = payload.models.filter(model => model.spec?.default);
    return {
      total: payload.models.length,
      cached: payload.models.filter(model => model.cached).length,
      defaultCached: defaults.filter(model => model.cached).length,
      defaultTotal: defaults.length,
    };
  }, [payload.models]);

  const settings = payload.settings;
  const system = payload.system;
  const primaryGpu = system?.gpu.gpus[0];
  const wslReady = Boolean(system?.wsl.available);
  const defaultsReady = modelCounts.defaultTotal > 0 && modelCounts.defaultCached === modelCounts.defaultTotal;
  const tokenPresent = Boolean(settings?.hf_token.present || system?.hfToken.present);

  function formatBoolean(value: boolean) {
    return value ? t('enabled') : t('disabled');
  }

  function formatValue(value: string | boolean | number) {
    if (typeof value === 'boolean') return formatBoolean(value);
    if (value === 'profile-specific') return t('profileSpecific');
    return String(value);
  }

  function labelForPath(item: SettingsPathItem) {
    const key = pathLabelKeys[item.id];
    return key ? t(key) : item.label;
  }

  function purposeForPath(item: SettingsPathItem) {
    const key = pathPurposeKeys[item.id];
    return key ? t(key) : item.purpose;
  }

  function labelForSafety(item: SafetyDefault) {
    const key = safetyLabelKeys[item.id];
    return key ? t(key) : item.label;
  }

  function reasonForSafety(item: SafetyDefault) {
    const key = safetyReasonKeys[item.id];
    return key ? t(key) : item.reason;
  }

  function runnerBackendLabel(value: string) {
    return value === 'WSL detached runner' ? t('wslDetachedRunner') : value;
  }

  function queueOrderLabel(value: string) {
    return value.toLowerCase() === 'fifo' ? t('fifoQueue') : value.toUpperCase();
  }

  function profileTierLabel(value: string) {
    return value === 'recommended_16gb' ? t('recommended16gbTier') : value.replace('_', ' ');
  }

  function profileLabel(recipeId: string, fallback: string) {
    if (recipeId === 'flux2-klein-identity') return t('profileFlux2Identity');
    if (recipeId === 'flux2-klein-editing') return t('profileFlux2Editing');
    if (recipeId === 'z-image-turbo-vlm') return t('profileZImage');
    return fallback;
  }

  function artifactModeLabel(value: string) {
    return value === 'samples_and_checkpoints' ? t('samplesAndCheckpointsMode') : value.replace(/_/g, ' + ');
  }

  if (!settings && loading) {
    return <div className="solid-panel rounded-lg p-4 text-sm text-ink-400">{t('loadingSettings')}</div>;
  }

  return (
    <div className="space-y-4">
      <section className="glass rounded-lg p-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2 text-sm font-medium text-white">
            <SlidersHorizontal className="h-4 w-4 text-aqua-300" />
            {t('runtimeSettings')}
            <StatusPill label={payload.ok ? t('liveReadOnly') : t('checkFailed')} tone={payload.ok ? 'good' : 'bad'} />
          </div>
          <button
            className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.06] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1] disabled:opacity-50"
            disabled={refreshing}
            aria-label={t('refreshRuntimeSettings')}
            title={t('refreshRuntimeSettings')}
            onClick={() => void loadSettings(true)}
          >
            <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            {shortTime(payload.generatedAt, t('waiting'))}
          </button>
        </div>
        {payload.error ? <div className="mt-3 rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.08] px-3 py-2 text-sm text-roseSoft-500">{payload.error}</div> : null}
      </section>

      <section className="grid gap-3 md:grid-cols-4">
        {[
          { label: t('backend'), value: wslReady ? settings?.backend.distro || 'WSL2' : t('wslMissing'), tone: yesNoTone(wslReady), icon: ServerCog },
          { label: t('defaultModels'), value: `${modelCounts.defaultCached} / ${modelCounts.defaultTotal}`, tone: defaultsReady ? 'good' : 'bad', icon: ShieldCheck },
          { label: t('hfToken'), value: tokenPresent ? t('present') : t('missing'), tone: tokenPresent ? 'good' : 'warn', icon: KeyRound },
          { label: t('queueSlots'), value: settings?.runner_policy.max_active_jobs ?? '-', tone: 'good', icon: Cpu },
        ].map(item => {
          const Icon = item.icon;
          return (
            <div key={item.label} className="solid-panel min-w-0 rounded-lg p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-ink-400">
                  <Icon className="h-3.5 w-3.5 text-aqua-300" />
                  {item.label}
                </div>
                <StatusPill label={String(item.value)} tone={item.tone as 'good' | 'warn' | 'bad'} />
              </div>
              <div className="truncate text-sm text-white" title={String(item.value)}>{item.value}</div>
            </div>
          );
        })}
      </section>

      {settings ? (
        <div className="grid gap-4 xl:grid-cols-[1.25fr_0.75fr]">
          <section className="solid-panel min-w-0 rounded-lg p-4">
            <SectionTitle icon={FolderCog} title={t('runtimePaths')} />
            <div className="space-y-2">
              {settings.paths.map(item => (
                <div key={item.id} className="grid gap-2 rounded-md border border-white/10 bg-white/[0.035] p-3 lg:grid-cols-[190px_1fr_auto]">
                  <div className="min-w-0">
                    <div className="text-sm text-white">{labelForPath(item)}</div>
                    <div className="mt-1 text-xs text-ink-400">{purposeForPath(item)}</div>
                  </div>
                  <div className="min-w-0 break-all rounded-md bg-black/20 px-2 py-1.5 text-xs text-ink-300">{item.path}</div>
                  <div className="flex items-start justify-end">
                    <StatusPill label={item.exists ? t('exists') : t('missing')} tone={item.exists ? 'good' : 'warn'} />
                  </div>
                </div>
              ))}
            </div>
          </section>

          <div className="space-y-4">
            <section className="solid-panel rounded-lg p-4">
              <SectionTitle icon={ServerCog} title={t('detectedBackend')} />
              <div className="space-y-3 text-sm">
                <div className="flex min-w-0 justify-between gap-3">
                  <span className="text-ink-400">{t('preferred')}</span>
                  <span className="truncate text-white">{settings.backend.preferred}</span>
                </div>
                <div className="flex min-w-0 justify-between gap-3">
                  <span className="text-ink-400">{t('wslDistro')}</span>
                  <span className="truncate text-white">{system?.wsl.distributions.join(', ') || settings.backend.distro}</span>
                </div>
                <div className="flex min-w-0 justify-between gap-3">
                  <span className="text-ink-400">GPU</span>
                  <span className="truncate text-white">{primaryGpu?.name || t('unavailable')}</span>
                </div>
                <div className="flex min-w-0 justify-between gap-3">
                  <span className="text-ink-400">Python</span>
                  <span className="truncate text-white">{system?.python.version || t('unavailable')}</span>
                </div>
              </div>
            </section>

            <section className="solid-panel rounded-lg p-4">
              <SectionTitle icon={ShieldCheck} title={t('runnerPolicy')} />
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
                {[
                  [t('backend'), runnerBackendLabel(settings.runner_policy.backend)],
                  [t('queue'), queueOrderLabel(settings.runner_policy.queue_order)],
                  [t('maxActive'), settings.runner_policy.max_active_jobs],
                  [t('deleteRunning'), formatBoolean(settings.runner_policy.delete_running_jobs)],
                  [t('autoPromote'), formatBoolean(settings.runner_policy.auto_promote_queued)],
                ].map(([label, value]) => (
                  <div key={label} className="flex min-w-0 justify-between gap-3 rounded-md bg-white/[0.04] px-3 py-2 text-sm">
                    <span className="text-ink-400">{label}</span>
                    <span className="truncate text-white">{value}</span>
                  </div>
                ))}
              </div>
            </section>
          </div>
        </div>
      ) : null}

      {settings ? (
        <section className="solid-panel rounded-lg p-4">
          <SectionTitle icon={SlidersHorizontal} title={t('lowVramSafetyDefaults')} />
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {settings.safety_defaults.map(item => (
              <article key={item.id} className="min-w-0 rounded-md border border-white/10 bg-white/[0.04] p-3">
                <div className="mb-2 flex items-start justify-between gap-3">
                  <div className="text-sm font-medium text-white">{labelForSafety(item)}</div>
                  <StatusPill label={formatValue(item.value)} tone={typeof item.value === 'boolean' ? yesNoTone(item.value) : 'neutral'} />
                </div>
                <div className="text-xs leading-5 text-ink-400">{reasonForSafety(item)}</div>
              </article>
            ))}
          </div>
        </section>
      ) : null}

      {settings ? (
        <section className="solid-panel rounded-lg p-4">
          <SectionTitle icon={ShieldCheck} title={t('verified16gbProfiles')} />
          <div className="grid gap-3 xl:grid-cols-3">
            {settings.production_profiles.map(profile => (
              <article key={profile.id} className="min-w-0 rounded-md border border-white/10 bg-white/[0.04] p-3">
                <div className="mb-3 flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-white">{profileLabel(profile.recipe_id, profile.label)}</div>
                    <div className="mt-1 text-xs text-ink-400">{profile.recipe_id}</div>
                  </div>
                  <StatusPill label={profileTierLabel(profile.tier)} tone="good" />
                </div>
                <div className="grid gap-2 text-sm sm:grid-cols-2 xl:grid-cols-1">
                  {[
                    [t('trainScale'), profile.resolution_scale],
                    [t('sampleScale'), profile.sample_resolution_scale || t('native')],
                    [t('steps'), profile.max_train_steps],
                    [t('artifactMode'), artifactModeLabel(profile.artifact_mode)],
                  ].map(([label, value]) => (
                    <div key={label} className="flex min-w-0 justify-between gap-3 rounded-md bg-black/20 px-3 py-2">
                      <span className="text-ink-400">{label}</span>
                      <span className="truncate text-white">{value}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-3 break-all rounded-md bg-black/25 p-2 font-mono text-xs leading-5 text-ink-400">
                  {profile.evidence[0]}
                </div>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
