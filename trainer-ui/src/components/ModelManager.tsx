'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Download, FolderOpen, HardDrive, LockKeyhole, RefreshCw, Save, ShieldCheck, TriangleAlert } from 'lucide-react';
import StatusPill from '@/components/StatusPill';
import { useI18n } from '@/lib/i18n';
import type { CustomModelPathEntry, ModelCacheSummary } from '@/lib/models';

type ModelsPayload = {
  ok: boolean;
  generatedAt: string;
  cacheRoot: string;
  models: ModelCacheSummary[];
  customPaths?: CustomModelPathEntry[];
  error?: string;
};

type ModelActionPayload = {
  ok: boolean;
  generatedAt: string;
  result?: { ok: boolean; model_id?: string; path?: string; error?: string };
  customPath?: CustomModelPathEntry | null;
  customPaths?: CustomModelPathEntry[];
  models?: ModelCacheSummary[];
  error?: string;
};

function formatBytes(bytes: number) {
  if (!bytes) return '-';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let index = 0;
  while (value >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  return `${value.toFixed(index < 2 ? 0 : 1)} ${units[index]}`;
}

function shortPath(filePath?: string | null) {
  if (!filePath) return '-';
  return filePath.replace(/\\/g, '/').split('/').slice(-4).join('/');
}

function modelTone(model: ModelCacheSummary): 'good' | 'warn' | 'bad' {
  if (model.cached) return 'good';
  if (model.spec?.default) return 'bad';
  return 'warn';
}

export default function ModelManager() {
  const [models, setModels] = useState<ModelCacheSummary[]>([]);
  const [customPaths, setCustomPaths] = useState<CustomModelPathEntry[]>([]);
  const [customPathInputs, setCustomPathInputs] = useState<Record<string, string>>({});
  const [cacheRoot, setCacheRoot] = useState('');
  const [loading, setLoading] = useState(true);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const { t } = useI18n();

  const mergePayload = useCallback((payload: Pick<ModelsPayload, 'models' | 'customPaths' | 'cacheRoot'>) => {
    const nextModels = payload.models ?? [];
    const nextCustomPaths = payload.customPaths ?? [];
    setModels(nextModels);
    setCustomPaths(nextCustomPaths);
    if (payload.cacheRoot != null) setCacheRoot(payload.cacheRoot);
    setCustomPathInputs(current => {
      const next = { ...current };
      for (const model of nextModels) {
        if (!(model.model_id in next)) next[model.model_id] = '';
      }
      for (const entry of nextCustomPaths) {
        next[entry.modelId] = entry.path;
      }
      return next;
    });
  }, []);

  const loadModels = useCallback(async () => {
    setLoading(true);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch('/api/models', { cache: 'no-store' });
      const payload = (await response.json()) as ModelsPayload;
      mergePayload(payload);
      if (!response.ok || !payload.ok) setError(payload.error || t('modelApiUnavailable'));
    } catch {
      setError(t('modelApiUnavailable'));
      setModels([]);
      setCustomPaths([]);
    } finally {
      setLoading(false);
    }
  }, [mergePayload, t]);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const counts = useMemo(() => {
    const defaults = models.filter(model => model.spec?.default);
    const cached = models.filter(model => model.cached);
    const size = models.reduce((totalSize, model) => totalSize + (model.size_bytes || 0), 0);
    return {
      total: models.length,
      defaultCached: defaults.filter(model => model.cached).length,
      defaultTotal: defaults.length,
      cached: cached.length,
      size,
    };
  }, [models]);
  const defaultsReady = counts.defaultTotal > 0 && counts.defaultCached === counts.defaultTotal;

  function customPathFor(modelId: string) {
    return customPaths.find(entry => entry.modelId === modelId);
  }

  function modelStatus(model: ModelCacheSummary) {
    if (model.cached) return t('cached');
    return model.spec?.default ? t('missing') : t('optionalFallback');
  }

  async function download(modelId: string) {
    setBusyAction(`download-${modelId}`);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch('/api/models', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ modelId }),
      });
      const payload = (await response.json()) as ModelActionPayload;
      if (payload.models) mergePayload({ models: payload.models, customPaths: payload.customPaths, cacheRoot });
      if (!response.ok || !payload.ok) setError(payload.result?.error || payload.error || t('downloadFailed'));
    } catch {
      setError(t('downloadRequestFailed'));
    } finally {
      setBusyAction(null);
    }
  }

  async function handleSaveCustomPath(modelId: string) {
    setBusyAction(`save-${modelId}`);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch('/api/models', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          action: 'save-custom-path',
          modelId,
          path: customPathInputs[modelId] || '',
        }),
      });
      const payload = (await response.json()) as ModelActionPayload;
      if (payload.models) mergePayload({ models: payload.models, customPaths: payload.customPaths, cacheRoot });
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('pathMissing'));
        return;
      }
      setNotice(t('customPathSaved'));
    } catch {
      setError(t('pathMissing'));
    } finally {
      setBusyAction(null);
    }
  }

  async function handleOpenModelFolder(model: ModelCacheSummary) {
    setBusyAction(`open-${model.model_id}`);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch('/api/models', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          action: 'open-folder',
          modelId: model.model_id,
          path: customPathInputs[model.model_id] || customPathFor(model.model_id)?.path || model.cache_dir,
        }),
      });
      const payload = (await response.json()) as ModelActionPayload;
      if (!response.ok || !payload.ok) {
        setError(payload.result?.error || payload.error || t('folderOpenFailed'));
      }
    } catch {
      setError(t('folderOpenFailed'));
    } finally {
      setBusyAction(null);
    }
  }

  return (
    <div className="space-y-4">
      <section className="grid gap-3 md:grid-cols-4">
        {[
          [t('models'), counts.total],
          [t('cached'), counts.cached],
          [t('defaults'), `${counts.defaultCached} / ${counts.defaultTotal}`],
          [t('size'), formatBytes(counts.size)],
        ].map(([label, value]) => (
          <div key={label} className="solid-panel rounded-lg p-4">
            <div className="text-xs uppercase tracking-wide text-ink-400">{label}</div>
            <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
          </div>
        ))}
      </section>

      <section className="glass rounded-lg p-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2 text-sm font-medium text-white">
            {defaultsReady ? <ShieldCheck className="h-4 w-4 text-mint-500" /> : <TriangleAlert className="h-4 w-4 text-roseSoft-500" />}
            {t('modelCache')}
            <StatusPill label={loading ? t('checking') : defaultsReady ? t('ready') : t('missing')} tone={loading ? 'neutral' : defaultsReady ? 'good' : 'bad'} />
          </div>
          <button
            className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.06] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1]"
            onClick={() => void loadModels()}
          >
            <RefreshCw className="h-4 w-4" />
            {t('refresh')}
          </button>
        </div>
        {cacheRoot ? <div className="mt-3 break-all rounded-md bg-black/20 px-3 py-2 text-xs text-ink-400">{cacheRoot}</div> : null}
        <div className="mt-3 rounded-md border border-white/10 bg-white/[0.035] px-3 py-2 text-xs leading-5 text-ink-400">
          <span className="font-medium text-ink-300">{t('customModelPath')}</span>
          <span className="ml-2">{t('customModelPathHelp')}</span>
        </div>
        {notice ? <div className="mt-3 rounded-md border border-mint-500/25 bg-mint-500/[0.08] px-3 py-2 text-sm text-mint-500">{notice}</div> : null}
        {error ? <div className="mt-3 rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.08] px-3 py-2 text-sm text-roseSoft-500">{error}</div> : null}
      </section>

      {loading ? (
        <div className="solid-panel rounded-lg p-4 text-sm text-ink-400">{t('loadingModels')}</div>
      ) : (
        <section className="grid gap-4 xl:grid-cols-2">
          {models.map(model => {
            const customPath = customPathFor(model.model_id);
            const inputValue = customPathInputs[model.model_id] ?? customPath?.path ?? '';
            const canOpen = Boolean(inputValue || model.cached);
            return (
              <article key={model.model_id} className="solid-panel min-w-0 rounded-lg p-4">
                <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h2 className="break-all font-medium text-white">{model.model_id}</h2>
                    <div className="mt-1 text-sm text-ink-400">{model.spec?.role || t('unregisteredModel')}</div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <StatusPill label={modelStatus(model)} tone={modelTone(model)} />
                    {customPath ? <StatusPill label={customPath.exists ? t('customModelPath') : t('pathMissing')} tone={customPath.exists ? 'good' : 'warn'} /> : null}
                    {model.spec?.experimental ? <StatusPill label={t('experimental')} tone="warn" /> : model.spec?.default ? <StatusPill label={t('default')} tone="good" /> : null}
                  </div>
                </div>

                <div className="grid gap-2 sm:grid-cols-3">
                  <div className="rounded-md bg-white/[0.05] p-3">
                    <HardDrive className="mb-2 h-4 w-4 text-aqua-300" />
                    <div className="text-xs text-ink-400">{t('size')}</div>
                    <div className="text-sm text-white">{formatBytes(model.size_bytes)}</div>
                  </div>
                  <div className="rounded-md bg-white/[0.05] p-3">
                    <FolderOpen className="mb-2 h-4 w-4 text-mint-500" />
                    <div className="text-xs text-ink-400">{t('snapshots')}</div>
                    <div className="text-sm text-white">{model.snapshot_count}</div>
                  </div>
                  <div className="rounded-md bg-white/[0.05] p-3">
                    <LockKeyhole className="mb-2 h-4 w-4 text-amberSoft-500" />
                    <div className="text-xs text-ink-400">{t('auth')}</div>
                    <div className="text-sm text-white">{model.spec?.gated_possible ? t('mayGate') : t('openAccess')}</div>
                  </div>
                </div>

                <div className="mt-3 space-y-2">
                  <div className="break-all rounded-md bg-black/20 p-2 text-xs text-ink-400">
                    {t('primary')}: {shortPath(model.primary_snapshot)}
                  </div>
                  <div className="break-all rounded-md bg-black/20 p-2 text-xs text-ink-400">
                    {t('cache')}: {model.cache_dir}
                  </div>
                </div>

                <div className="mt-3 rounded-md border border-white/10 bg-white/[0.035] p-3">
                  <label className="mb-2 block text-xs font-medium text-ink-300" htmlFor={`custom-model-path-${model.model_id}`}>
                    {t('customModelPath')}
                  </label>
                  <div className="grid gap-2 lg:grid-cols-[1fr_auto_auto]">
                    <input
                      id={`custom-model-path-${model.model_id}`}
                      className="min-w-0 rounded-md border border-white/10 bg-black/25 px-3 py-2 text-sm text-white outline-none transition placeholder:text-ink-600 focus:border-aqua-500/50"
                      value={inputValue}
                      onChange={event => setCustomPathInputs(current => ({ ...current, [model.model_id]: event.target.value }))}
                      placeholder={t('customModelPathPlaceholder')}
                    />
                    <button
                      className="inline-flex items-center justify-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.12] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/20 disabled:opacity-50"
                      disabled={busyAction === `save-${model.model_id}`}
                      onClick={() => void handleSaveCustomPath(model.model_id)}
                    >
                      <Save className="h-4 w-4" />
                      {busyAction === `save-${model.model_id}` ? t('saving') : t('savePath')}
                    </button>
                    <button
                      className="inline-flex items-center justify-center gap-2 rounded-md border border-white/10 bg-white/[0.05] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1] disabled:opacity-50"
                      disabled={!canOpen || busyAction === `open-${model.model_id}`}
                      onClick={() => void handleOpenModelFolder(model)}
                    >
                      <FolderOpen className="h-4 w-4" />
                      {busyAction === `open-${model.model_id}` ? t('opening') : t('openFolder')}
                    </button>
                  </div>
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  <button
                    className="inline-flex items-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.12] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/20 disabled:opacity-50"
                    disabled={busyAction === `download-${model.model_id}` || (model.cached && !model.spec?.experimental)}
                    onClick={() => void download(model.model_id)}
                  >
                    <Download className="h-4 w-4" />
                    {busyAction === `download-${model.model_id}` ? t('working') : model.cached ? t('cached') : t('download')}
                  </button>
                </div>
              </article>
            );
          })}
        </section>
      )}
    </div>
  );
}
