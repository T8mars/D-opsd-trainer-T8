'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  CheckCircle2,
  FileJson,
  ImageIcon,
  Plus,
  RefreshCw,
  Save,
  Search,
  Trash2,
  TriangleAlert,
  Upload,
} from 'lucide-react';
import StatusPill from '@/components/StatusPill';
import { useI18n, type MessageKey } from '@/lib/i18n';
import { recipes, type RecipeId } from '@/lib/recipes';
import type { DatasetRegistryItem, DatasetSummary, ManagedDatasetItem } from '@/lib/datasets';

type DatasetResult = DatasetRegistryItem & {
  summary: DatasetSummary;
};

type DatasetsPayload = {
  ok: boolean;
  generatedAt: string;
  datasets: DatasetResult[];
  error?: string;
};

type ValidatePayload = {
  ok: boolean;
  generatedAt: string;
  dataset: DatasetResult;
  error?: string;
};

function statusTone(ok: boolean): 'good' | 'bad' {
  return ok ? 'good' : 'bad';
}

function shortPath(filePath: string) {
  return filePath.replace(/\\/g, '/').split('/').slice(-4).join('/');
}

function imagePathUrl(filePath: string) {
  return `/api/datasets/image?path=${encodeURIComponent(filePath.replace(/\\/g, '/'))}`;
}

function formatBuckets(summary: DatasetSummary) {
  if (!summary.bucket_summary.length) return '-';
  return summary.bucket_summary
    .slice(0, 3)
    .map(bucket => `${bucket.width}x${bucket.height} x${bucket.count}`)
    .join(' · ');
}

function datasetShapeLabel(shape: string, t: (key: MessageKey) => string) {
  if (shape === 'single image') return t('singleImageDataset');
  if (shape === 'reference + target') return t('referenceTargetDataset');
  if (shape === 'combined') return t('combinedDataset');
  return shape;
}

function DatasetItemEditor({
  item,
  prompt,
  onPromptChange,
  onSave,
  onDelete,
}: {
  item: ManagedDatasetItem;
  prompt: string;
  onPromptChange: (value: string) => void;
  onSave: () => void;
  onDelete: () => void;
}) {
  const { t } = useI18n();
  return (
    <div className="grid gap-3 rounded-md border border-white/10 bg-black/20 p-3 lg:grid-cols-[180px_1fr_auto]">
      <div className="grid grid-cols-2 gap-2">
        {item.imagePaths.slice(0, 2).map(imagePath => (
          <div key={imagePath} className="min-w-0">
            <img
              src={imagePathUrl(imagePath)}
              alt={shortPath(imagePath)}
              className="aspect-square w-full rounded-md border border-white/10 object-cover"
            />
            <div className="mt-1 truncate text-[11px] text-ink-400">{shortPath(imagePath)}</div>
          </div>
        ))}
      </div>
      <textarea
        className="min-h-24 w-full rounded-md border border-white/10 bg-black/25 px-3 py-2 text-sm leading-5 text-white outline-none transition placeholder:text-ink-600 focus:border-aqua-500/50"
        value={prompt}
        placeholder={t('captionPlaceholder')}
        onChange={event => onPromptChange(event.target.value)}
      />
      <div className="flex gap-2 lg:flex-col">
        <button
          className="inline-flex items-center justify-center gap-2 rounded-md border border-mint-500/30 bg-mint-500/[0.12] px-3 py-2 text-sm text-mint-500 transition hover:bg-mint-500/20"
          onClick={onSave}
        >
          <Save className="h-4 w-4" />
          {t('saveCaption')}
        </button>
        <button
          className="inline-flex items-center justify-center gap-2 rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.1] px-3 py-2 text-sm text-roseSoft-500 transition hover:bg-roseSoft-500/[0.16]"
          onClick={onDelete}
        >
          <Trash2 className="h-4 w-4" />
          {t('deleteImage')}
        </button>
      </div>
    </div>
  );
}

function DatasetCard({
  dataset,
  selected,
  onToggleSelected,
  onUpdateItem,
  onDeleteItem,
  onAddItem,
  onDeleteDataset,
}: {
  dataset: DatasetResult;
  selected: boolean;
  onToggleSelected: () => void;
  onUpdateItem: (datasetId: string, itemId: string, prompt: string) => void;
  onDeleteItem: (datasetId: string, itemId: string) => void;
  onAddItem: (datasetId: string, files: File[], prompt: string) => void;
  onDeleteDataset: (datasetId: string) => void;
}) {
  const issues = dataset.summary.issues.filter(issue => issue.severity === 'error');
  const preview = dataset.summary.previews[0];
  const { t } = useI18n();
  const [itemPrompts, setItemPrompts] = useState<Record<string, string>>({});
  const [addFiles, setAddFiles] = useState<File[]>([]);
  const [addPrompt, setAddPrompt] = useState('');

  useEffect(() => {
    const next: Record<string, string> = {};
    for (const item of dataset.items ?? []) {
      next[item.id] = item.prompt;
    }
    setItemPrompts(next);
  }, [dataset.items]);

  const managedItems = dataset.managed ? dataset.items ?? [] : [];

  return (
    <article className="solid-panel min-w-0 rounded-lg p-4">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <label className="flex min-w-0 items-start gap-3">
          <input
            type="checkbox"
            checked={selected}
            className="mt-1 h-4 w-4 shrink-0 accent-cyan-300"
            onChange={onToggleSelected}
          />
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-white">
              <FileJson className="h-4 w-4 shrink-0 text-aqua-300" />
              <span className="truncate">{dataset.name}</span>
            </div>
            <div className="mt-1 text-xs text-ink-400">
              {datasetShapeLabel(dataset.shape, t)} · {dataset.recipeId} · {dataset.managed ? t('managedDataset') : t('readOnlyDataset')}
            </div>
          </div>
        </label>
        <div className="flex flex-wrap items-center gap-2">
          <StatusPill label={dataset.summary.ok ? t('valid') : t('issues')} tone={statusTone(dataset.summary.ok)} />
          {dataset.managed ? (
            <button
              className="inline-flex items-center gap-2 rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.08] px-2.5 py-1.5 text-xs text-roseSoft-500 transition hover:bg-roseSoft-500/[0.14]"
              onClick={() => onDeleteDataset(dataset.id)}
            >
              <Trash2 className="h-3.5 w-3.5" />
              {t('deleteDataset')}
            </button>
          ) : null}
        </div>
      </div>

      <div className="mb-3 break-all rounded-md bg-black/20 p-2 text-xs text-ink-400">{dataset.path}</div>

      <div className="grid gap-2 sm:grid-cols-3">
        <div className="rounded-md bg-white/[0.04] p-2">
          <div className="text-xs text-ink-400">{t('rows')}</div>
          <div className="mt-1 text-sm text-white">{dataset.summary.valid_rows} / {dataset.summary.rows}</div>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <div className="text-xs text-ink-400">{t('issues')}</div>
          <div className="mt-1 text-sm text-white">{issues.length}</div>
        </div>
        <div className="rounded-md bg-white/[0.04] p-2">
          <div className="text-xs text-ink-400">{t('buckets')}</div>
          <div className="mt-1 truncate text-sm text-white" title={formatBuckets(dataset.summary)}>{dataset.summary.bucket_summary.length}</div>
        </div>
      </div>

      <div className="mt-3 rounded-md border border-white/10 bg-black/20 p-3">
        <div className="mb-2 flex items-center gap-2 text-xs text-ink-400">
          <ImageIcon className="h-3.5 w-3.5" />
          {formatBuckets(dataset.summary)}
        </div>
        {preview ? (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              {preview.images.slice(0, 3).map(image => (
                <div key={image.path} className="min-w-0">
                  {image.url ? (
                    <img
                      src={image.url}
                      alt={shortPath(image.path)}
                      className="aspect-square w-full rounded-md border border-white/10 object-cover"
                    />
                  ) : (
                    <div className="aspect-square rounded-md border border-white/10 bg-white/[0.04]" />
                  )}
                  <div className="mt-1 truncate text-[11px] text-ink-400">{image.width && image.height ? `${image.width}x${image.height}` : shortPath(image.path)}</div>
                </div>
              ))}
            </div>
            <p className="line-clamp-3 text-xs leading-5 text-ink-300">{preview.prompt || t('noPromptPreview')}</p>
          </div>
        ) : (
          <div className="text-xs text-ink-400">{t('noPreviewRows')}</div>
        )}
      </div>

      {dataset.managed ? (
        <div className="mt-4 space-y-3">
          <div className="flex items-center justify-between gap-3 text-sm font-medium text-white">
            <span>{t('editCaption')}</span>
            <StatusPill label={`${managedItems.length} ${t('images')}`} tone="neutral" />
          </div>
          <div className="rounded-md border border-aqua-500/20 bg-aqua-500/[0.06] p-3">
            <div className="mb-2 text-sm font-medium text-aqua-300">{t('addImage')}</div>
            <div className="grid gap-2 lg:grid-cols-[1fr_1fr_auto]">
              <input
                type="file"
                multiple
                accept="image/*,.txt,.caption,.json"
                className="min-w-0 rounded-md border border-white/10 bg-black/25 px-3 py-2 text-sm text-ink-300 file:mr-3 file:rounded-md file:border-0 file:bg-aqua-500/20 file:px-2 file:py-1 file:text-aqua-300"
                onChange={event => setAddFiles(Array.from(event.target.files ?? []))}
              />
              <input
                className="min-w-0 rounded-md border border-white/10 bg-black/25 px-3 py-2 text-sm text-white outline-none transition placeholder:text-ink-600 focus:border-aqua-500/50"
                value={addPrompt}
                placeholder={t('captionPlaceholder')}
                onChange={event => setAddPrompt(event.target.value)}
              />
              <button
                className="inline-flex items-center justify-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.12] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/20 disabled:opacity-50"
                disabled={!addFiles.length}
                onClick={() => {
                  onAddItem(dataset.id, addFiles, addPrompt);
                  setAddFiles([]);
                  setAddPrompt('');
                }}
              >
                <Plus className="h-4 w-4" />
                {t('addImage')}
              </button>
            </div>
          </div>
          {managedItems.map(item => (
            <DatasetItemEditor
              key={item.id}
              item={item}
              prompt={itemPrompts[item.id] ?? item.prompt}
              onPromptChange={value => setItemPrompts(current => ({ ...current, [item.id]: value }))}
              onSave={() => onUpdateItem(dataset.id, item.id, itemPrompts[item.id] ?? item.prompt)}
              onDelete={() => onDeleteItem(dataset.id, item.id)}
            />
          ))}
        </div>
      ) : null}

      {issues.length ? (
        <div className="mt-3 space-y-1">
          {issues.slice(0, 3).map(issue => (
            <div key={`${issue.row}-${issue.message}`} className="rounded-md border border-roseSoft-500/20 bg-roseSoft-500/[0.08] px-2 py-1 text-xs text-roseSoft-500">
              {t('rowLabel')} {issue.row}: {issue.message}
            </div>
          ))}
        </div>
      ) : null}
    </article>
  );
}

export default function DatasetValidator() {
  const [datasets, setDatasets] = useState<DatasetResult[]>([]);
  const [selectedDatasetPaths, setSelectedDatasetPaths] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [pathValue, setPathValue] = useState('flux2-klein_self-distill-edit/dataset/corgi/data.jsonl');
  const [recipeId, setRecipeId] = useState<RecipeId>('flux2-klein-identity');
  const [importName, setImportName] = useState('');
  const [importRecipeId, setImportRecipeId] = useState<RecipeId>('flux2-klein-identity');
  const [importFiles, setImportFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const { t } = useI18n();

  const upsertDataset = useCallback((dataset: DatasetResult) => {
    setDatasets(current => [dataset, ...current.filter(item => item.id !== dataset.id && item.path !== dataset.path)]);
  }, []);

  const loadDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/datasets', { cache: 'no-store' });
      const payload = (await response.json()) as DatasetsPayload;
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('datasetApiUnavailable'));
        setDatasets([]);
        return;
      }
      const nextDatasets = payload.datasets ?? [];
      setDatasets(nextDatasets);
      setSelectedDatasetPaths(current => current.filter(datasetPath => nextDatasets.some(dataset => dataset.path === datasetPath)));
    } catch {
      setError(t('datasetApiUnavailable'));
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void loadDatasets();
  }, [loadDatasets]);

  const counts = useMemo(() => {
    const valid = datasets.filter(dataset => dataset.summary.ok).length;
    const rows = datasets.reduce((totalRows, dataset) => totalRows + dataset.summary.rows, 0);
    const issues = datasets.reduce((totalIssues, dataset) => totalIssues + dataset.summary.issues.filter(issue => issue.severity === 'error').length, 0);
    return { total: datasets.length, valid, rows, issues };
  }, [datasets]);
  const selectedDatasetRecipeIds = useMemo(
    () => Array.from(new Set(datasets.filter(dataset => selectedDatasetPaths.includes(dataset.path)).map(dataset => dataset.recipeId))),
    [datasets, selectedDatasetPaths],
  );
  const canCombineSelection = selectedDatasetPaths.length >= 2 && selectedDatasetRecipeIds.length === 1;

  function toggleDatasetPath(datasetPath: string) {
    setSelectedDatasetPaths(current => (
      current.includes(datasetPath)
        ? current.filter(item => item !== datasetPath)
        : [...current, datasetPath]
    ));
  }

  async function validateCustom() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch('/api/datasets', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ path: pathValue, recipeId }),
      });
      const payload = (await response.json()) as ValidatePayload;
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('validationFailed'));
        return;
      }
      upsertDataset(payload.dataset);
    } catch {
      setError(t('validationRequestFailed'));
    } finally {
      setBusy(false);
    }
  }

  async function handleImportDataset() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const formData = new FormData();
      formData.append('action', 'import-files');
      formData.append('name', importName || 'dataset');
      formData.append('recipeId', importRecipeId);
      for (const file of importFiles) {
        formData.append('files', file);
      }
      const response = await fetch('/api/datasets', { method: 'POST', body: formData });
      const payload = (await response.json()) as ValidatePayload;
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('importDatasetFailed'));
        return;
      }
      upsertDataset(payload.dataset);
      setSelectedDatasetPaths(current => Array.from(new Set([payload.dataset.path, ...current])));
      setImportFiles([]);
      setImportName('');
      setNotice(t('datasetImported'));
    } catch {
      setError(t('importDatasetFailed'));
    } finally {
      setBusy(false);
    }
  }

  async function handleAddDatasetItem(datasetId: string, files: File[], prompt: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const formData = new FormData();
      formData.append('action', 'add-item');
      formData.append('datasetId', datasetId);
      formData.append('prompt', prompt);
      for (const file of files) {
        formData.append('files', file);
      }
      const response = await fetch('/api/datasets', { method: 'POST', body: formData });
      const payload = (await response.json()) as ValidatePayload;
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('datasetItemUpdateFailed'));
        return;
      }
      upsertDataset(payload.dataset);
      setNotice(t('datasetItemUpdated'));
    } catch {
      setError(t('datasetItemUpdateFailed'));
    } finally {
      setBusy(false);
    }
  }

  async function handleUpdateDatasetItem(datasetId: string, itemId: string, prompt: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch('/api/datasets', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ action: 'update-item', datasetId, itemId, prompt }),
      });
      const payload = (await response.json()) as ValidatePayload;
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('datasetItemUpdateFailed'));
        return;
      }
      upsertDataset(payload.dataset);
      setNotice(t('datasetItemUpdated'));
    } catch {
      setError(t('datasetItemUpdateFailed'));
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteDatasetItem(datasetId: string, itemId: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch('/api/datasets', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ action: 'delete-item', datasetId, itemId }),
      });
      const payload = (await response.json()) as ValidatePayload;
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('datasetItemUpdateFailed'));
        return;
      }
      upsertDataset(payload.dataset);
      setNotice(t('datasetItemUpdated'));
    } catch {
      setError(t('datasetItemUpdateFailed'));
    } finally {
      setBusy(false);
    }
  }

  async function handleDeleteDataset(datasetId: string) {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const response = await fetch('/api/datasets', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ action: 'delete-dataset', datasetId }),
      });
      const payload = (await response.json()) as { ok: boolean; error?: string };
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('datasetItemUpdateFailed'));
        return;
      }
      setDatasets(current => current.filter(dataset => dataset.id !== datasetId));
      setSelectedDatasetPaths(current => current.filter(datasetPath => datasets.find(dataset => dataset.id === datasetId)?.path !== datasetPath));
      setNotice(t('datasetDeleted'));
    } catch {
      setError(t('datasetItemUpdateFailed'));
    } finally {
      setBusy(false);
    }
  }

  async function combineSelection() {
    if (!selectedDatasetPaths.length) {
      setError(t('selectDatasetFirst'));
      return;
    }
    if (selectedDatasetRecipeIds.length !== 1) {
      setError(t('sameRecipeDatasetRequired'));
      return;
    }
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const selectedRecipe = datasets.find(dataset => dataset.path === selectedDatasetPaths[0])?.recipeId ?? recipeId;
      const response = await fetch('/api/datasets', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ action: 'combine-selection', recipeId: selectedRecipe, datasetPaths: selectedDatasetPaths }),
      });
      const payload = (await response.json()) as ValidatePayload;
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('validationFailed'));
        return;
      }
      upsertDataset(payload.dataset);
      setNotice(t('selectionCombined'));
    } catch {
      setError(t('validationRequestFailed'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <section className="grid gap-3 md:grid-cols-4">
        {[
          [t('datasets'), counts.total],
          [t('valid'), counts.valid],
          [t('rows'), counts.rows],
          [t('issues'), counts.issues],
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
            {counts.issues ? <TriangleAlert className="h-4 w-4 text-roseSoft-500" /> : <CheckCircle2 className="h-4 w-4 text-mint-500" />}
            {t('datasetValidator')}
            <StatusPill label={loading ? t('checking') : counts.issues ? t('review') : t('ready')} tone={loading ? 'neutral' : counts.issues ? 'bad' : 'good'} />
          </div>
          <button
            className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.06] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1]"
            onClick={() => void loadDatasets()}
          >
            <RefreshCw className="h-4 w-4" />
            {t('refresh')}
          </button>
        </div>
      </section>

      <section className="solid-panel rounded-lg p-4">
        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-white">
          <Upload className="h-4 w-4 text-aqua-300" />
          {t('importDataset')}
        </div>
        <div className="grid gap-3 xl:grid-cols-[1fr_220px_1.2fr_auto]">
          <input
            className="min-w-0 rounded-md border border-white/10 bg-black/25 px-3 py-2 text-sm text-white outline-none transition placeholder:text-ink-600 focus:border-aqua-500/50"
            value={importName}
            placeholder={t('datasetNamePlaceholder')}
            onChange={event => setImportName(event.target.value)}
          />
          <select
            className="rounded-md border border-white/10 bg-black/25 px-3 py-2 text-sm text-white outline-none transition focus:border-aqua-500/50"
            value={importRecipeId}
            onChange={event => setImportRecipeId(event.target.value as RecipeId)}
          >
            {recipes.map(recipe => (
              <option key={recipe.id} value={recipe.id}>{recipe.shortName}</option>
            ))}
          </select>
          <input
            type="file"
            multiple
            accept="image/*,.txt,.caption,.json"
            className="min-w-0 rounded-md border border-white/10 bg-black/25 px-3 py-2 text-sm text-ink-300 file:mr-3 file:rounded-md file:border-0 file:bg-aqua-500/20 file:px-2 file:py-1 file:text-aqua-300"
            onChange={event => setImportFiles(Array.from(event.target.files ?? []))}
          />
          <button
            className="inline-flex items-center justify-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.12] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/20 disabled:opacity-50"
            disabled={busy || !importFiles.length}
            onClick={() => void handleImportDataset()}
          >
            <Upload className="h-4 w-4" />
            {t('uploadImages')}
          </button>
        </div>
        <div className="mt-2 text-xs text-ink-400">{t('captionFileHelp')}</div>
      </section>

      <section className="solid-panel rounded-lg p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div className="text-sm font-medium text-white">{t('multiDatasetSelection')}</div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill label={`${t('selectedDatasets')} ${selectedDatasetPaths.length}`} tone="neutral" />
            <button
              className="inline-flex items-center justify-center gap-2 rounded-md border border-mint-500/30 bg-mint-500/[0.12] px-3 py-2 text-sm text-mint-500 transition hover:bg-mint-500/20 disabled:opacity-50"
              disabled={busy || !canCombineSelection}
              onClick={() => void combineSelection()}
            >
              <Plus className="h-4 w-4" />
              {t('createDraftFromMerged')}
            </button>
          </div>
        </div>
        <div className="text-xs leading-5 text-ink-400">{t('multiDatasetHelp')}</div>
      </section>

      <section className="solid-panel rounded-lg p-4">
        <div className="grid gap-3 lg:grid-cols-[1fr_220px_auto]">
          <input
            className="min-w-0 rounded-md border border-white/10 bg-black/25 px-3 py-2 text-sm text-white outline-none transition placeholder:text-ink-600 focus:border-aqua-500/50"
            value={pathValue}
            onChange={event => setPathValue(event.target.value)}
            placeholder={t('datasetPathPlaceholder')}
          />
          <select
            className="rounded-md border border-white/10 bg-black/25 px-3 py-2 text-sm text-white outline-none transition focus:border-aqua-500/50"
            value={recipeId}
            onChange={event => setRecipeId(event.target.value as RecipeId)}
          >
            {recipes.map(recipe => (
              <option key={recipe.id} value={recipe.id}>{recipe.shortName}</option>
            ))}
          </select>
          <button
            className="inline-flex items-center justify-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.12] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/20 disabled:opacity-50"
            disabled={busy}
            onClick={() => void validateCustom()}
          >
            <Search className="h-4 w-4" />
            {t('validate')}
          </button>
        </div>
        {notice ? <div className="mt-3 rounded-md border border-mint-500/25 bg-mint-500/[0.08] px-3 py-2 text-sm text-mint-500">{notice}</div> : null}
        {error ? <div className="mt-3 rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.08] px-3 py-2 text-sm text-roseSoft-500">{error}</div> : null}
      </section>

      {loading ? (
        <div className="solid-panel rounded-lg p-4 text-sm text-ink-400">{t('loadingDataset')}...</div>
      ) : (
        <section className="grid gap-4 xl:grid-cols-3">
          {datasets.map(dataset => (
            <DatasetCard
              key={`${dataset.id}-${dataset.path}`}
              dataset={dataset}
              selected={selectedDatasetPaths.includes(dataset.path)}
              onToggleSelected={() => toggleDatasetPath(dataset.path)}
              onUpdateItem={(datasetId, itemId, prompt) => void handleUpdateDatasetItem(datasetId, itemId, prompt)}
              onDeleteItem={(datasetId, itemId) => void handleDeleteDatasetItem(datasetId, itemId)}
              onAddItem={(datasetId, files, prompt) => void handleAddDatasetItem(datasetId, files, prompt)}
              onDeleteDataset={datasetId => void handleDeleteDataset(datasetId)}
            />
          ))}
        </section>
      )}
    </div>
  );
}
