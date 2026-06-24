'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { CheckCircle2, Cpu, FileJson, HardDrive, ImageIcon, Play, RefreshCw, SlidersHorizontal, TriangleAlert, Wand2, Zap } from 'lucide-react';
import StatusPill from '@/components/StatusPill';
import { useI18n, type MessageKey } from '@/lib/i18n';
import { recipes, type RecipeId } from '@/lib/recipes';
import type { DatasetRegistryItem, DatasetSummary } from '@/lib/datasets';
import type { TrainingOverrides } from '@/lib/jobs';

type DatasetResult = DatasetRegistryItem & {
  summary: DatasetSummary;
};

type DatasetsPayload = {
  ok: boolean;
  datasets: DatasetResult[];
  error?: string;
};

type CreateJobPayload = {
  ok: boolean;
  error?: string;
  job?: {
    id: string;
    name: string;
  };
};

type TrainingForm = Required<Pick<
  TrainingOverrides,
  | 'maxTrainSteps'
  | 'epochs'
  | 'learningRateGen'
  | 'batchSize'
  | 'gradientAccumulationSteps'
  | 'checkpointSteps'
  | 'sampleSteps'
  | 'targetResolution'
  | 'resolutionScale'
  | 'sampleResolutionScale'
  | 'finalSampleResolutionScale'
  | 'skipInitialSample'
  | 'saveSamples'
  | 'saveCheckpoints'
  | 'lowVram'
  | 'use8bitAdam'
  | 'blockOffload'
  | 'blockOffloadNumBlocks'
>>;

type NumericTrainingField =
  | 'maxTrainSteps'
  | 'epochs'
  | 'learningRateGen'
  | 'batchSize'
  | 'gradientAccumulationSteps'
  | 'checkpointSteps'
  | 'sampleSteps'
  | 'targetResolution'
  | 'resolutionScale'
  | 'sampleResolutionScale'
  | 'finalSampleResolutionScale'
  | 'blockOffloadNumBlocks';

type BooleanTrainingField =
  | 'skipInitialSample'
  | 'saveSamples'
  | 'saveCheckpoints'
  | 'lowVram'
  | 'use8bitAdam'
  | 'blockOffload';

const defaultRecipe = recipes.find(recipe => recipe.id === 'flux2-klein-identity') ?? recipes[0];

function defaultTrainingForm(recipe = defaultRecipe): TrainingForm {
  const profile = recipe.productionProfile;
  const maxTrainSteps = recipe.defaultSteps;
  return {
    maxTrainSteps,
    epochs: 2,
    learningRateGen: Number(recipe.defaultLr),
    batchSize: 1,
    gradientAccumulationSteps: 1,
    checkpointSteps: Math.min(maxTrainSteps, 500),
    sampleSteps: Math.min(maxTrainSteps, 500),
    targetResolution: 1024,
    resolutionScale: Number(profile.resolutionScale),
    sampleResolutionScale: Number(profile.sampleResolutionScale ?? 1),
    finalSampleResolutionScale: Number(profile.finalSampleResolutionScale ?? profile.sampleResolutionScale ?? 1),
    skipInitialSample: true,
    saveSamples: profile.saveSamples,
    saveCheckpoints: profile.saveCheckpoints,
    lowVram: true,
    use8bitAdam: true,
    blockOffload: profile.blockOffload,
    blockOffloadNumBlocks: 2,
  };
}

function shortPath(filePath: string) {
  return filePath.replace(/\\/g, '/').split('/').slice(-4).join('/');
}

function statusTone(ok?: boolean): 'good' | 'bad' | 'neutral' {
  if (ok == null) return 'neutral';
  return ok ? 'good' : 'bad';
}

function imageRole(recipeId: RecipeId, index: number, count: number, t: (key: MessageKey) => string) {
  if (recipeId !== 'flux2-klein-editing') return t('target');
  if (index === count - 1) return t('target');
  return index === 0 ? t('reference') : `${t('reference')} ${index + 1}`;
}

function formatBuckets(summary?: DatasetSummary) {
  if (!summary?.bucket_summary.length) return '-';
  return summary.bucket_summary
    .slice(0, 3)
    .map(bucket => `${bucket.width}x${bucket.height} x${bucket.count}`)
    .join(' | ');
}

function formatScale(scale: string | number | undefined, t: (key: MessageKey) => string) {
  if (scale == null) return t('native');
  const value = Number(scale);
  if (!Number.isFinite(value)) return scale;
  return `${Math.round(value * 1000) / 10}%`;
}

function formatNumber(value: number) {
  return Number.isInteger(value) ? String(value) : String(value);
}

export default function NewJobWizard() {
  const { t } = useI18n();
  const [selectedRecipeId, setSelectedRecipeId] = useState<RecipeId>('flux2-klein-identity');
  const [trainingOverrides, setTrainingOverrides] = useState<TrainingForm>(() => defaultTrainingForm(defaultRecipe));
  const [selectedDatasetPaths, setSelectedDatasetPaths] = useState<string[]>([]);
  const [datasets, setDatasets] = useState<DatasetResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createdJobName, setCreatedJobName] = useState<string | null>(null);

  const selectedRecipe = useMemo(
    () => recipes.find(recipe => recipe.id === selectedRecipeId) ?? recipes[0],
    [selectedRecipeId],
  );
  const recipeDatasets = useMemo(
    () => datasets.filter(dataset => dataset.recipeId === selectedRecipeId),
    [datasets, selectedRecipeId],
  );
  const selectedDatasets = useMemo(
    () => recipeDatasets.filter(dataset => selectedDatasetPaths.includes(dataset.path)),
    [recipeDatasets, selectedDatasetPaths],
  );
  const selectedDataset = selectedDatasets[0] ?? recipeDatasets[0];
  const preview = selectedDataset?.summary.previews[0];
  const issueCount = selectedDatasets.reduce(
    (total, dataset) => total + dataset.summary.issues.filter(issue => issue.severity === 'error').length,
    0,
  );
  const selectedRows = selectedDatasets.reduce((total, dataset) => total + dataset.summary.rows, 0);
  const selectedValidRows = selectedDatasets.reduce((total, dataset) => total + dataset.summary.valid_rows, 0);
  const selectionOk = selectedDatasets.length > 0 && selectedDatasets.every(dataset => dataset.summary.ok);
  const selectedProfile = selectedRecipe.productionProfile;
  const effectiveBucketSize = Math.round(trainingOverrides.targetResolution * trainingOverrides.resolutionScale);
  const commandPreview = [
    `EXP_NAME=${selectedRecipe.shortName.toLowerCase().replace(/[^a-z0-9]+/g, '_')}`,
    `TARGET_RESOLUTION=${trainingOverrides.targetResolution}`,
    `RESOLUTION_SCALE=${trainingOverrides.resolutionScale}`,
    `SAMPLE_RESOLUTION_SCALE=${trainingOverrides.sampleResolutionScale}`,
    `FINAL_SAMPLE_RESOLUTION_SCALE=${trainingOverrides.finalSampleResolutionScale}`,
    `MAX_TRAIN_STEPS=${trainingOverrides.maxTrainSteps}`,
    `EPOCHS=${trainingOverrides.epochs}`,
    `SAMPLE_STEPS=${trainingOverrides.sampleSteps}`,
    `CHECKPOINT_STEPS=${trainingOverrides.checkpointSteps}`,
    `LEARNING_RATE_GEN=${trainingOverrides.learningRateGen}`,
    `BATCH_SIZE=${trainingOverrides.batchSize}`,
    `GRADIENT_ACCUMULATION_STEPS=${trainingOverrides.gradientAccumulationSteps}`,
    `SAVE_SAMPLES=${trainingOverrides.saveSamples ? '1' : '0'}`,
    `SAVE_CHECKPOINTS=${trainingOverrides.saveCheckpoints ? '1' : '0'}`,
    `SKIP_INITIAL_SAMPLE=${trainingOverrides.skipInitialSample ? '1' : '0'}`,
    `LOW_VRAM=${trainingOverrides.lowVram ? '1' : '0'}`,
    `USE_8BIT_ADAM=${trainingOverrides.use8bitAdam ? '1' : '0'}`,
    `BLOCK_OFFLOAD=${trainingOverrides.blockOffload ? '1' : '0'}`,
    `BLOCK_OFFLOAD_NUM_BLOCKS=${trainingOverrides.blockOffloadNumBlocks}`,
    'DATA_PATH_TRAIN_JSONL=<selected dataset>',
    `timeout ${Math.max(selectedProfile.timeoutSeconds, Math.min(172800, Math.ceil(trainingOverrides.maxTrainSteps * 180 + 1800)))}`,
    `bash ${selectedProfile.runnerScript}`,
  ].filter(Boolean).join(' ');

  const loadDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/api/datasets', { cache: 'no-store' });
      const payload = (await response.json()) as DatasetsPayload;
      if (!response.ok || !payload.ok) {
        setError(payload.error || t('datasetPreflightFailed'));
        setDatasets([]);
        return;
      }
      setDatasets(payload.datasets ?? []);
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

  useEffect(() => {
    setTrainingOverrides(defaultTrainingForm(selectedRecipe));
  }, [selectedRecipe]);

  useEffect(() => {
    setSelectedDatasetPaths(current => {
      const available = recipeDatasets.map(dataset => dataset.path);
      const retained = current.filter(datasetPath => available.includes(datasetPath));
      if (retained.length) return retained;
      const first = recipeDatasets.find(dataset => dataset.summary.ok) ?? recipeDatasets[0];
      return first ? [first.path] : [];
    });
  }, [recipeDatasets]);

  function toggleDatasetPath(datasetPath: string) {
    setSelectedDatasetPaths(current => (
      current.includes(datasetPath)
        ? current.filter(item => item !== datasetPath)
        : [...current, datasetPath]
    ));
  }

  function updateNumberField(field: NumericTrainingField, value: string) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return;
    setTrainingOverrides(current => ({ ...current, [field]: parsed }));
  }

  function updateBooleanField(field: BooleanTrainingField, checked: boolean) {
    setTrainingOverrides(current => ({ ...current, [field]: checked }));
  }

  async function createDraft() {
    const datasetPaths = selectedDatasets.map(dataset => dataset.path);
    if (!datasetPaths.length) {
      setError(t('selectDatasetFirst'));
      return;
    }
    if (!selectionOk) {
      setError(t('datasetIssuesMustBeFixed'));
      return;
    }

    setBusy(true);
    setError(null);
    setCreatedJobName(null);
    try {
      const response = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ recipeId: selectedRecipeId, datasetPath: datasetPaths[0], datasetPaths, trainingOverrides }),
      });
      const payload = (await response.json()) as CreateJobPayload;
      if (!response.ok || !payload.ok || !payload.job) {
        setError(payload.error || t('draftCouldNotBeCreated'));
        return;
      }
      setCreatedJobName(payload.job.name);
    } catch {
      setError(t('draftRequestFailed'));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-white">{t('newJobTitle')}</h1>
          <p className="mt-1 text-sm text-ink-400">{t('newJobSubtitle')}</p>
        </div>
        <StatusPill label={loading ? t('checkingDatasets') : selectionOk ? t('ready') : t('reviewDataset')} tone={loading ? 'neutral' : statusTone(selectionOk)} />
      </div>

      <div className="grid gap-4 xl:grid-cols-3">
        {recipes.map(recipe => {
          const active = recipe.id === selectedRecipeId;
          const recipeDatasetCount = datasets.filter(item => item.recipeId === recipe.id).length;
          const firstDataset = datasets.find(item => item.recipeId === recipe.id);
          return (
            <section key={recipe.id} className={`solid-panel rounded-lg p-4 ${active ? 'outline outline-1 outline-aqua-500/45' : ''}`}>
              <div className="mb-4 flex items-start justify-between gap-3">
                <div>
                  <h2 className="font-medium text-white">{recipe.name}</h2>
                  <div className="mt-1 text-xs text-ink-400">{recipe.model}</div>
                </div>
                <StatusPill label={recipe.status === 'ready' ? t('ready') : t('advanced')} tone={recipe.status === 'ready' ? 'good' : 'warn'} />
              </div>
              <div className="space-y-3 text-sm text-ink-300">
                <div className="rounded-md border border-white/10 bg-black/20 p-3">
                  <div className="text-xs uppercase tracking-wide text-ink-400">{t('dataset')}</div>
                  <div className="mt-1">{recipe.datasetShape}</div>
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                    <StatusPill label={firstDataset?.summary.ok ? t('preflightOk') : firstDataset ? t('issues') : t('notChecked')} tone={statusTone(firstDataset?.summary.ok)} />
                    <span className="text-ink-400">{recipeDatasetCount ? `${recipeDatasetCount} ${t('datasets')}` : t('loading')}</span>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <div className="rounded-md bg-white/[0.05] p-2">
                    <div className="text-xs text-ink-400">{t('steps')}</div>
                    <div className="text-white">{recipe.defaultSteps}</div>
                  </div>
                  <div className="rounded-md bg-white/[0.05] p-2">
                    <div className="text-xs text-ink-400">{t('lr')}</div>
                    <div className="text-white">{recipe.defaultLr}</div>
                  </div>
                  <div className="rounded-md bg-white/[0.05] p-2">
                    <div className="text-xs text-ink-400">GPU</div>
                    <div className="text-white">{recipe.defaultGpus}</div>
                  </div>
                </div>
                <div className="rounded-md border border-mint-500/20 bg-mint-500/[0.08] p-3">
                  <div className="mb-1 flex items-center gap-2 text-xs uppercase tracking-wide text-mint-500">
                    <Cpu className="h-3.5 w-3.5" />
                    {t('verified16gbProfile')}
                  </div>
                  <div className="text-sm text-ink-200">{recipe.memoryProfile}</div>
                  <div className="mt-2 text-xs text-ink-400">{recipe.productionProfile.maxTrainSteps} {t('steps')} | {t('artifactsOn')}</div>
                </div>
              </div>
              <button
                className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.12] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/20"
                onClick={() => setSelectedRecipeId(recipe.id)}
              >
                <Wand2 className="h-4 w-4" />
                {active ? t('selectedRecipe') : t('configureRecipe')}
              </button>
            </section>
          );
        })}
      </div>

      <section className="solid-panel rounded-lg p-4">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm font-medium text-white">
              <FileJson className="h-4 w-4 text-aqua-300" />
              {t('pairPreflight')}
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-ink-400">
              <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-1">{t('multiDatasetSelection')}</span>
              <span className="text-ink-600">{t('to')}</span>
              <span className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-1">{t('createDraftFromMerged')}</span>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill label={selectedRecipe.shortName} tone="neutral" />
            <StatusPill label={selectionOk ? t('datasetReady') : t('datasetBlocked')} tone={statusTone(selectionOk)} />
            <StatusPill label={`${t('selectedDatasets')} ${selectedDatasets.length}`} tone="neutral" />
            <button
              className="inline-flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.06] px-3 py-2 text-sm text-ink-300 transition hover:bg-white/[0.1]"
              onClick={() => void loadDatasets()}
            >
              <RefreshCw className="h-4 w-4" />
              {t('refresh')}
            </button>
          </div>
        </div>

        <div className="grid gap-4 xl:grid-cols-[1.05fr_0.95fr]">
          <div className="rounded-md border border-white/10 bg-black/20 p-3">
            <div className="mb-3 space-y-2">
              {recipeDatasets.map(dataset => (
                <label key={`${dataset.id}-${dataset.path}`} className="flex items-start gap-3 rounded-md border border-white/10 bg-white/[0.04] p-3">
                  <input
                    type="checkbox"
                    checked={selectedDatasetPaths.includes(dataset.path)}
                    className="mt-1 h-4 w-4 shrink-0 accent-cyan-300"
                    onChange={() => toggleDatasetPath(dataset.path)}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-sm text-white">{dataset.name}</span>
                    <span className="mt-1 block break-all text-xs text-ink-400">{dataset.path}</span>
                  </span>
                  <StatusPill label={dataset.summary.ok ? t('valid') : t('issues')} tone={statusTone(dataset.summary.ok)} />
                </label>
              ))}
              {!recipeDatasets.length ? (
                <div className="rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.08] px-3 py-2 text-sm text-roseSoft-500">
                  {t('selectDatasetFirst')}
                </div>
              ) : null}
            </div>
            <div className="grid gap-2 sm:grid-cols-3">
              <div className="rounded-md bg-white/[0.04] p-2">
                <div className="text-xs text-ink-400">{t('rows')}</div>
                <div className="mt-1 text-sm text-white">{selectedDatasets.length ? `${selectedValidRows} / ${selectedRows}` : '-'}</div>
              </div>
              <div className="rounded-md bg-white/[0.04] p-2">
                <div className="text-xs text-ink-400">{t('issues')}</div>
                <div className="mt-1 text-sm text-white">{issueCount}</div>
              </div>
              <div className="rounded-md bg-white/[0.04] p-2">
                <div className="text-xs text-ink-400">{t('buckets')}</div>
                <div className="mt-1 truncate text-sm text-white" title={formatBuckets(selectedDataset?.summary)}>{selectedDataset?.summary.bucket_summary.length ?? '-'}</div>
              </div>
            </div>
            {issueCount ? (
              <div className="mt-3 rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.08] px-3 py-2 text-sm text-roseSoft-500">
                {t('datasetIssuesMustBeFixed')}
              </div>
            ) : null}
          </div>

          <div className="rounded-md border border-white/10 bg-black/20 p-3">
            <div className="mb-2 flex items-center gap-2 text-xs text-ink-400">
              <ImageIcon className="h-3.5 w-3.5" />
              {formatBuckets(selectedDataset?.summary)}
            </div>
            {preview ? (
              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {preview.images.slice(0, 3).map((image, index) => (
                    <div key={image.path} className="min-w-0">
                      <div className="mb-1 text-[11px] uppercase tracking-wide text-ink-400">{imageRole(selectedRecipeId, index, preview.images.length, t)}</div>
                      {image.url ? (
                        <img
                          src={image.url}
                          alt={`${imageRole(selectedRecipeId, index, preview.images.length, t)} ${shortPath(image.path)}`}
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
        </div>
      </section>

      <section className="solid-panel rounded-lg p-4">
        <div className="mb-4 flex items-center gap-2 text-sm font-medium text-white">
          <SlidersHorizontal className="h-4 w-4 text-aqua-300" />
          {t('memoryAndLaunch')}
        </div>

        <div className="grid gap-4 xl:grid-cols-[0.95fr_1.05fr]">
          <div className="space-y-4">
            <div>
              <div className="mb-2 text-xs uppercase tracking-wide text-ink-400">{t('launcher')}</div>
              <div className="grid grid-cols-3 gap-2">
                {['Python', 'Accelerate', 'DeepSpeed'].map((item, index) => (
                  <button
                    key={item}
                    className={`rounded-md border px-3 py-2 text-sm transition ${
                      index === 0
                        ? 'border-aqua-500/35 bg-aqua-500/[0.14] text-aqua-300'
                        : 'border-white/10 bg-white/[0.04] text-ink-300 hover:bg-white/[0.08]'
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-2 text-xs uppercase tracking-wide text-ink-400">{t('trainingParameters')}</div>
              <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {([
                  ['maxTrainSteps', t('totalSteps'), 1, 200000, 100],
                  ['epochs', t('epochs'), 1, 10000, 1],
                  ['learningRateGen', t('learningRate'), 0.00000001, 1, 0.000001],
                  ['batchSize', t('batchSize'), 1, 64, 1],
                  ['gradientAccumulationSteps', t('gradientAccumulation'), 1, 1024, 1],
                  ['checkpointSteps', t('checkpointInterval'), 1, 200000, 50],
                  ['sampleSteps', t('sampleInterval'), 1, 200000, 50],
                  ['targetResolution', t('targetResolution'), 256, 2048, 64],
                ] satisfies Array<[NumericTrainingField, string, number, number, number]>).map(([field, label, min, max, step]) => (
                  <label key={String(field)} className="rounded-md border border-white/10 bg-white/[0.04] p-2">
                    <span className="block text-xs text-ink-400">{label}</span>
                    <input
                      className="mt-1 w-full rounded-md border border-white/10 bg-black/25 px-2 py-1.5 text-sm text-white outline-none focus:border-aqua-500/50"
                      min={Number(min)}
                      max={Number(max)}
                      step={Number(step)}
                      type="number"
                      value={formatNumber(trainingOverrides[field])}
                      onChange={event => updateNumberField(field, event.target.value)}
                    />
                  </label>
                ))}
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {([
                ['lowVram', t('lowVramOffload')],
                ['use8bitAdam', t('eightBitAdam')],
                ['saveSamples', t('saveSamples')],
                ['saveCheckpoints', t('saveCheckpoints')],
                ['skipInitialSample', t('skipInitialSample')],
                ['blockOffload', t('blockOffload')],
              ] satisfies Array<[BooleanTrainingField, string]>).map(([field, label]) => (
                <label key={String(field)} className="flex min-h-11 items-center gap-3 rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-ink-200">
                  <input
                    checked={trainingOverrides[field]}
                    type="checkbox"
                    className="h-4 w-4 accent-cyan-300"
                    onChange={event => updateBooleanField(field, event.target.checked)}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between gap-3 text-xs uppercase tracking-wide text-ink-400">
                <span>{t('trainingScale')}</span>
                <span className="text-aqua-300">{formatScale(trainingOverrides.resolutionScale, t)}</span>
              </div>
              <input
                value={trainingOverrides.resolutionScale}
                max={1}
                min={0.25}
                step={0.0625}
                type="range"
                className="w-full accent-cyan-300"
                onChange={event => updateNumberField('resolutionScale', event.target.value)}
              />
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              <div className="rounded-md border border-white/10 bg-white/[0.04] p-3">
                <div className="text-xs uppercase tracking-wide text-ink-400">{t('sampleScale')}</div>
                <div className="mt-1 text-sm text-white">{formatScale(trainingOverrides.sampleResolutionScale, t)}</div>
                <input
                  value={trainingOverrides.sampleResolutionScale}
                  max={1}
                  min={0.25}
                  step={0.0625}
                  type="range"
                  className="mt-2 w-full accent-cyan-300"
                  onChange={event => updateNumberField('sampleResolutionScale', event.target.value)}
                />
              </div>
              <div className="rounded-md border border-white/10 bg-white/[0.04] p-3">
                <div className="text-xs uppercase tracking-wide text-ink-400">{t('finalSampleScale')}</div>
                <div className="mt-1 text-sm text-white">{formatScale(trainingOverrides.finalSampleResolutionScale, t)}</div>
                <input
                  value={trainingOverrides.finalSampleResolutionScale}
                  max={1}
                  min={0.25}
                  step={0.0625}
                  type="range"
                  className="mt-2 w-full accent-cyan-300"
                  onChange={event => updateNumberField('finalSampleResolutionScale', event.target.value)}
                />
              </div>
            </div>

            <div>
              <div className="mb-2 flex items-center justify-between gap-3 text-xs uppercase tracking-wide text-ink-400">
                <span>{t('transformerBlocks')}</span>
                <span className="text-aqua-300">{trainingOverrides.blockOffloadNumBlocks} {t('perGroup')}</span>
              </div>
              <input
                value={trainingOverrides.blockOffloadNumBlocks}
                max={8}
                min={1}
                step={1}
                type="range"
                className="w-full accent-cyan-300"
                onChange={event => updateNumberField('blockOffloadNumBlocks', event.target.value)}
              />
            </div>

            <div className="rounded-md border border-amberSoft-500/20 bg-amberSoft-500/[0.07] p-3 text-xs leading-5 text-amberSoft-500">
              {t('effectiveBuckets')}: {effectiveBucketSize}px · {t('longRunHint')}
            </div>
          </div>

          <div className="rounded-md border border-white/10 bg-black/25 p-3">
            <div className="mb-3 flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 text-sm font-medium text-white">
                <HardDrive className="h-4 w-4 text-mint-500" />
                {t('commandPreview')}
              </div>
              <StatusPill label={t('verified16gbProfile')} tone="good" />
            </div>
            <div className="rounded-md bg-black/35 p-3 font-mono text-xs leading-6 text-ink-300">
              {commandPreview}
            </div>
            <div className="mt-2 rounded-md bg-black/25 p-2 text-xs leading-5 text-amberSoft-500">
              {selectedProfile.caution}
            </div>
            <div className="mt-2 rounded-md bg-black/25 p-2 font-mono text-xs leading-5 text-ink-500">
              {t('optionalFallback')}: --block-offload --block-offload-num-blocks 2
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-3">
              {[
                ['VRAM', 'low + tiled VAE'],
                [t('dataset'), selectedDatasets.length > 1 ? t('createDraftFromMerged') : selectedDataset?.name ?? '-'],
                [t('run'), selectedProfile.verifiedRun],
              ].map(([label, value]) => (
                <div key={label} className="rounded-md bg-white/[0.04] p-2">
                  <div className="flex items-center gap-1.5 text-xs text-ink-400">
                    <Zap className="h-3 w-3 text-amberSoft-500" />
                    {label}
                  </div>
                  <div className="mt-1 text-sm text-white">{value}</div>
                </div>
              ))}
            </div>
            <button
              className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.12] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/20 disabled:opacity-50"
              disabled={loading || busy || !selectionOk}
              onClick={() => void createDraft()}
            >
              <Play className="h-4 w-4" />
              {busy ? t('creatingDraft') : selectedDatasets.length > 1 ? t('createDraftFromMerged') : t('createDraft')}
            </button>
            {createdJobName ? (
              <div className="mt-3 rounded-md border border-mint-500/25 bg-mint-500/[0.08] px-3 py-2 text-sm text-mint-500">
                {t('draftCreated')}: {createdJobName}
              </div>
            ) : null}
            {error ? (
              <div className="mt-3 rounded-md border border-roseSoft-500/25 bg-roseSoft-500/[0.08] px-3 py-2 text-sm text-roseSoft-500">
                {error}
              </div>
            ) : null}
          </div>
        </div>
      </section>
    </div>
  );
}
