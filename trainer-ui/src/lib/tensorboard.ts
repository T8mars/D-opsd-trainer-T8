import fs from 'fs/promises';
import path from 'path';
import { resolveProjectRoot } from '@/lib/project';
import type { RecipeId } from '@/lib/recipes';

export const tensorboardMetricKeys = ['lossTotal', 'lossDopsd', 'gradNorm'] as const;
export type TensorboardMetricKey = (typeof tensorboardMetricKeys)[number];

export type TensorboardRunSource = 'job' | 'run-directory' | 'smoke-directory';

export type TensorboardRun = {
  id: string;
  name: string;
  jobId?: string;
  recipeId?: RecipeId;
  recipeName?: string;
  status: string;
  outputDir: string;
  lossPath: string;
  tensorboardDir: string;
  eventFiles: number;
  pointCount: number;
  metricKeys: TensorboardMetricKey[];
  latestStep: number;
  latestLoss?: number;
  bestLoss?: number;
  updatedAt?: string;
  source: TensorboardRunSource;
};

export type TensorboardScalarPoint = {
  step: number;
  value: number;
  raw?: number;
  epoch?: number;
};

export type TensorboardScalarSeries = {
  runId: string;
  runName: string;
  metric: TensorboardMetricKey;
  points: TensorboardScalarPoint[];
  latest?: number;
  min?: number;
  max?: number;
  trend: 'down' | 'up' | 'flat' | 'volatile' | 'unknown';
};

export type ReadScalarSeriesOptions = {
  runIds?: string[];
  metrics?: TensorboardMetricKey[];
  limit?: number;
  stride?: number;
  smooth?: number;
};

type RawJob = {
  id?: string;
  name?: string;
  recipeId?: RecipeId;
  recipeName?: string;
  status?: string;
  outputDir?: string;
  expName?: string;
  updatedAt?: string;
};

type LossPoint = {
  step: number;
  epoch?: number;
  lossTotal?: number;
  lossDopsd?: number;
  gradNorm?: number;
};

const LOSS_LOG_RELATIVE_PATH = 'loss_log/loss_gen_log.jsonl';
const RUNS_RELATIVE_PATH = 'trainer-data/runs';
const SMOKE_RUNS_RELATIVE_PATH = 'trainer-data/smoke-runs';
const MAX_SERIES_LIMIT = 5000;
const DEFAULT_SERIES_LIMIT = 1600;
const DEFAULT_METRICS: TensorboardMetricKey[] = ['lossTotal'];

const metricValueGetters: Record<TensorboardMetricKey, (point: LossPoint) => number | undefined> = {
  lossTotal: point => point.lossTotal,
  lossDopsd: point => point.lossDopsd,
  gradNorm: point => point.gradNorm,
};

function projectRoot() {
  return resolveProjectRoot();
}

function resolveProjectPath(candidate: string) {
  const resolved = path.resolve(path.isAbsolute(candidate) ? candidate : path.join(projectRoot(), candidate));
  const relative = path.relative(projectRoot(), resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error('TensorBoard path is outside the project root');
  }
  return resolved;
}

function projectRelative(candidate: string) {
  return path.relative(projectRoot(), candidate).split(path.sep).join('/');
}

async function exists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function statFile(filePath: string) {
  try {
    return await fs.stat(filePath);
  } catch {
    return null;
  }
}

function maybeNumber(value: unknown) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function parseLossLine(line: string): LossPoint | null {
  if (!line.trim().startsWith('{')) return null;
  try {
    const payload = JSON.parse(line) as Record<string, unknown>;
    const step = maybeNumber(payload.glo_s ?? payload.step) ?? 0;
    return {
      step,
      epoch: maybeNumber(payload.epoch),
      lossTotal: maybeNumber(payload.loss_total),
      lossDopsd: maybeNumber(payload.loss_dopsd),
      gradNorm: maybeNumber(payload.grad_n),
    };
  } catch {
    return null;
  }
}

async function readLossPoints(lossPath: string) {
  const safePath = resolveProjectPath(lossPath);
  if (!(await exists(safePath))) return [];
  const raw = await fs.readFile(safePath, 'utf-8');
  return raw
    .split(/\r?\n/)
    .map(parseLossLine)
    .filter((point): point is LossPoint => Boolean(point));
}

function metricKeysFor(points: LossPoint[]) {
  return tensorboardMetricKeys.filter(key => points.some(point => metricValueGetters[key](point) != null));
}

async function countTensorboardEvents(outputDir: string) {
  const tensorboardDir = path.join(outputDir, 'tensorboard');
  try {
    const entries = await fs.readdir(tensorboardDir, { withFileTypes: true });
    return entries.filter(entry => entry.isFile() && entry.name.startsWith('events.out.tfevents')).length;
  } catch {
    return 0;
  }
}

function runIdFor(source: TensorboardRunSource, key: string) {
  return `${source}:${Buffer.from(key).toString('base64url')}`;
}

async function runFromLossPath(params: {
  source: TensorboardRunSource;
  outputDir: string;
  name?: string;
  job?: RawJob;
}): Promise<TensorboardRun | null> {
  const outputDir = resolveProjectPath(params.outputDir);
  const lossPath = path.join(outputDir, ...LOSS_LOG_RELATIVE_PATH.split('/'));
  if (!(await exists(lossPath))) return null;

  const points = await readLossPoints(lossPath);
  if (!points.length) return null;

  const latest = points[points.length - 1];
  const lossValues = points.map(point => point.lossTotal).filter((value): value is number => value != null);
  const stat = await statFile(lossPath);
  const relativeLossPath = projectRelative(lossPath);
  const eventFiles = await countTensorboardEvents(outputDir);

  return {
    id: params.job?.id ? `job:${params.job.id}` : runIdFor(params.source, relativeLossPath),
    name: params.name || params.job?.name || params.job?.expName || path.basename(outputDir),
    jobId: params.job?.id,
    recipeId: params.job?.recipeId,
    recipeName: params.job?.recipeName,
    status: params.job?.status || 'orphan',
    outputDir: projectRelative(outputDir),
    lossPath: relativeLossPath,
    tensorboardDir: projectRelative(path.join(outputDir, 'tensorboard')),
    eventFiles,
    pointCount: points.length,
    metricKeys: metricKeysFor(points),
    latestStep: latest?.step ?? 0,
    latestLoss: latest?.lossTotal,
    bestLoss: lossValues.length ? Math.min(...lossValues) : undefined,
    updatedAt: params.job?.updatedAt || stat?.mtime.toISOString(),
    source: params.source,
  };
}

async function readJobsLedger() {
  const jobsPath = path.join(projectRoot(), 'trainer-data', 'jobs', 'jobs.json');
  if (!(await exists(jobsPath))) return [];
  try {
    const payload = JSON.parse(await fs.readFile(jobsPath, 'utf-8')) as { jobs?: RawJob[] };
    return Array.isArray(payload.jobs) ? payload.jobs : [];
  } catch {
    return [];
  }
}

async function scanRunDirectory(relativeRoot: string, source: TensorboardRunSource) {
  const root = resolveProjectPath(relativeRoot);
  try {
    const entries = await fs.readdir(root, { withFileTypes: true });
    const runs = await Promise.all(
      entries
        .filter(entry => entry.isDirectory())
        .map(entry => runFromLossPath({
          source,
          outputDir: path.join(root, entry.name),
          name: entry.name,
        })),
    );
    return runs.filter((run): run is TensorboardRun => Boolean(run));
  } catch {
    return [];
  }
}

function sortRuns(runs: TensorboardRun[]) {
  return [...runs].sort((left, right) => {
    const leftTime = left.updatedAt ? Date.parse(left.updatedAt) : 0;
    const rightTime = right.updatedAt ? Date.parse(right.updatedAt) : 0;
    return rightTime - leftTime || right.latestStep - left.latestStep;
  });
}

export async function discoverTensorboardRuns() {
  const jobs = await readJobsLedger();
  const jobRuns = await Promise.all(
    jobs
      .filter(job => Boolean(job.outputDir))
      .map(job => runFromLossPath({
        source: 'job',
        outputDir: job.outputDir || '',
        job,
      })),
  );

  const scannedRuns = await Promise.all([
    scanRunDirectory(RUNS_RELATIVE_PATH, 'run-directory'),
    scanRunDirectory(SMOKE_RUNS_RELATIVE_PATH, 'smoke-directory'),
  ]);

  const byLossPath = new Map<string, TensorboardRun>();
  for (const run of [...jobRuns.filter((run): run is TensorboardRun => Boolean(run)), ...scannedRuns.flat()]) {
    if (!byLossPath.has(run.lossPath) || byLossPath.get(run.lossPath)?.source !== 'job') {
      byLossPath.set(run.lossPath, run);
    }
  }

  return sortRuns([...byLossPath.values()]);
}

function normalizeMetrics(metrics?: TensorboardMetricKey[]) {
  const requested = metrics?.length ? metrics : DEFAULT_METRICS;
  const allowed = new Set<TensorboardMetricKey>(tensorboardMetricKeys);
  return requested.filter((metric): metric is TensorboardMetricKey => allowed.has(metric));
}

function normalizeLimit(value?: number) {
  if (!Number.isFinite(value)) return DEFAULT_SERIES_LIMIT;
  return Math.max(1, Math.min(MAX_SERIES_LIMIT, Math.floor(Number(value))));
}

function normalizeStride(value?: number) {
  if (!Number.isFinite(value)) return 1;
  return Math.max(1, Math.min(200, Math.floor(Number(value))));
}

function normalizeSmooth(value?: number) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(0.95, Number(value)));
}

function smoothPoints(points: TensorboardScalarPoint[], smooth: number) {
  if (smooth <= 0 || points.length < 2) return points;
  let previous: number | undefined;
  return points.map(point => {
    const raw = point.value;
    const value = previous == null ? raw : (previous * smooth) + (raw * (1 - smooth));
    previous = value;
    return { ...point, value, raw };
  });
}

function trendFor(points: TensorboardScalarPoint[]): TensorboardScalarSeries['trend'] {
  if (points.length < 3) return 'unknown';
  const window = points.slice(-Math.min(points.length, 24));
  const first = window[0]?.value;
  const last = window[window.length - 1]?.value;
  if (first == null || last == null) return 'unknown';
  const denominator = Math.max(Math.abs(first), 0.000001);
  const change = (last - first) / denominator;
  let flips = 0;
  let previousDirection = 0;
  for (let index = 1; index < window.length; index += 1) {
    const diff = window[index].value - window[index - 1].value;
    const direction = Math.abs(diff) < denominator * 0.002 ? 0 : Math.sign(diff);
    if (direction && previousDirection && direction !== previousDirection) flips += 1;
    if (direction) previousDirection = direction;
  }
  if (flips >= Math.max(4, Math.floor(window.length / 4))) return 'volatile';
  if (Math.abs(change) < 0.02) return 'flat';
  return change < 0 ? 'down' : 'up';
}

export async function readScalarSeries(options: ReadScalarSeriesOptions = {}) {
  const runs = await discoverTensorboardRuns();
  const runIds = options.runIds?.filter(Boolean) ?? [];
  const selectedRuns = runIds.length ? runs.filter(run => runIds.includes(run.id)) : runs.slice(0, 3);
  const metrics = normalizeMetrics(options.metrics);
  const limit = normalizeLimit(options.limit);
  const stride = normalizeStride(options.stride);
  const smooth = normalizeSmooth(options.smooth);
  const series: TensorboardScalarSeries[] = [];

  for (const run of selectedRuns) {
    const points = await readLossPoints(run.lossPath);
    for (const metric of metrics) {
      const getter = metricValueGetters[metric];
      const metricPoints: TensorboardScalarPoint[] = points
        .reduce<TensorboardScalarPoint[]>((acc, point) => {
          const value = getter(point);
          if (value == null) return acc;
          acc.push({
            step: point.step,
            ...(point.epoch == null ? {} : { epoch: point.epoch }),
            value,
          });
          return acc;
        }, [])
        .filter((_, index) => index % stride === 0)
        .slice(-limit);
      const smoothed = smoothPoints(metricPoints, smooth);
      const values = smoothed.map(point => point.value);
      series.push({
        runId: run.id,
        runName: run.name,
        metric,
        points: smoothed,
        latest: values[values.length - 1],
        min: values.length ? Math.min(...values) : undefined,
        max: values.length ? Math.max(...values) : undefined,
        trend: trendFor(smoothed),
      });
    }
  }

  return {
    runs: selectedRuns,
    series,
    options: { runIds, metrics, limit, stride, smooth },
  };
}
