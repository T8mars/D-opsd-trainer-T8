import crypto from 'crypto';
import { execFile, spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { promisify } from 'util';
import { bundledDatasets, combineDatasetSelections, findRegisteredDataset, validateDataset } from '@/lib/datasets';
import { resolveProjectRoot } from '@/lib/project';
import { productionProfileForRecipe, recipes, type RecipeId, type RecipeProductionProfile } from '@/lib/recipes';
import {
  flattenTrainingConfigForRunner,
  migrateTrainingOverridesToV2,
  type LegacyTrainingOverrides,
  type TrainingOverridesV2,
} from '@/lib/trainingConfig';

export type JobStatus = 'draft' | 'queued' | 'running' | 'completed' | 'failed' | 'stopped';

export type TrainingOverrides = LegacyTrainingOverrides;

export type TrainerJob = {
  id: string;
  name: string;
  recipeId: RecipeId;
  recipeName: string;
  status: JobStatus;
  createdAt: string;
  updatedAt: string;
  expName: string;
  outputDir: string;
  workDir: string;
  gpu: string;
  launcher: 'python' | 'accelerate' | 'deepspeed';
  lowVram: boolean;
  profileId?: string;
  resolutionScale?: number;
  sampleResolutionScale?: number;
  finalSampleResolutionScale?: number;
  targetResolution?: number;
  use8bitAdam: boolean;
  blockOffload?: boolean;
  blockOffloadNumBlocks?: number;
  trainingConfig?: TrainingOverridesV2;
  saveSamples: boolean;
  saveCheckpoints: boolean;
  skipInitialSample?: boolean;
  datasetPath?: string;
  datasetPaths?: string[];
  datasetRows?: number;
  datasetValidRows?: number;
  datasetIssueCount?: number;
  maxTrainSteps: number;
  epochs?: number;
  learningRateGen?: number;
  batchSize?: number;
  gradientAccumulationSteps?: number;
  checkpointSteps?: number;
  sampleSteps?: number;
  currentStep: number;
  command: string;
  notes?: string;
  source: 'seeded-smoke' | 'seeded-failure' | 'ui-draft' | 'cloned' | 'runner-probe';
  queuedAt?: string;
  probeDurationSeconds?: number;
  runner?: RunnerMeta;
};

export type RunnerMeta = {
  windowsPid?: number;
  startedAt?: string;
  stoppedAt?: string;
  finishedAt?: string;
  exitCode?: number | null;
  statusPath?: string;
  logPath?: string;
  linuxPidPath?: string;
  childPidPath?: string;
  scriptPath?: string;
  mode: 'wsl';
};

export type LossPoint = {
  step: number;
  epoch?: number;
  lossDopsd?: number;
  lossTotal?: number;
  gradNorm?: number;
};

export type ArtifactEntry = {
  name: string;
  relativePath: string;
  sizeBytes: number;
  modifiedAt: string;
  isImage: boolean;
  kind: 'sample' | 'trajectory' | 'checkpoint';
};

export type JobSummary = TrainerJob & {
  latestStep: number;
  latestLoss?: number;
  trainableParams?: number;
  failureReason?: string;
  errorTail: string[];
  logTail: string[];
  lossHistory: LossPoint[];
  runnerLogTail: string[];
  runnerPid?: number;
  runnerExitCode?: number | null;
  artifacts: {
    args: boolean;
    log: boolean;
    loss: boolean;
    runnerLog: boolean;
    checkpoints: boolean;
    samples: boolean;
  };
  artifactItems: {
    samples: ArtifactEntry[];
    sampleTrajectories: ArtifactEntry[];
    checkpoints: ArtifactEntry[];
  };
  artifactCounts: {
    samples: number;
    sampleTrajectories: number;
    checkpoints: number;
  };
};

export type JobLogLine = {
  source: 'training' | 'runner' | 'stderr';
  line: string;
};

const SMOKE_EXP = 'flux2_identity_smoke_1step_lowvram_python';
const ZIMAGE_SMOKE_TIMEOUT_SECONDS = 1800;
const FLUX2_EDITING_SMOKE_TIMEOUT_SECONDS = 1800;
const execFileAsync = promisify(execFile);
const ARTIFACT_DIRS = new Set(['samples', 'samples_trajectory', 'checkpoints']);
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp']);
const FAILURE_MARKERS = ['traceback', 'runtimeerror:', 'outofmemory', 'out of memory', 'cuda driver error', 'torch.cuda'];
const MAX_LOG_LINES = 300;
const RUNNER_STARTUP_TIMEOUT_MS = 90000;
const RUNNER_STARTUP_POLL_MS = 500;

type JobsLedger = {
  version?: number;
  jobs?: TrainerJob[];
  knownFailureRunsSeeded?: boolean;
};

type KnownFailureRun = {
  id: string;
  expName: string;
  name: string;
  reason: string;
  currentStep: number;
  notes: string;
};

const KNOWN_FAILURE_RUNS: KnownFailureRun[] = [
  {
    id: 'failed-flux2-editing-sample-vae-encode-oom',
    expName: 'flux2_editing_sample_smoke_1step_lowvram_20260622052603',
    name: 'FLUX2 Editing sample OOM - VAE encode',
    reason: 'CUDA OOM while VAE-encoding unscaled reference images before the resize fix.',
    currentStep: 0,
    notes: 'Real failed smoke captured from FLUX2 Editing sample generation before condition images were resized for VAE encode.',
  },
  {
    id: 'failed-flux2-editing-sample-vae-decode-oom',
    expName: 'flux2_editing_sample_smoke_1step_lowvram_resizedrefs_20260622053926',
    name: 'FLUX2 Editing sample OOM - VAE decode',
    reason: 'CUDA OOM while VAE-decoding post-train samples before training tensors were released.',
    currentStep: 1,
    notes: 'Real failed smoke captured from FLUX2 Editing sample generation before post-train sampling released training tensors.',
  },
];

function projectRoot() {
  return resolveProjectRoot();
}

function jobsDir() {
  return path.join(projectRoot(), 'trainer-data', 'jobs');
}

function jobsPath() {
  return path.join(jobsDir(), 'jobs.json');
}

function runnerDir(jobId: string) {
  return path.join(jobsDir(), 'runner', jobId);
}

function toProjectRelative(filePath: string) {
  return path.relative(projectRoot(), filePath);
}

function toWslPath(filePath: string) {
  const resolved = path.resolve(filePath);
  const parsed = path.parse(resolved);
  const drive = parsed.root.replace(/[:\\]/g, '').toLowerCase();
  const rest = resolved.slice(parsed.root.length).replace(/\\/g, '/');
  return `/mnt/${drive}/${rest}`;
}

function bashQuote(value: string) {
  return `'${value.replace(/'/g, `'\"'\"'`)}'`;
}

async function exists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function directoryHasEntries(dirPath: string) {
  try {
    const entries = await fs.readdir(dirPath);
    return entries.length > 0;
  } catch {
    return false;
  }
}

async function sleep(milliseconds: number) {
  await new Promise(resolve => setTimeout(resolve, milliseconds));
}

function stripAnsi(line: string) {
  return line.replace(/\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])/g, '').trim();
}

function resolveProjectPath(filePath: string) {
  return path.isAbsolute(filePath) ? filePath : path.join(projectRoot(), filePath);
}

function ensureProjectChildPath(filePath: string) {
  const root = path.resolve(projectRoot());
  const resolved = path.resolve(filePath);
  const relative = path.relative(root, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    return null;
  }
  return resolved;
}

function commandForFlux2Smoke(expName: string) {
  const rootWsl = bashQuote(toWslPath(projectRoot()));
  return [
    'wsl -d Ubuntu-22.04 -- bash -lc',
    `"cd ${rootWsl}; EXP_NAME=${expName} timeout 1200 bash scripts/run_flux2_smoke.sh"`,
  ].join(' ');
}

function commandForFlux2EditingSmoke(expName: string) {
  const rootWsl = bashQuote(toWslPath(projectRoot()));
  return [
    'wsl -d Ubuntu-22.04 -- bash -lc',
    `"cd ${rootWsl}; EXP_NAME=${expName} timeout ${FLUX2_EDITING_SMOKE_TIMEOUT_SECONDS} bash scripts/run_flux2_editing_smoke.sh"`,
  ].join(' ');
}

function commandForZImageSmoke(expName: string) {
  const rootWsl = bashQuote(toWslPath(projectRoot()));
  return [
    'wsl -d Ubuntu-22.04 -- bash -lc',
    `"cd ${rootWsl}; EXP_NAME=${expName} timeout ${ZIMAGE_SMOKE_TIMEOUT_SECONDS} bash scripts/run_zimage_smoke.sh"`,
  ].join(' ');
}

function commandForRecipeSmoke(recipeId: RecipeId, expName: string) {
  if (recipeId === 'z-image-turbo-vlm') return commandForZImageSmoke(expName);
  if (recipeId === 'flux2-klein-editing') return commandForFlux2EditingSmoke(expName);
  return commandForFlux2Smoke(expName);
}

function profileEnvAssignments(
  profile: RecipeProductionProfile,
  expName: string,
  outputDir = '../trainer-data/runs',
  datasetEnv = '',
  overrides: TrainingOverrides | TrainingOverridesV2 = {},
) {
  const values = defaultTrainingValues(profile.id.includes('zimage') ? 'z-image-turbo-vlm' : profile.id.includes('editing') ? 'flux2-klein-editing' : 'flux2-klein-identity', profile, overrides);
  return [
    `EXP_NAME=${shellSafeValue(expName)}`,
    `OUTPUT_DIR=${outputDir}`,
    datasetEnv,
    values.resolutionScale ? `RESOLUTION_SCALE=${values.resolutionScale}` : '',
    values.sampleResolutionScale ? `SAMPLE_RESOLUTION_SCALE=${values.sampleResolutionScale}` : '',
    values.finalSampleResolutionScale ? `FINAL_SAMPLE_RESOLUTION_SCALE=${values.finalSampleResolutionScale}` : '',
    `TARGET_RESOLUTION=${values.targetResolution}`,
    `MAX_TRAIN_STEPS=${values.maxTrainSteps}`,
    `EPOCHS=${internalEpochsForMaxTrainSteps(values.maxTrainSteps)}`,
    `SAMPLE_STEPS=${values.sampleSteps}`,
    `CHECKPOINT_STEPS=${values.checkpointSteps}`,
    `LEARNING_RATE_GEN=${values.learningRateGen}`,
    `BATCH_SIZE=${values.batchSize}`,
    `GRADIENT_ACCUMULATION_STEPS=${values.gradientAccumulationSteps}`,
    `SAVE_SAMPLES=${values.saveSamples ? '1' : '0'}`,
    `SAVE_CHECKPOINTS=${values.saveCheckpoints ? '1' : '0'}`,
    `SKIP_INITIAL_SAMPLE=${values.skipInitialSample ? '1' : '0'}`,
    `LOW_VRAM=${values.lowVram ? '1' : '0'}`,
    `USE_8BIT_ADAM=${values.use8bitAdam ? '1' : '0'}`,
    `BLOCK_OFFLOAD=${values.blockOffload ? '1' : '0'}`,
    `BLOCK_OFFLOAD_NUM_BLOCKS=${values.blockOffloadNumBlocks}`,
  ].filter(Boolean).join(' ');
}

function datasetEnvAssignmentsForDatasetPath(datasetPath?: string) {
  if (!datasetPath) return '';
  const datasetWslPath = toWslPath(resolveProjectPath(datasetPath));
  return [
    `DATA_PATH_TRAIN_JSONL=${bashQuote(datasetWslPath)}`,
    `DATA_PATH_TEST_JSONL=${bashQuote(datasetWslPath)}`,
  ].join(' ');
}

function datasetEnvAssignmentsForJob(job: Pick<TrainerJob, 'datasetPath'>) {
  if (!job.datasetPath) return '';
  const datasetWslPath = toWslPath(resolveProjectPath(job.datasetPath));
  return [
    `DATA_PATH_TRAIN_JSONL=${bashQuote(datasetWslPath)}`,
    `DATA_PATH_TEST_JSONL=${bashQuote(datasetWslPath)}`,
  ].join(' ');
}

function commandForProductionProfile(recipeId: RecipeId, expName: string, datasetPath?: string, trainingOverrides?: TrainingOverrides | TrainingOverridesV2) {
  const profile = productionProfileForRecipe(recipeId);
  if (!profile) return commandForRecipeSmoke(recipeId, expName);
  const rootWsl = bashQuote(toWslPath(projectRoot()));
  return [
    'wsl -d Ubuntu-22.04 -- bash -lc',
    `"cd ${rootWsl}; ${profileEnvAssignments(profile, expName, '../trainer-data/runs', datasetEnvAssignmentsForDatasetPath(datasetPath), trainingOverrides)} timeout ${timeoutForOverrides(profile, trainingOverrides)} bash ${profile.runnerScript}"`,
  ].join(' ');
}

function isVerifiedSmokeRecipe(recipeId: RecipeId) {
  return recipeId === 'flux2-klein-identity' || recipeId === 'z-image-turbo-vlm' || recipeId === 'flux2-klein-editing';
}

function smokeOutputDir(expName: string) {
  return path.join('trainer-data', 'smoke-runs', expName);
}

function runOutputDir(expName: string) {
  return path.join('trainer-data', 'runs', expName);
}

function slug(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 48);
}

function timestampSlug(date = new Date()) {
  return date.toISOString().replace(/[-:.TZ]/g, '').slice(0, 17);
}

function finiteNumber(value: unknown) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function boundedInt(value: unknown, min: number, max: number) {
  const numberValue = finiteNumber(value);
  if (numberValue == null) return undefined;
  return Math.min(max, Math.max(min, Math.floor(numberValue)));
}

function boundedFloat(value: unknown, min: number, max: number) {
  const numberValue = finiteNumber(value);
  if (numberValue == null) return undefined;
  return Math.min(max, Math.max(min, numberValue));
}

function boolOrUndefined(value: unknown) {
  return typeof value === 'boolean' ? value : undefined;
}

function normalizeTrainingOverrides(trainingOverrides?: TrainingOverrides | TrainingOverridesV2) {
  return migrateTrainingOverridesToV2(trainingOverrides);
}

function defaultLearningRate(recipeId: RecipeId) {
  const raw = recipes.find(recipe => recipe.id === recipeId)?.defaultLr;
  return finiteNumber(raw) ?? (recipeId === 'z-image-turbo-vlm' ? 1e-4 : 2e-5);
}

function internalEpochsForMaxTrainSteps(maxTrainSteps: number) {
  return Math.max(1, Math.floor(maxTrainSteps) + 1);
}

function defaultTrainingValues(recipeId: RecipeId, profile?: RecipeProductionProfile, config?: TrainingOverrides | TrainingOverridesV2) {
  const recipe = recipes.find(item => item.id === recipeId);
  const overrides = flattenTrainingConfigForRunner(migrateTrainingOverridesToV2(config));
  const maxTrainSteps = overrides?.maxTrainSteps ?? recipe?.defaultSteps ?? profile?.maxTrainSteps ?? 10;
  return {
    maxTrainSteps,
    epochs: internalEpochsForMaxTrainSteps(maxTrainSteps),
    learningRateGen: overrides?.learningRateGen ?? defaultLearningRate(recipeId),
    batchSize: overrides?.batchSize ?? 1,
    gradientAccumulationSteps: overrides?.gradientAccumulationSteps ?? 1,
    checkpointSteps: overrides?.checkpointSteps ?? Math.max(1, Math.min(maxTrainSteps, 500)),
    sampleSteps: overrides?.sampleSteps ?? Math.max(1, Math.min(maxTrainSteps, 500)),
    targetResolution: overrides?.targetResolution ?? 1024,
    resolutionScale: overrides?.resolutionScale ?? (profile ? Number(profile.resolutionScale) : undefined),
    sampleResolutionScale: overrides?.sampleResolutionScale ?? (profile?.sampleResolutionScale ? Number(profile.sampleResolutionScale) : undefined),
    finalSampleResolutionScale: overrides?.finalSampleResolutionScale ?? (profile?.finalSampleResolutionScale ? Number(profile.finalSampleResolutionScale) : undefined),
    skipInitialSample: overrides?.skipInitialSample ?? true,
    saveSamples: overrides?.saveSamples ?? profile?.saveSamples ?? false,
    saveCheckpoints: overrides?.saveCheckpoints ?? profile?.saveCheckpoints ?? false,
    lowVram: overrides?.lowVram ?? Boolean(profile),
    use8bitAdam: overrides?.use8bitAdam ?? Boolean(profile),
    blockOffload: overrides?.blockOffload ?? profile?.blockOffload ?? false,
    blockOffloadNumBlocks: overrides?.blockOffloadNumBlocks ?? 2,
  };
}

function timeoutForStepCount(profile: RecipeProductionProfile, maxTrainSteps: number) {
  const estimated = Math.ceil((maxTrainSteps * 180) + 1800);
  return Math.max(profile.timeoutSeconds, Math.min(estimated, 172800));
}

function timeoutForOverrides(profile: RecipeProductionProfile, overrides?: TrainingOverrides | TrainingOverridesV2) {
  const values = defaultTrainingValues(
    profile.id.includes('zimage') ? 'z-image-turbo-vlm' : profile.id.includes('editing') ? 'flux2-klein-editing' : 'flux2-klein-identity',
    profile,
    overrides,
  );
  return timeoutForStepCount(profile, values.maxTrainSteps);
}

function timeoutForJob(profile: RecipeProductionProfile, job: TrainerJob) {
  const values = defaultTrainingValues(job.recipeId, profile, trainingConfigForJob(job));
  return timeoutForStepCount(profile, values.maxTrainSteps || profile.maxTrainSteps);
}

function trainingConfigForJob(job: TrainerJob) {
  return job.trainingConfig ?? migrateTrainingOverridesToV2(job);
}

function recipeName(recipeId: RecipeId) {
  return recipes.find(recipe => recipe.id === recipeId)?.shortName ?? recipeId;
}

function recipeWorkDir(recipeId: RecipeId) {
  return recipes.find(recipe => recipe.id === recipeId)?.corePath ?? '';
}

async function preflightDraftDataset(recipeId: RecipeId, datasetPath?: string) {
  const registryItem = await findRegisteredDataset(datasetPath, recipeId);
  const isBundledId = bundledDatasets.some(dataset => dataset.id === datasetPath);
  const resolvedDatasetPath = datasetPath && !isBundledId && !registryItem
    ? datasetPath
    : registryItem?.path;

  if (!resolvedDatasetPath) {
    return null;
  }

  const summary = await validateDataset(resolvedDatasetPath, recipeId);
  const datasetIssueCount = summary.issues.filter(issue => issue.severity === 'error').length;
  if (!summary.ok) {
    throw new Error('Dataset issues must be fixed before launch');
  }

  return {
    datasetPath: registryItem?.path ?? resolvedDatasetPath,
    datasetRows: summary.rows,
    datasetValidRows: summary.valid_rows,
    datasetIssueCount,
  };
}

async function preflightDraftDatasetPaths(recipeId: RecipeId, datasetPath?: string, datasetPaths?: string[]) {
  const selectedPaths = Array.from(new Set((datasetPaths?.length ? datasetPaths : datasetPath ? [datasetPath] : []).filter(Boolean)));
  if (selectedPaths.length <= 1) {
    const single = await preflightDraftDataset(recipeId, selectedPaths[0] ?? datasetPath);
    return single ? { ...single, datasetPaths: single.datasetPath ? [single.datasetPath] : [], note: undefined as string | undefined } : null;
  }

  const combined = await combineDatasetSelections(selectedPaths, recipeId);
  if (combined.datasetIssueCount > 0) {
    throw new Error('Dataset issues must be fixed before launch');
  }
  return {
    datasetPath: combined.datasetPath,
    datasetPaths: combined.sourcePaths,
    datasetRows: combined.datasetRows,
    datasetValidRows: combined.datasetValidRows,
    datasetIssueCount: combined.datasetIssueCount,
    note: `Combined dataset selection: ${combined.sourcePaths.length} datasets.`,
  };
}

async function preflightLaunchDataset(job: TrainerJob) {
  if (job.source === 'runner-probe') {
    return job;
  }

  try {
    const datasetPreflight = await preflightDraftDatasetPaths(job.recipeId, job.datasetPath, job.datasetPaths);
    if (!datasetPreflight) return job;

    return {
      ...job,
      datasetPath: datasetPreflight.datasetPath,
      datasetPaths: datasetPreflight.datasetPaths,
      datasetRows: datasetPreflight.datasetRows,
      datasetValidRows: datasetPreflight.datasetValidRows,
      datasetIssueCount: datasetPreflight.datasetIssueCount,
      notes: appendJobNote(
        job.notes,
        `${datasetPreflight.note ? `${datasetPreflight.note} ` : ''}Dataset preflight OK before launch: ${datasetPreflight.datasetValidRows}/${datasetPreflight.datasetRows} rows.`,
      ),
    };
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(
      detail.startsWith('Dataset issues must be fixed before launch')
        ? detail
        : `Dataset issues must be fixed before launch: ${detail}`,
    );
  }
}

async function seedJobs(): Promise<TrainerJob[]> {
  const now = new Date().toISOString();
  const smokeOutput = smokeOutputDir(SMOKE_EXP);
  const smokeRunExists = await exists(resolveProjectPath(smokeOutput));
  if (!smokeRunExists) {
    return [];
  }

  return [
    {
      id: 'verified-flux2-lowvram-smoke',
      name: SMOKE_EXP,
      recipeId: 'flux2-klein-identity',
      recipeName: 'FLUX2 Identity',
      status: 'completed',
      createdAt: now,
      updatedAt: now,
      expName: SMOKE_EXP,
      outputDir: smokeOutput,
      workDir: 'flux2-klein_self-distill-edit',
      gpu: '0',
      launcher: 'python',
      lowVram: true,
      resolutionScale: 0.25,
      use8bitAdam: true,
      saveSamples: false,
      saveCheckpoints: false,
      maxTrainSteps: 1,
      currentStep: 1,
      command: commandForFlux2Smoke(SMOKE_EXP),
      notes: 'One-step low-VRAM reference run on RTX 4060 Ti 16GB.',
      source: 'seeded-smoke',
    },
  ];
}

async function readJobsLedger(): Promise<JobsLedger> {
  return readJobsLedgerWithRetry(jobsPath());
}

async function readJobsLedgerWithRetry(filePath: string): Promise<JobsLedger> {
  if (!(await exists(filePath))) return { version: 1, jobs: [] };

  let lastError: unknown;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const raw = await fs.readFile(filePath, 'utf-8');
      return JSON.parse(raw.replace(/^\uFEFF/, '')) as JobsLedger;
    } catch (error) {
      lastError = error;
      if (attempt < 4) await sleep(50);
    }
  }

  const detail = lastError instanceof Error ? lastError.message : String(lastError);
  throw new Error(`Invalid jobs ledger JSON at ${filePath}: ${detail}`);
}

async function ensureJobsLedger() {
  await fs.mkdir(jobsDir(), { recursive: true });
  if (!(await exists(jobsPath()))) {
    await writeJobs(await seedJobs());
  }
}

async function seedKnownFailureJobs(jobs: TrainerJob[], ledger: JobsLedger): Promise<TrainerJob[]> {
  if (ledger.knownFailureRunsSeeded) return jobs;

  const existingIds = new Set(jobs.map(job => job.id));
  const additions: TrainerJob[] = [];
  for (const failure of KNOWN_FAILURE_RUNS) {
    if (existingIds.has(failure.id)) continue;

    const outputDir = smokeOutputDir(failure.expName);
    const outputPath = resolveProjectPath(outputDir);
    const stderrPath = `${outputPath}.runner.err.log`;
    if (!(await exists(outputPath)) || !(await exists(stderrPath))) continue;

    const stat = await fs.stat(stderrPath).catch(() => null);
    const timestamp = stat?.mtime?.toISOString() || new Date().toISOString();
    additions.push({
      id: failure.id,
      name: failure.name,
      recipeId: 'flux2-klein-editing',
      recipeName: 'FLUX2 Editing',
      status: 'failed',
      createdAt: timestamp,
      updatedAt: timestamp,
      expName: failure.expName,
      outputDir,
      workDir: 'flux2-klein-edit-self-distill-gt-ref',
      gpu: '0',
      launcher: 'python',
      lowVram: true,
      resolutionScale: 0.25,
      use8bitAdam: true,
      saveSamples: true,
      saveCheckpoints: false,
      maxTrainSteps: 1,
      currentStep: failure.currentStep,
      command: `${commandForFlux2EditingSmoke(failure.expName)} # sample-generation failure evidence`,
      notes: `${failure.notes} ${failure.reason}`,
      source: 'seeded-failure',
    });
  }

  const nextJobs = [...additions, ...jobs];
  await writeJobs(nextJobs, { knownFailureRunsSeeded: true });
  return nextJobs;
}

async function readRawJobsLedger(): Promise<TrainerJob[]> {
  await ensureJobsLedger();
  const ledger = await readJobsLedger();
  const jobs = Array.isArray(ledger.jobs) ? ledger.jobs : [];
  return seedKnownFailureJobs(jobs, ledger);
}

async function writeJobs(jobs: TrainerJob[], options: { knownFailureRunsSeeded?: boolean } = {}) {
  await fs.mkdir(jobsDir(), { recursive: true });
  const ledger = await readJobsLedger();
  const knownFailureRunsSeeded = options.knownFailureRunsSeeded ?? ledger.knownFailureRunsSeeded;
  const payload: JobsLedger = { version: 1, jobs };
  if (knownFailureRunsSeeded) payload.knownFailureRunsSeeded = true;
  const tempPath = path.join(jobsDir(), `jobs.json.tmp-${process.pid}-${Date.now()}-${crypto.randomUUID()}`);
  try {
    await fs.writeFile(tempPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf-8');
    await fs.rename(tempPath, jobsPath());
  } catch (error) {
    await fs.unlink(tempPath).catch(() => undefined);
    throw error;
  }
}

export async function readJobs(): Promise<JobSummary[]> {
  const jobs = await promoteNextQueuedJob(await refreshRunnerState(await readRawJobsLedger()));
  return Promise.all(jobs.map(summarizeJob));
}

export async function createDraftJob(
  recipeId: RecipeId = 'flux2-klein-identity',
  datasetPath?: string,
  datasetPaths?: string[],
  trainingOverrides?: TrainingOverrides,
) {
  const jobs = await readRawJobs();
  const now = new Date().toISOString();
  const base = recipes.find(recipe => recipe.id === recipeId) ?? recipes[1];
  const profile = productionProfileForRecipe(base.id);
  const trainingConfig = normalizeTrainingOverrides(trainingOverrides);
  const trainingValues = defaultTrainingValues(base.id, profile, trainingConfig);
  const datasetPreflight = await preflightDraftDatasetPaths(base.id, datasetPath, datasetPaths);
  const expName = `${slug(base.shortName)}_${timestampSlug()}`;
  const verifiedProfileRecipe = Boolean(profile);
  const job: TrainerJob = {
    id: crypto.randomUUID(),
    name: expName,
    recipeId: base.id,
    recipeName: base.shortName,
    status: 'draft',
    createdAt: now,
    updatedAt: now,
    expName,
    outputDir: verifiedProfileRecipe ? runOutputDir(expName) : smokeOutputDir(expName),
    workDir: base.corePath,
    gpu: '0',
    launcher: 'python',
    lowVram: trainingValues.lowVram,
    profileId: profile?.id,
    resolutionScale: trainingValues.resolutionScale,
    sampleResolutionScale: trainingValues.sampleResolutionScale,
    finalSampleResolutionScale: trainingValues.finalSampleResolutionScale,
    targetResolution: trainingValues.targetResolution,
    use8bitAdam: trainingValues.use8bitAdam,
    blockOffload: trainingValues.blockOffload,
    blockOffloadNumBlocks: trainingValues.blockOffloadNumBlocks,
    trainingConfig,
    saveSamples: trainingValues.saveSamples,
    saveCheckpoints: trainingValues.saveCheckpoints,
    skipInitialSample: trainingValues.skipInitialSample,
    datasetPath: datasetPreflight?.datasetPath,
    datasetPaths: datasetPreflight?.datasetPaths,
    datasetRows: datasetPreflight?.datasetRows,
    datasetValidRows: datasetPreflight?.datasetValidRows,
    datasetIssueCount: datasetPreflight?.datasetIssueCount,
    maxTrainSteps: trainingValues.maxTrainSteps,
    epochs: trainingValues.epochs,
    learningRateGen: trainingValues.learningRateGen,
    batchSize: trainingValues.batchSize,
    gradientAccumulationSteps: trainingValues.gradientAccumulationSteps,
    checkpointSteps: trainingValues.checkpointSteps,
    sampleSteps: trainingValues.sampleSteps,
    currentStep: 0,
    command:
      verifiedProfileRecipe
        ? commandForProductionProfile(base.id, expName, datasetPreflight?.datasetPath, trainingConfig)
        : `python scripts/check_runtime.py build-command --recipe-id ${base.id} --exp-name ${expName}`,
    notes: `${profile
      ? `Draft created from the UI. Start uses editable training controls on top of the recommended 16GB memory profile (${profile.label}).`
      : 'Draft created from the UI. Runner launch for this recipe is pending verification.'}${datasetPreflight ? ` ${datasetPreflight.note ? `${datasetPreflight.note} ` : ''}Dataset preflight OK: ${datasetPreflight.datasetValidRows}/${datasetPreflight.datasetRows} rows.` : ''}`,
    source: 'ui-draft',
  };

  await writeJobs([job, ...jobs]);
  return summarizeJob(job);
}

export async function createRunnerProbeJob(durationSeconds = 1) {
  const jobs = await readRawJobs();
  const now = new Date().toISOString();
  const safeDuration = Math.max(0, Math.min(120, Math.floor(durationSeconds)));
  const expName = `runner_probe_${timestampSlug()}`;
  const job: TrainerJob = {
    id: crypto.randomUUID(),
    name: expName,
    recipeId: 'flux2-klein-identity',
    recipeName: 'Runner Probe',
    status: 'draft',
    createdAt: now,
    updatedAt: now,
    expName,
    outputDir: path.join('trainer-data', 'probe-runs', expName),
    workDir: '.',
    gpu: '0',
    launcher: 'python',
    lowVram: false,
    resolutionScale: undefined,
    use8bitAdam: false,
    saveSamples: false,
    saveCheckpoints: false,
    maxTrainSteps: 0,
    currentStep: 0,
    command: `wsl -d Ubuntu-22.04 -- bash -lc "cd ${bashQuote(toWslPath(projectRoot()))}; sleep ${safeDuration}; echo runner probe"`,
    notes: 'Internal short runner probe; safe to delete after verification.',
    source: 'runner-probe',
    probeDurationSeconds: safeDuration,
  };

  await writeJobs([job, ...jobs]);
  return summarizeJob(job);
}

export async function cloneJob(id: string) {
  const jobs = await readRawJobs();
  const source = jobs.find(job => job.id === id);
  if (!source) return null;

  const now = new Date().toISOString();
  const expName = `${slug(source.expName)}_copy_${timestampSlug()}`;
  let cloned: TrainerJob = {
    ...source,
    id: crypto.randomUUID(),
    name: expName,
    status: 'draft',
    createdAt: now,
    updatedAt: now,
    expName,
    outputDir: productionProfileForRecipe(source.recipeId) ? runOutputDir(expName) : smokeOutputDir(expName),
    currentStep: 0,
    command: source.command.replace(source.expName, expName),
    notes: `Cloned from ${source.name}.`,
    source: 'cloned',
    queuedAt: undefined,
    runner: undefined,
  };
  cloned = {
    ...cloned,
    command: productionProfileForRecipe(source.recipeId)
      ? commandForProductionProfile(source.recipeId, expName, source.datasetPath, trainingConfigForJob(cloned))
      : cloned.command,
  };

  await writeJobs([cloned, ...jobs]);
  return summarizeJob(cloned);
}

export async function deleteJob(id: string) {
  const jobs = await readRawJobs();
  const job = jobs.find(item => item.id === id);
  if (!job) return { ok: false, status: 404, error: 'Job not found' };
  if (job.status === 'running' || job.status === 'queued') {
    return { ok: false, status: 409, error: 'Stop the job before deleting it' };
  }
  const next = jobs.filter(item => item.id !== id);
  await writeJobs(next);
  return { ok: true, status: 200 };
}

export async function startJob(id: string) {
  const jobs = await promoteNextQueuedJob(await refreshRunnerState(await readRawJobs()));
  const index = jobs.findIndex(job => job.id === id);
  if (index < 0) return { ok: false, status: 404, error: 'Job not found' };

  let job = jobs[index];
  if (job.status === 'running' || job.status === 'queued') {
    return { ok: true, status: 200, job: await summarizeJob(job) };
  }
  if (job.status === 'completed') {
    return { ok: false, status: 409, error: 'Completed jobs must be cloned before rerun' };
  }
  if (job.source === 'seeded-failure') {
    return { ok: false, status: 409, error: 'Seeded failure evidence jobs must be cloned before rerun' };
  }

  const runnerCommand = runnerCommandForJob(job);
  if (!runnerCommand.ok) {
    return { ok: false, status: 400, error: runnerCommand.error };
  }

  try {
    job = await preflightLaunchDataset(job);
    jobs[index] = job;
  } catch (error) {
    return blockJobForDatasetPreflight(jobs, index, error);
  }

  if (jobs.some(item => item.id !== id && (item.status === 'running' || item.status === 'queued'))) {
    const now = new Date().toISOString();
    const nextJob: TrainerJob = {
      ...job,
      status: 'queued',
      queuedAt: job.queuedAt || now,
      updatedAt: now,
      runner: undefined,
    };
    jobs[index] = nextJob;
    await writeJobs(jobs);
    return { ok: true, status: 202, job: await summarizeJob(nextJob) };
  }

  return launchJob(jobs, index);
}

async function launchJob(jobs: TrainerJob[], index: number) {
  let job = jobs[index];
  try {
    job = await preflightLaunchDataset(job);
    jobs[index] = job;
  } catch (error) {
    return blockJobForDatasetPreflight(jobs, index, error);
  }

  const runnerCommand = runnerCommandForJob(job);
  if (!runnerCommand.ok) {
    return { ok: false, status: 400, error: runnerCommand.error };
  }

  const now = new Date().toISOString();
  const dir = runnerDir(job.id);
  await fs.mkdir(dir, { recursive: true });

  const logPath = path.join(dir, 'runner.log');
  const stdoutPath = path.join(dir, 'runner.stdout.log');
  const stderrPath = path.join(dir, 'runner.stderr.log');
  const statusPath = path.join(dir, 'runner-state.json');
  const linuxPidPath = path.join(dir, 'linux.pid');
  const childPidPath = path.join(dir, 'child.pid');
  const scriptPath = path.join(dir, 'runner.sh');
  const windowsPidPath = path.join(dir, 'windows.pid');
  await Promise.all([
    fs.writeFile(logPath, '', 'utf-8'),
    fs.writeFile(stdoutPath, '', 'utf-8'),
    fs.writeFile(stderrPath, '', 'utf-8'),
  ]);

  const wrapper = buildWslRunnerScript({
    command: runnerCommand.command,
    logPath,
    statusPath,
    linuxPidPath,
    childPidPath,
  });
  await fs.writeFile(scriptPath, wrapper, 'utf-8');

  const windowsPid = await launchDetachedWslRunner({
    scriptPath,
    stdoutPath,
    stderrPath,
    statusPath,
    linuxPidPath,
    windowsPidPath,
    startedAt: now,
  }).catch(async error => {
    const detail = error instanceof Error ? error.message : String(error);
    await fs.appendFile(stderrPath, `Runner startup stderr: ${detail}\n`, 'utf-8').catch(() => undefined);
    return null;
  });

  const nextJob: TrainerJob = {
    ...job,
    status: windowsPid ? 'running' : 'failed',
    updatedAt: now,
    queuedAt: undefined,
    runner: {
      mode: 'wsl',
      windowsPid: windowsPid ?? undefined,
      startedAt: now,
      statusPath: toProjectRelative(statusPath),
      logPath: toProjectRelative(logPath),
      linuxPidPath: toProjectRelative(linuxPidPath),
      childPidPath: toProjectRelative(childPidPath),
      scriptPath: toProjectRelative(scriptPath),
      exitCode: null,
    },
  };
  jobs[index] = nextJob;
  await writeJobs(jobs);

  if (!windowsPid) {
    const failedJob = {
      ...nextJob,
      notes: appendJobNote(nextJob.notes, 'Runner failed to start WSL process. Runner startup stderr is available in runner.stderr.log.'),
    };
    jobs[index] = failedJob;
    await writeRunnerState(failedJob, { status: 'failed', exitCode: null, finishedAt: now });
    await writeJobs(jobs);
    return { ok: false, status: 500, error: 'Runner failed to start WSL process', job: await summarizeJob(failedJob) };
  }

  const startupReady = await waitForRunnerStartup(nextJob);
  if (!startupReady) {
    await stopRunnerProcess(nextJob);
    const failedAt = new Date().toISOString();
    const stderrTail = (await readLogLines(stderrPath)).slice(-6).join(' ');
    const failedJob: TrainerJob = {
      ...nextJob,
      status: 'failed',
      updatedAt: failedAt,
      notes: appendJobNote(
        nextJob.notes,
        `Runner failed to write Linux startup state before timeout. Runner startup stderr: ${stderrTail || 'empty'}`,
      ),
      runner: {
        ...(nextJob.runner as RunnerMeta),
        finishedAt: failedAt,
        exitCode: null,
      },
    };
    jobs[index] = failedJob;
    await writeRunnerState(failedJob, { status: 'failed', exitCode: null, finishedAt: failedAt });
    await writeJobs(jobs);
    return {
      ok: false,
      status: 500,
      error: 'Runner failed to write Linux startup state before timeout',
      job: await summarizeJob(failedJob),
    };
  }

  return { ok: true, status: 200, job: await summarizeJob(nextJob) };
}

async function blockJobForDatasetPreflight(jobs: TrainerJob[], index: number, error: unknown) {
  const detail = error instanceof Error ? error.message : String(error);
  const now = new Date().toISOString();
  const blockedJob: TrainerJob = {
    ...jobs[index],
    status: 'draft',
    updatedAt: now,
    queuedAt: undefined,
    runner: undefined,
    notes: appendJobNote(jobs[index].notes, `Job dataset preflight failed before launch: ${detail}`),
  };
  jobs[index] = blockedJob;
  await writeJobs(jobs);
  return {
    ok: false,
    status: 400,
    error: detail,
    job: await summarizeJob(blockedJob),
  };
}

export async function stopJob(id: string) {
  const jobs = await readRawJobs();
  const index = jobs.findIndex(job => job.id === id);
  if (index < 0) return { ok: false, status: 404, error: 'Job not found' };

  const job = jobs[index];
  if (job.status !== 'running' && job.status !== 'queued') {
    if (job.status === 'stopped' && job.runner) {
      await stopRunnerProcess(job);
    }
    return { ok: true, status: 200, job: await summarizeJob(job) };
  }

  await stopRunnerProcess(job);
  const now = new Date().toISOString();
  const nextJob: TrainerJob = {
    ...job,
    status: 'stopped',
    updatedAt: now,
    runner: job.runner
      ? {
          ...job.runner,
          stoppedAt: now,
          finishedAt: now,
          exitCode: job.runner.exitCode ?? null,
        }
      : undefined,
  };
  await writeRunnerState(nextJob, { status: 'stopped', exitCode: null, finishedAt: now });
  jobs[index] = nextJob;
  await writeJobs(jobs);
  return { ok: true, status: 200, job: await summarizeJob(nextJob) };
}

export async function openJobOutputFolder(id: string) {
  const jobs = await readRawJobs();
  const job = jobs.find(item => item.id === id);
  if (!job) return { ok: false as const, status: 404, error: 'Job not found' };

  const outputDir = ensureProjectChildPath(resolveProjectPath(job.outputDir));
  if (!outputDir) {
    return { ok: false as const, status: 400, error: 'Output folder is outside the project workspace' };
  }

  try {
    const stat = await fs.stat(outputDir);
    if (!stat.isDirectory()) {
      return { ok: false as const, status: 404, error: 'Output folder does not exist yet' };
    }
  } catch {
    return { ok: false as const, status: 404, error: 'Output folder does not exist yet' };
  }

  const child = spawn('explorer.exe', [outputDir], {
    detached: true,
    stdio: 'ignore',
  });
  child.unref();

  return { ok: true as const, status: 200, path: toProjectRelative(outputDir) };
}

async function refreshRunnerState(jobs: TrainerJob[]) {
  let changed = false;
  const nextJobs: TrainerJob[] = [];

  for (const job of jobs) {
    if (!job.runner) {
      nextJobs.push(job);
      continue;
    }

    const runnerState = await readRunnerState(job);
    if (runnerState?.status === 'completed' || runnerState?.status === 'failed' || runnerState?.status === 'stopped') {
      const nextJob: TrainerJob = {
        ...job,
        status: runnerState.status,
        updatedAt: runnerState.finishedAt || job.updatedAt,
        runner: {
          ...job.runner,
          finishedAt: runnerState.finishedAt || job.runner.finishedAt,
          exitCode: runnerState.exitCode,
        },
      };
      changed = changed || JSON.stringify(nextJob) !== JSON.stringify(job);
      nextJobs.push(nextJob);
      continue;
    }

    if ((job.status === 'running' || job.status === 'queued') && job.runner.windowsPid && !(await isWindowsPidAlive(job.runner.windowsPid))) {
      const runnerAliveInWsl = await isLinuxPidAlive(job);
      const linuxPidFileExists = job.runner.linuxPidPath ? await exists(resolveProjectPath(job.runner.linuxPidPath)) : false;
      if (runnerAliveInWsl || linuxPidFileExists) {
        const recoveryNote = runnerAliveInWsl
          ? 'Runner process monitor detached while Linux runner is still alive.'
          : 'Runner process monitor detached while Linux PID check is inconclusive.';
        const nextJob: TrainerJob = {
          ...job,
          status: 'running',
          notes: appendJobNote(job.notes, recoveryNote),
        };
        changed = changed || JSON.stringify(nextJob) !== JSON.stringify(job);
        nextJobs.push(nextJob);
        continue;
      }

      const now = new Date().toISOString();
      const nextJob: TrainerJob = {
        ...job,
        status: 'failed',
        updatedAt: now,
        notes: `${job.notes || ''} Runner process exited before writing completion state.`.trim(),
        runner: {
          ...job.runner,
          finishedAt: now,
          exitCode: job.runner.exitCode ?? null,
        },
      };
      await writeRunnerState(nextJob, { status: 'failed', exitCode: null, finishedAt: now });
      changed = true;
      nextJobs.push(nextJob);
      continue;
    }

    nextJobs.push(job);
  }

  if (changed) {
    await writeJobs(nextJobs);
  }
  return nextJobs;
}

async function promoteNextQueuedJob(jobs: TrainerJob[]) {
  if (jobs.some(job => job.status === 'running')) return jobs;
  const queued = jobs
    .map((job, index) => ({ job, index }))
    .filter(item => item.job.status === 'queued')
    .sort((a, b) => queueTimestamp(a.job).localeCompare(queueTimestamp(b.job)));
  if (!queued.length) return jobs;

  const result = await launchJob(jobs, queued[0].index);
  if (!result.ok) {
    const now = new Date().toISOString();
    const resultError = result.error || 'Queue launch failed';
    const nextJobs = await readRawJobs();
    const nextIndex = nextJobs.findIndex(job => job.id === queued[0].job.id);
    if (resultError.startsWith('Dataset issues must be fixed before launch')) {
      if (nextIndex < 0) return nextJobs;
      nextJobs[nextIndex] = {
        ...nextJobs[nextIndex],
        status: 'draft',
        updatedAt: now,
        queuedAt: undefined,
        notes: appendJobNote(nextJobs[nextIndex].notes, `Queue launch blocked by dataset preflight: ${resultError}`),
      };
      await writeJobs(nextJobs);
      return nextJobs;
    }

    const failedIndex = nextIndex < 0 ? queued[0].index : nextIndex;
    nextJobs[failedIndex] = {
      ...nextJobs[failedIndex],
      status: 'failed',
      updatedAt: now,
      notes: `${nextJobs[failedIndex].notes || ''} Queue launch failed: ${resultError}`.trim(),
      queuedAt: undefined,
    };
    await writeJobs(nextJobs);
    return nextJobs;
  }

  return readRawJobs();
}

function queueTimestamp(job: TrainerJob) {
  return job.queuedAt || job.updatedAt || job.createdAt || '';
}

function runnerCommandForJob(job: TrainerJob): { ok: true; command: string } | { ok: false; error: string } {
  const datasetEnv = datasetEnvAssignmentsForJob(job);
  if (job.source === 'runner-probe') {
    const outputDir = toWslPath(resolveProjectPath(job.outputDir));
    const logPath = `${outputDir}/log.txt`;
    const argsPath = `${outputDir}/args.json`;
    const lossDir = `${outputDir}/loss_log`;
    const lossPath = `${lossDir}/loss_gen_log.jsonl`;
    const samplesDir = `${outputDir}/samples`;
    const trajectoryDir = `${outputDir}/samples_trajectory`;
    const checkpointsDir = `${outputDir}/checkpoints`;
    const samplePath = `${samplesDir}/probe-sample.png`;
    const trajectoryPath = `${trajectoryDir}/probe-trajectory.png`;
    const checkpointPath = `${checkpointsDir}/probe-adapter.safetensors`;
    const tinyPng = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=';
    const duration = Math.max(0, Math.min(120, Math.floor(job.probeDurationSeconds ?? 1)));
    return {
      ok: true,
      command: [
        `mkdir -p ${bashQuote(outputDir)} ${bashQuote(lossDir)} ${bashQuote(samplesDir)} ${bashQuote(trajectoryDir)} ${bashQuote(checkpointsDir)}`,
        `printf '{"probe":true,"durationSeconds":${duration}}\\n' > ${bashQuote(argsPath)}`,
        `printf '[runner] Probe started\\n' > ${bashQuote(logPath)}`,
        duration > 0 ? `sleep ${duration}` : ':',
        `printf '%s' ${bashQuote(tinyPng)} | base64 -d > ${bashQuote(samplePath)}`,
        `printf '%s' ${bashQuote(tinyPng)} | base64 -d > ${bashQuote(trajectoryPath)}`,
        `printf 'probe checkpoint\\n' > ${bashQuote(checkpointPath)}`,
        `printf 'loss for few step generator\\n{"loss_dopsd":0,"loss_total":0,"glo_s":0,"epoch":0,"grad_n":0}\\n' > ${bashQuote(lossPath)}`,
        `printf '[runner] Training completed.\\n' >> ${bashQuote(logPath)}`,
      ].join(' && '),
    };
  }

  const jobConfig = trainingConfigForJob(job);

  if (job.recipeId === 'flux2-klein-identity') {
    const profile = productionProfileForRecipe(job.recipeId);
    if (profile) {
      return {
        ok: true,
        command: `${profileEnvAssignments(profile, job.expName, '../trainer-data/runs', datasetEnv, jobConfig)} timeout ${timeoutForJob(profile, job)} bash ${profile.runnerScript}`,
      };
    }
    return {
      ok: true,
      command: `${datasetEnv} EXP_NAME=${shellSafeValue(job.expName)} timeout 1200 bash scripts/run_flux2_smoke.sh`.trim(),
    };
  }

  if (job.recipeId === 'z-image-turbo-vlm') {
    const profile = productionProfileForRecipe(job.recipeId);
    if (profile) {
      return {
        ok: true,
        command: `${profileEnvAssignments(profile, job.expName, '../trainer-data/runs', datasetEnv, jobConfig)} timeout ${timeoutForJob(profile, job)} bash ${profile.runnerScript}`,
      };
    }
    return {
      ok: true,
      command: `${datasetEnv} EXP_NAME=${shellSafeValue(job.expName)} timeout ${ZIMAGE_SMOKE_TIMEOUT_SECONDS} bash scripts/run_zimage_smoke.sh`.trim(),
    };
  }

  if (job.recipeId === 'flux2-klein-editing') {
    const profile = productionProfileForRecipe(job.recipeId);
    if (profile) {
      return {
        ok: true,
        command: `${profileEnvAssignments(profile, job.expName, '../trainer-data/runs', datasetEnv, jobConfig)} timeout ${timeoutForJob(profile, job)} bash ${profile.runnerScript}`,
      };
    }
    return {
      ok: true,
      command: `${datasetEnv} EXP_NAME=${shellSafeValue(job.expName)} timeout ${FLUX2_EDITING_SMOKE_TIMEOUT_SECONDS} bash scripts/run_flux2_editing_smoke.sh`.trim(),
    };
  }

  return {
    ok: false,
    error: 'Runner start is currently verified only for low-VRAM smoke jobs.',
  };
}

function buildWslRunnerScript({
  command,
  logPath,
  statusPath,
  linuxPidPath,
  childPidPath,
}: {
  command: string;
  logPath: string;
  statusPath: string;
  linuxPidPath: string;
  childPidPath: string;
}) {
  const rootWsl = toWslPath(projectRoot());
  const logWsl = toWslPath(logPath);
  const statusWsl = toWslPath(statusPath);
  const linuxPidWsl = toWslPath(linuxPidPath);
  const childPidWsl = toWslPath(childPidPath);
  const startedAt = new Date().toISOString();

  return [
    '#!/usr/bin/env bash',
    'set -u',
    `cd ${bashQuote(rootWsl)}`,
    `mkdir -p "$(dirname ${bashQuote(logWsl)})"`,
    `echo $$ > ${bashQuote(linuxPidWsl)}`,
    `printf '{"status":"running","startedAt":"${startedAt}","exitCode":null}\\n' > ${bashQuote(statusWsl)}`,
    'child_pid=""',
    'cleanup_child() {',
    '  if [ -n "$child_pid" ]; then',
    '    kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true',
    '  fi',
    '}',
    'trap cleanup_child TERM INT',
    `{`,
    `  echo "[runner] started at ${startedAt}"`,
    `  setsid bash -lc ${bashQuote(command)} &`,
    `  child_pid=$!`,
    `  echo "$child_pid" > ${bashQuote(childPidWsl)}`,
    `  wait "$child_pid"`,
    `  code=$?`,
    `  child_pid=""`,
    `  trap - TERM INT`,
    `  if [ "$code" -eq 0 ]; then`,
    `    status="completed"`,
    `  else`,
    `    status="failed"`,
    `  fi`,
    `  finished_at="$(date -Iseconds)"`,
    `  printf '{"status":"%s","exitCode":%s,"finishedAt":"%s"}\\n' "$status" "$code" "$finished_at" > ${bashQuote(statusWsl)}`,
    `  echo "[runner] finished with status=$status exit=$code at $finished_at"`,
    `  exit "$code"`,
    `} >> ${bashQuote(logWsl)} 2>&1`,
    '',
  ].join('\n');
}

async function launchDetachedWslRunner({
  scriptPath,
  stdoutPath,
  stderrPath,
  statusPath,
  linuxPidPath,
  windowsPidPath,
  startedAt,
}: {
  scriptPath: string;
  stdoutPath: string;
  stderrPath: string;
  statusPath: string;
  linuxPidPath: string;
  windowsPidPath: string;
  startedAt: string;
}) {
  const launchCommand = buildDetachedWslLaunchCommand({
    scriptPath,
    stdoutPath,
    stderrPath,
    statusPath,
    linuxPidPath,
    startedAt,
  });
  const launchScriptPath = path.join(path.dirname(scriptPath), 'launch.sh');
  await fs.writeFile(launchScriptPath, launchCommand, 'utf-8');
  const child = spawn('wsl.exe', ['-d', 'Ubuntu-22.04', '--', 'bash', toWslPath(launchScriptPath)], {
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  });

  const windowsPid = child.pid;
  if (!windowsPid) {
    throw new Error('wsl.exe did not return a Windows PID');
  }

  await fs.writeFile(windowsPidPath, `${windowsPid}\n`, 'utf-8');
  const result = await collectChildOutput(child);
  if (result.stdout) await fs.appendFile(stdoutPath, result.stdout, 'utf-8');
  if (result.stderr) await fs.appendFile(stderrPath, result.stderr, 'utf-8');
  if (result.code !== 0) {
    throw new Error(`wsl.exe launch exited ${result.code}: ${(result.stderr || result.stdout).trim()}`);
  }

  return windowsPid;
}

async function waitForRunnerStartup(job: TrainerJob) {
  const deadline = Date.now() + RUNNER_STARTUP_TIMEOUT_MS;
  while (Date.now() < deadline) {
    const state = await readRunnerState(job);
    if (state) return true;

    if (job.runner?.linuxPidPath && await exists(resolveProjectPath(job.runner.linuxPidPath))) {
      return true;
    }

    await sleep(RUNNER_STARTUP_POLL_MS);
  }
  return false;
}

function buildDetachedWslLaunchCommand({
  scriptPath,
  stdoutPath,
  stderrPath,
  statusPath,
  linuxPidPath,
  startedAt,
}: {
  scriptPath: string;
  stdoutPath: string;
  stderrPath: string;
  statusPath: string;
  linuxPidPath: string;
  startedAt: string;
}) {
  const scriptWsl = toWslPath(scriptPath);
  const stdoutWsl = toWslPath(stdoutPath);
  const stderrWsl = toWslPath(stderrPath);
  const statusWsl = toWslPath(statusPath);
  const linuxPidWsl = toWslPath(linuxPidPath);
  return [
    `cd ${bashQuote(toWslPath(projectRoot()))}`,
    `mkdir -p "$(dirname ${bashQuote(stdoutWsl)})"`,
    `nohup bash ${bashQuote(scriptWsl)} >> ${bashQuote(stdoutWsl)} 2>> ${bashQuote(stderrWsl)} < /dev/null &`,
    'for _runner_start in $(seq 1 100); do',
    `  if [ -s ${bashQuote(linuxPidWsl)} ] || [ -s ${bashQuote(statusWsl)} ]; then break; fi`,
    '  sleep 0.2',
    'done',
    `if [ ! -s ${bashQuote(linuxPidWsl)} ] && [ ! -s ${bashQuote(statusWsl)} ]; then echo "runner startup state missing" >&2; exit 1; fi`,
    `cat ${bashQuote(linuxPidWsl)} 2>/dev/null || true`,
  ].join('\n');
}

async function collectChildOutput(child: ReturnType<typeof spawn>) {
  let stdout = '';
  let stderr = '';
  child.stdout?.on('data', chunk => {
    stdout += chunk.toString();
  });
  child.stderr?.on('data', chunk => {
    stderr += chunk.toString();
  });

  return new Promise<{ code: number | null; stdout: string; stderr: string }>((resolve, reject) => {
    child.once('error', reject);
    child.once('close', code => resolve({ code, stdout, stderr }));
  });
}

async function stopRunnerProcess(job: TrainerJob) {
  await stopWslPidFile(job.runner?.childPidPath);
  await stopWslPidFile(job.runner?.linuxPidPath);
  await killWslJobCommandMatches(job);

  if (job.runner?.windowsPid) {
    await execFileAsync('taskkill.exe', ['/PID', String(job.runner.windowsPid), '/T', '/F'], {
      timeout: 7000,
      windowsHide: true,
    }).catch(() => undefined);
  }
}

async function stopWslPidFile(pidPath?: string) {
  const pid = await readRunnerPidFile(pidPath);
  if (!pid) return;

  await signalWslPidOrGroup(pid, 'TERM');
  for (let attempt = 0; attempt < 20; attempt += 1) {
    if (!(await isWslPidOrGroupAlive(pid))) return;
    await sleep(250);
  }
  await signalWslPidOrGroup(pid, 'KILL');
}

async function readRunnerPidFile(pidPath?: string) {
  if (!pidPath) return null;
  try {
    const raw = await fs.readFile(resolveProjectPath(pidPath), 'utf-8');
    const match = raw.match(/\d+/);
    const pid = match ? Number(match[0]) : NaN;
    return Number.isSafeInteger(pid) && pid > 1 ? pid : null;
  } catch {
    return null;
  }
}

async function signalWslPidOrGroup(pid: number, signal: 'TERM' | 'KILL') {
  await execFileAsync('wsl.exe', ['-d', 'Ubuntu-22.04', '--', '/bin/kill', `-${signal}`, '--', `-${pid}`], {
    timeout: 7000,
    windowsHide: true,
  }).catch(() => undefined);
  await execFileAsync('wsl.exe', ['-d', 'Ubuntu-22.04', '--', '/bin/kill', `-${signal}`, String(pid)], {
    timeout: 7000,
    windowsHide: true,
  }).catch(() => undefined);
}

async function isWslPidOrGroupAlive(pid: number) {
  const command = `kill -0 ${pid} 2>/dev/null || ps -eo pgid= | tr -d ' ' | grep -qx ${bashQuote(String(pid))}`;
  try {
    await execFileAsync('wsl.exe', ['-d', 'Ubuntu-22.04', '--', 'bash', '-lc', command], {
      timeout: 15000,
      windowsHide: true,
    });
    return true;
  } catch {
    return false;
  }
}

async function killWslJobCommandMatches(job: TrainerJob) {
  const marker = shellSafeValue(job.expName);
  if (marker.length < 8) return;

  const command = [
    `marker=${bashQuote(marker)}`,
    `root=${bashQuote(toWslPath(projectRoot()))}`,
    `output=${bashQuote(toWslPath(resolveProjectPath(job.outputDir)))}`,
    `self=$$`,
    `self_pgid="$(ps -o pgid= -p "$self" | tr -d ' ')"`,
    `groups="$(ps -eo pid=,pgid=,cmd= | while read -r pid pgid cmd; do`,
    `  if [ -z "$pid" ] || [ "$pid" = "$self" ] || [ "$pgid" = "$self_pgid" ]; then continue; fi`,
    `  case "$cmd" in`,
    `    *"$marker"*)`,
    `      case "$cmd" in`,
    `        *"$root"*|*"$output"*|*train_dopsd.py*|*run_flux2*|*run_zimage*) echo "$pgid" ;;`,
    `      esac`,
    `      ;;`,
    `  esac`,
    `done | sort -u)"`,
    `for pgid in $groups; do`,
    `  kill -TERM -- "-$pgid" 2>/dev/null || true`,
    `done`,
    `if [ -n "$groups" ]; then sleep 1; fi`,
    `for pgid in $groups; do`,
    `  if ps -eo pgid= | tr -d ' ' | grep -qx "$pgid"; then kill -KILL -- "-$pgid" 2>/dev/null || true; fi`,
    `done`,
  ].join('\n');

  await execFileAsync('wsl.exe', ['-d', 'Ubuntu-22.04', '--', 'bash', '-lc', command], {
    timeout: 15000,
    windowsHide: true,
  }).catch(() => undefined);
}

async function readRunnerState(job: TrainerJob) {
  if (!job.runner?.statusPath) return null;
  const statusPath = resolveProjectPath(job.runner.statusPath);
  if (!(await exists(statusPath))) return null;
  try {
    return JSON.parse(await fs.readFile(statusPath, 'utf-8')) as {
      status: JobStatus;
      exitCode: number | null;
      finishedAt?: string;
    };
  } catch {
    return null;
  }
}

async function writeRunnerState(job: TrainerJob, state: { status: JobStatus; exitCode: number | null; finishedAt: string }) {
  if (!job.runner?.statusPath) return;
  const statusPath = resolveProjectPath(job.runner.statusPath);
  await fs.mkdir(path.dirname(statusPath), { recursive: true });
  await fs.writeFile(statusPath, `${JSON.stringify(state)}\n`, 'utf-8');
}

async function isWindowsPidAlive(pid: number) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

async function isLinuxPidAlive(job: TrainerJob) {
  if (await isWslPidFileAlive(job.runner?.childPidPath, true)) return true;
  return isWslPidFileAlive(job.runner?.linuxPidPath);
}

async function isWslPidFileAlive(pidPath?: string, includeProcessGroup = false) {
  const pid = await readRunnerPidFile(pidPath);
  if (!pid) return false;
  if (includeProcessGroup) return isWslPidOrGroupAlive(pid);
  try {
    await execFileAsync('wsl.exe', ['-d', 'Ubuntu-22.04', '--', '/bin/kill', '-0', String(pid)], {
      timeout: 15000,
      windowsHide: true,
    });
    return true;
  } catch {
    return false;
  }
}

function shellSafeValue(value: string) {
  return value.replace(/[^A-Za-z0-9_.-]/g, '_');
}

function appendJobNote(notes: string | undefined, addition: string) {
  const current = notes || '';
  if (current.includes(addition)) return current;
  return `${current} ${addition}`.trim();
}

async function readRawJobs(): Promise<TrainerJob[]> {
  return readRawJobsLedger();
}

async function summarizeJob(job: TrainerJob): Promise<JobSummary> {
  const outputDir = resolveProjectPath(job.outputDir);
  const argsPath = path.join(outputDir, 'args.json');
  const logPath = path.join(outputDir, 'log.txt');
  const lossPath = path.join(outputDir, 'loss_log', 'loss_gen_log.jsonl');
  const checkpointsPath = path.join(outputDir, 'checkpoints');
  const samplesPath = path.join(outputDir, 'samples');
  const runnerLogPath = job.runner?.logPath ? resolveProjectPath(job.runner.logPath) : '';
  const runnerLogDir = runnerLogPath ? path.dirname(runnerLogPath) : '';
  const runnerStdoutLogPath = runnerLogDir ? path.join(runnerLogDir, 'runner.stdout.log') : '';
  const runnerStderrLogPath = runnerLogDir ? path.join(runnerLogDir, 'runner.stderr.log') : '';
  const siblingRunnerOutLogPath = `${outputDir}.runner.out.log`;
  const siblingRunnerErrLogPath = `${outputDir}.runner.err.log`;
  const runnerLogPaths = [runnerLogPath, runnerStdoutLogPath, runnerStderrLogPath, siblingRunnerOutLogPath, siblingRunnerErrLogPath].filter(Boolean);

  const [logTail, runnerLogTail, rawErrorTail, lossHistory, artifacts, sampleItems, trajectoryItems, checkpointItems] = await Promise.all([
    readLogTail(logPath),
    readCombinedLogTail(runnerLogPaths),
    readErrorTail([logPath, runnerLogPath, runnerStderrLogPath, siblingRunnerErrLogPath].filter(Boolean)),
    readLossHistory(lossPath),
    Promise.all([
      exists(argsPath),
      exists(logPath),
      exists(lossPath),
      runnerLogPath ? exists(runnerLogPath) : Promise.resolve(false),
      runnerStdoutLogPath ? exists(runnerStdoutLogPath) : Promise.resolve(false),
      runnerStderrLogPath ? exists(runnerStderrLogPath) : Promise.resolve(false),
      exists(siblingRunnerOutLogPath),
      exists(siblingRunnerErrLogPath),
      directoryHasEntries(checkpointsPath),
      directoryHasEntries(samplesPath),
    ]),
    listArtifactEntries(outputDir, 'samples', 'sample'),
    listArtifactEntries(outputDir, 'samples_trajectory', 'trajectory'),
    listArtifactEntries(outputDir, 'checkpoints', 'checkpoint'),
  ]);

  const latest = lossHistory[lossHistory.length - 1];
  const runnerState = await readRunnerState(job);
  const runnerFinished = runnerState?.status === 'completed' || runnerState?.status === 'failed' || runnerState?.status === 'stopped';
  const failureReason = extractFailureReason([...logTail, ...runnerLogTail, ...rawErrorTail]);
  const errorTail = failureReason ? (rawErrorTail.length ? rawErrorTail : [...runnerLogTail, ...logTail].slice(-12)) : [];
  const inferredStatus = runnerFinished ? runnerState.status : inferStatus(job.status, logTail, lossHistory, failureReason);
  return {
    ...job,
    recipeName: job.recipeName || recipeName(job.recipeId),
    workDir: job.workDir || recipeWorkDir(job.recipeId),
    status: inferredStatus,
    latestStep: latest?.step ?? job.currentStep,
    latestLoss: latest?.lossTotal,
    trainableParams: parseTrainableParams(logTail),
    failureReason,
    errorTail,
    logTail,
    lossHistory,
    runnerLogTail,
    runnerPid: job.runner?.windowsPid,
    runnerExitCode: runnerState?.exitCode ?? job.runner?.exitCode,
    artifacts: {
      args: artifacts[0],
      log: artifacts[1],
      loss: artifacts[2],
      runnerLog: artifacts[3] || artifacts[4] || artifacts[5] || artifacts[6] || artifacts[7],
      checkpoints: checkpointItems.length > 0,
      samples: sampleItems.length > 0 || trajectoryItems.length > 0,
    },
    artifactItems: {
      samples: sampleItems,
      sampleTrajectories: trajectoryItems,
      checkpoints: checkpointItems,
    },
    artifactCounts: {
      samples: sampleItems.length,
      sampleTrajectories: trajectoryItems.length,
      checkpoints: checkpointItems.length,
    },
  };
}

export async function getJobArtifactFile(id: string, relativePath: string) {
  const jobs = await readRawJobs();
  const job = jobs.find(item => item.id === id);
  if (!job) return { ok: false as const, status: 404, error: 'Job not found' };
  if (!relativePath || path.isAbsolute(relativePath)) {
    return { ok: false as const, status: 400, error: 'Artifact path is required' };
  }

  const outputDir = path.resolve(resolveProjectPath(job.outputDir));
  const filePath = path.resolve(outputDir, relativePath);
  const relative = path.relative(outputDir, filePath);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    return { ok: false as const, status: 400, error: 'Artifact path is outside this job output' };
  }

  const normalized = relative.split(path.sep).join('/');
  const rootDir = normalized.split('/')[0];
  if (!ARTIFACT_DIRS.has(rootDir)) {
    return { ok: false as const, status: 400, error: 'Only sample and checkpoint artifacts can be served' };
  }

  try {
    const stat = await fs.stat(filePath);
    if (!stat.isFile()) return { ok: false as const, status: 404, error: 'Artifact file not found' };
    return {
      ok: true as const,
      status: 200,
      filePath,
      name: path.basename(filePath),
      contentType: contentTypeFor(filePath),
      sizeBytes: stat.size,
    };
  } catch {
    return { ok: false as const, status: 404, error: 'Artifact file not found' };
  }
}

export async function getJobLogs(id: string) {
  const jobs = await readRawJobs();
  const job = jobs.find(item => item.id === id);
  if (!job) return { ok: false as const, status: 404, error: 'Job not found' };

  const outputDir = resolveProjectPath(job.outputDir);
  const trainingLogPath = path.join(outputDir, 'log.txt');
  const runnerLogPath = job.runner?.logPath ? resolveProjectPath(job.runner.logPath) : '';
  const runnerLogDir = runnerLogPath ? path.dirname(runnerLogPath) : '';
  const runnerStdoutLogPath = runnerLogDir ? path.join(runnerLogDir, 'runner.stdout.log') : '';
  const runnerStderrLogPath = runnerLogDir ? path.join(runnerLogDir, 'runner.stderr.log') : '';
  const siblingRunnerOutLogPath = `${outputDir}.runner.out.log`;
  const siblingRunnerErrLogPath = `${outputDir}.runner.err.log`;

  const [training, runnerPrimary, runnerStartupOut, runnerOut, runnerStartupErr, stderr] = await Promise.all([
    readLogLines(trainingLogPath),
    runnerLogPath ? readLogLines(runnerLogPath) : Promise.resolve([]),
    runnerStdoutLogPath ? readLogLines(runnerStdoutLogPath) : Promise.resolve([]),
    readLogLines(siblingRunnerOutLogPath),
    runnerStderrLogPath ? readLogLines(runnerStderrLogPath) : Promise.resolve([]),
    readLogLines(siblingRunnerErrLogPath),
  ]);
  const runner = [...runnerPrimary, ...runnerStartupOut, ...runnerOut];
  const stderrLines = [...runnerStartupErr, ...stderr];
  const combined = tailLogLines([
    ...training.map(line => ({ source: 'training' as const, line })),
    ...runner.map(line => ({ source: 'runner' as const, line })),
    ...stderrLines.map(line => ({ source: 'stderr' as const, line })),
  ]);

  return {
    ok: true as const,
    status: 200,
    generatedAt: new Date().toISOString(),
    jobId: job.id,
    jobName: job.name,
    truncated: training.length + runner.length + stderrLines.length > MAX_LOG_LINES,
    training: tailLogLines(training),
    runner: tailLogLines(runner),
    stderr: tailLogLines(stderrLines),
    combined,
  };
}

async function readLogTail(logPath: string) {
  return (await readLogLines(logPath)).slice(-12);
}

function tailLogLines<T>(lines: T[], limit = MAX_LOG_LINES) {
  return lines.slice(-limit);
}

async function readCombinedLogTail(logPaths: string[]) {
  const lines = (await Promise.all(logPaths.map(readLogLines))).flat();
  return lines.slice(-12);
}

async function readErrorTail(logPaths: string[], limit = 12) {
  const lines = (await Promise.all(logPaths.map(readLogLines))).flat();
  const filtered = lines.filter(line => line && !isProgressLine(line));
  const failureIndex = findLastIndex(filtered, isFailureLine);
  if (failureIndex < 0) return [];
  return filtered.slice(Math.max(0, failureIndex - limit + 1), failureIndex + 1);
}

async function readLogLines(logPath: string) {
  if (!(await exists(logPath))) return [];
  const raw = await fs.readFile(logPath, 'utf-8');
  return raw
    .replace(/\r/g, '\n')
    .split(/\n/)
    .map(stripAnsi)
    .map(line => line.replace(/\s+/g, ' ').trim())
    .filter(Boolean);
}

async function readLossHistory(lossPath: string) {
  if (!(await exists(lossPath))) return [];
  const raw = await fs.readFile(lossPath, 'utf-8');
  const points: LossPoint[] = [];
  for (const rawLine of raw.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line.startsWith('{')) continue;
    try {
      const payload = JSON.parse(line) as Record<string, number>;
      points.push({
        step: Number(payload.glo_s ?? payload.step ?? 0),
        epoch: payload.epoch,
        lossDopsd: maybeNumber(payload.loss_dopsd),
        lossTotal: maybeNumber(payload.loss_total),
        gradNorm: maybeNumber(payload.grad_n),
      });
    } catch {
      // Ignore non-data lines in mixed JSONL logs.
    }
  }
  return points;
}

function inferStatus(status: JobStatus, logTail: string[], lossHistory: LossPoint[], failureReason?: string): JobStatus {
  const joined = logTail.join('\n').toLowerCase();
  if (failureReason) return 'failed';
  if (joined.includes('training completed')) return 'completed';
  if (joined.includes('traceback') || joined.includes('outofmemory') || joined.includes('out of memory')) return 'failed';
  if (status === 'draft' || status === 'queued' || status === 'stopped') return status;
  if (lossHistory.length > 0) return 'running';
  return status;
}

function extractFailureReason(lines: string[]) {
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index];
    if (isProgressLine(line)) continue;
    if (isFailureLine(line)) return line.slice(-320);
  }
  return undefined;
}

function isFailureLine(line: string) {
  const lowered = line.toLowerCase();
  return FAILURE_MARKERS.some(marker => lowered.includes(marker));
}

function isProgressLine(line: string) {
  return line.trimStart().startsWith('Steps:');
}

function findLastIndex<T>(items: T[], predicate: (item: T) => boolean) {
  for (let index = items.length - 1; index >= 0; index -= 1) {
    if (predicate(items[index])) return index;
  }
  return -1;
}

function parseTrainableParams(logTail: string[]) {
  for (const line of logTail) {
    const match = line.match(/Total trainable parameters in gen_model:\s*(\d+)/);
    if (match) return Number(match[1]);
  }
  return undefined;
}

function maybeNumber(value: unknown) {
  const numberValue = Number(value);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

async function listArtifactEntries(
  outputDir: string,
  dirname: 'samples' | 'samples_trajectory' | 'checkpoints',
  kind: ArtifactEntry['kind'],
  limit = 200,
): Promise<ArtifactEntry[]> {
  const root = path.join(outputDir, dirname);
  const files = await listFiles(root);
  const entries: ArtifactEntry[] = [];

  for (const filePath of files) {
    try {
      const stat = await fs.stat(filePath);
      const relativePath = path.relative(outputDir, filePath).split(path.sep).join('/');
      entries.push({
        name: path.basename(filePath),
        relativePath,
        sizeBytes: stat.size,
        modifiedAt: stat.mtime.toISOString(),
        isImage: IMAGE_EXTENSIONS.has(path.extname(filePath).toLowerCase()),
        kind,
      });
    } catch {
      // Artifact directories can change while a job is writing; skip files that disappear mid-read.
    }
  }

  return entries
    .sort((a, b) => b.modifiedAt.localeCompare(a.modifiedAt) || a.relativePath.localeCompare(b.relativePath))
    .slice(0, limit);
}

async function listFiles(dirPath: string): Promise<string[]> {
  try {
    const entries = await fs.readdir(dirPath, { withFileTypes: true });
    const nested = await Promise.all(entries.map(async entry => {
      const entryPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) return listFiles(entryPath);
      return entry.isFile() ? [entryPath] : [];
    }));
    return nested.flat();
  } catch {
    return [];
  }
}

function contentTypeFor(filePath: string) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === '.png') return 'image/png';
  if (ext === '.jpg' || ext === '.jpeg') return 'image/jpeg';
  if (ext === '.webp') return 'image/webp';
  if (ext === '.json') return 'application/json; charset=utf-8';
  if (ext === '.txt' || ext === '.log') return 'text/plain; charset=utf-8';
  if (ext === '.safetensors') return 'application/octet-stream';
  return 'application/octet-stream';
}
