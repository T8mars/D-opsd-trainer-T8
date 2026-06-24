export type DatasetTrainingOverride = {
  path?: string;
  weight?: number;
};

export type TrainingOverridesV2 = {
  version: 2;
  basics: {
    maxTrainSteps?: number;
    targetResolution?: number;
    resolutionScale?: number;
  };
  lora: {
    rank?: number;
    alpha?: number;
  };
  optimizer: {
    learningRateGen?: number;
    batchSize?: number;
    gradientAccumulationSteps?: number;
    use8bitAdam?: boolean;
  };
  sampling: {
    checkpointSteps?: number;
    sampleSteps?: number;
    sampleResolutionScale?: number;
    finalSampleResolutionScale?: number;
    skipInitialSample?: boolean;
    saveSamples?: boolean;
    saveCheckpoints?: boolean;
  };
  datasets: {
    items?: DatasetTrainingOverride[];
    captionDropout?: number;
  };
  memory: {
    lowVram?: boolean;
    blockOffload?: boolean;
    blockOffloadNumBlocks?: number;
    tileSize?: number;
  };
  advancedDopsd: {
    timestepBias?: 'none' | 'early' | 'late';
    lossWeight?: number;
  };
};

export type LegacyTrainingOverrides = {
  maxTrainSteps?: number;
  epochs?: number;
  learningRateGen?: number;
  batchSize?: number;
  gradientAccumulationSteps?: number;
  checkpointSteps?: number;
  sampleSteps?: number;
  targetResolution?: number;
  resolutionScale?: number;
  sampleResolutionScale?: number;
  finalSampleResolutionScale?: number;
  skipInitialSample?: boolean;
  saveSamples?: boolean;
  saveCheckpoints?: boolean;
  lowVram?: boolean;
  use8bitAdam?: boolean;
  blockOffload?: boolean;
  blockOffloadNumBlocks?: number;
  networkDim?: number;
  rank?: number;
  networkAlpha?: number;
  alpha?: number;
  captionDropout?: number;
  datasets?: DatasetTrainingOverride[];
  tileSize?: number;
};

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

function stringOrUndefined(value: unknown) {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

function dropUndefined<T extends Record<string, unknown>>(value: T) {
  return Object.fromEntries(
    Object.entries(value).filter(([, entry]) => entry !== undefined),
  ) as { [K in keyof T]?: Exclude<T[K], undefined> };
}

function isTrainingOverridesV2(value: unknown): value is TrainingOverridesV2 {
  return Boolean(value && typeof value === 'object' && (value as { version?: unknown }).version === 2);
}

function normalizeDatasetOverrides(datasets: DatasetTrainingOverride[] | undefined) {
  const normalized = (datasets ?? [])
    .map(dataset => dropUndefined({
      path: stringOrUndefined(dataset.path),
      weight: boundedFloat(dataset.weight, 0, 10),
    }))
    .filter(dataset => dataset.path || dataset.weight !== undefined);

  return normalized.length ? normalized : undefined;
}

export function normalizeTrainingConfigV2(config?: TrainingOverridesV2): TrainingOverridesV2 {
  const basics = config?.basics ?? {};
  const lora = config?.lora ?? {};
  const optimizer = config?.optimizer ?? {};
  const sampling = config?.sampling ?? {};
  const datasets = config?.datasets ?? {};
  const memory = config?.memory ?? {};
  const advancedDopsd = config?.advancedDopsd ?? {};

  return {
    version: 2,
    basics: dropUndefined({
      maxTrainSteps: boundedInt(basics.maxTrainSteps, 1, 200000),
      targetResolution: boundedInt(basics.targetResolution, 256, 2048),
      resolutionScale: boundedFloat(basics.resolutionScale, 0.125, 2),
    }),
    lora: dropUndefined({
      rank: boundedInt(lora.rank, 1, 512),
      alpha: boundedInt(lora.alpha, 1, 512),
    }),
    optimizer: dropUndefined({
      learningRateGen: boundedFloat(optimizer.learningRateGen, 1e-8, 1),
      batchSize: boundedInt(optimizer.batchSize, 1, 64),
      gradientAccumulationSteps: boundedInt(optimizer.gradientAccumulationSteps, 1, 1024),
      use8bitAdam: boolOrUndefined(optimizer.use8bitAdam),
    }),
    sampling: dropUndefined({
      checkpointSteps: boundedInt(sampling.checkpointSteps, 1, 200000),
      sampleSteps: boundedInt(sampling.sampleSteps, 0, 200000),
      sampleResolutionScale: boundedFloat(sampling.sampleResolutionScale, 0.125, 2),
      finalSampleResolutionScale: boundedFloat(sampling.finalSampleResolutionScale, 0.125, 2),
      skipInitialSample: boolOrUndefined(sampling.skipInitialSample),
      saveSamples: boolOrUndefined(sampling.saveSamples),
      saveCheckpoints: boolOrUndefined(sampling.saveCheckpoints),
    }),
    datasets: dropUndefined({
      items: normalizeDatasetOverrides(datasets.items),
      captionDropout: boundedFloat(datasets.captionDropout, 0, 1),
    }),
    memory: dropUndefined({
      lowVram: boolOrUndefined(memory.lowVram),
      blockOffload: boolOrUndefined(memory.blockOffload),
      blockOffloadNumBlocks: boundedInt(memory.blockOffloadNumBlocks, 1, 16),
      tileSize: boundedInt(memory.tileSize, 16, 512),
    }),
    advancedDopsd: dropUndefined({
      timestepBias: advancedDopsd.timestepBias,
      lossWeight: boundedFloat(advancedDopsd.lossWeight, 0, 100),
    }),
  };
}

export function migrateTrainingOverridesToV2(legacy?: LegacyTrainingOverrides | TrainingOverridesV2): TrainingOverridesV2 {
  if (isTrainingOverridesV2(legacy)) return normalizeTrainingConfigV2(legacy);
  if (!legacy) return normalizeTrainingConfigV2();

  return normalizeTrainingConfigV2({
    version: 2,
    basics: dropUndefined({
      maxTrainSteps: boundedInt(legacy.maxTrainSteps, 1, 200000),
      targetResolution: boundedInt(legacy.targetResolution, 256, 2048),
      resolutionScale: boundedFloat(legacy.resolutionScale, 0.125, 2),
    }),
    lora: dropUndefined({
      rank: boundedInt(legacy.networkDim ?? legacy.rank, 1, 512),
      alpha: boundedInt(legacy.networkAlpha ?? legacy.alpha, 1, 512),
    }),
    optimizer: dropUndefined({
      learningRateGen: boundedFloat(legacy.learningRateGen, 1e-8, 1),
      batchSize: boundedInt(legacy.batchSize, 1, 64),
      gradientAccumulationSteps: boundedInt(legacy.gradientAccumulationSteps, 1, 1024),
      use8bitAdam: boolOrUndefined(legacy.use8bitAdam),
    }),
    sampling: dropUndefined({
      checkpointSteps: boundedInt(legacy.checkpointSteps, 1, 200000),
      sampleSteps: boundedInt(legacy.sampleSteps, 0, 200000),
      sampleResolutionScale: boundedFloat(legacy.sampleResolutionScale, 0.125, 2),
      finalSampleResolutionScale: boundedFloat(legacy.finalSampleResolutionScale, 0.125, 2),
      skipInitialSample: boolOrUndefined(legacy.skipInitialSample),
      saveSamples: boolOrUndefined(legacy.saveSamples),
      saveCheckpoints: boolOrUndefined(legacy.saveCheckpoints),
    }),
    datasets: dropUndefined({
      items: normalizeDatasetOverrides(legacy.datasets),
      captionDropout: boundedFloat(legacy.captionDropout, 0, 1),
    }),
    memory: dropUndefined({
      lowVram: boolOrUndefined(legacy.lowVram),
      blockOffload: boolOrUndefined(legacy.blockOffload),
      blockOffloadNumBlocks: boundedInt(legacy.blockOffloadNumBlocks, 1, 16),
      tileSize: boundedInt(legacy.tileSize, 16, 512),
    }),
    advancedDopsd: {},
  });
}

export function flattenTrainingConfigForRunner(config?: TrainingOverridesV2): LegacyTrainingOverrides {
  const normalized = normalizeTrainingConfigV2(config);
  return dropUndefined({
    maxTrainSteps: normalized.basics.maxTrainSteps,
    targetResolution: normalized.basics.targetResolution,
    resolutionScale: normalized.basics.resolutionScale,
    networkDim: normalized.lora.rank,
    networkAlpha: normalized.lora.alpha,
    learningRateGen: normalized.optimizer.learningRateGen,
    batchSize: normalized.optimizer.batchSize,
    gradientAccumulationSteps: normalized.optimizer.gradientAccumulationSteps,
    use8bitAdam: normalized.optimizer.use8bitAdam,
    checkpointSteps: normalized.sampling.checkpointSteps,
    sampleSteps: normalized.sampling.sampleSteps,
    sampleResolutionScale: normalized.sampling.sampleResolutionScale,
    finalSampleResolutionScale: normalized.sampling.finalSampleResolutionScale,
    skipInitialSample: normalized.sampling.skipInitialSample,
    saveSamples: normalized.sampling.saveSamples,
    saveCheckpoints: normalized.sampling.saveCheckpoints,
    captionDropout: normalized.datasets.captionDropout,
    lowVram: normalized.memory.lowVram,
    blockOffload: normalized.memory.blockOffload,
    blockOffloadNumBlocks: normalized.memory.blockOffloadNumBlocks,
    tileSize: normalized.memory.tileSize,
  });
}
