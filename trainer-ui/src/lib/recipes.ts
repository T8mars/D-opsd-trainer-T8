export type RecipeId = 'z-image-turbo-vlm' | 'flux2-klein-identity' | 'flux2-klein-editing';

export type RecipeProductionProfile = {
  id: string;
  label: string;
  tier: 'recommended_16gb';
  resolutionScale: string;
  sampleResolutionScale?: string;
  finalSampleResolutionScale?: string;
  maxTrainSteps: number;
  sampleSteps: number;
  checkpointSteps: number;
  saveSamples: boolean;
  saveCheckpoints: boolean;
  blockOffload: boolean;
  runnerScript: string;
  timeoutSeconds: number;
  caution: string;
};

export type Recipe = {
  id: RecipeId;
  name: string;
  shortName: string;
  model: string;
  companionModel?: string;
  corePath: string;
  datasetShape: string;
  defaultSteps: number;
  defaultLr: string;
  defaultGpus: number;
  memoryProfile: string;
  productionProfile: RecipeProductionProfile;
  status: 'ready' | 'advanced';
};

export const recipes: Recipe[] = [
  {
    id: 'z-image-turbo-vlm',
    name: 'Z-Image Turbo Style LoRA',
    shortName: 'Z-Image Turbo',
    model: 'Tongyi-MAI/Z-Image-Turbo',
    companionModel: 'Qwen/Qwen3-VL-4B-Instruct',
    corePath: 'z-image-turbo_self-distill-vlm',
    datasetShape: 'single target image per JSONL row',
    defaultSteps: 2000,
    defaultLr: '1e-4',
    defaultGpus: 1,
    memoryProfile: '16GB profile: 0.5 train scale, artifacts on',
    productionProfile: {
      id: 'zimage_turbo_recommended_16gb',
      label: 'Z-Image Turbo 16GB starter',
      tier: 'recommended_16gb',
      resolutionScale: '0.5',
      maxTrainSteps: 2,
      sampleSteps: 2,
      checkpointSteps: 2,
      saveSamples: true,
      saveCheckpoints: true,
      blockOffload: false,
      runnerScript: 'scripts/run_zimage_smoke.sh',
      timeoutSeconds: 3600,
      caution: '0.75 is close to the 16GB ceiling; 1.0 OOMs on this GPU.',
    },
    status: 'ready',
  },
  {
    id: 'flux2-klein-identity',
    name: 'FLUX2 Klein Identity LoRA',
    shortName: 'FLUX2 Identity',
    model: 'black-forest-labs/FLUX.2-klein-4B',
    corePath: 'flux2-klein_self-distill-edit',
    datasetShape: 'single identity target image per JSONL row',
    defaultSteps: 3000,
    defaultLr: '2e-5',
    defaultGpus: 1,
    memoryProfile: '16GB profile: 0.625 train, 0.5 sample',
    productionProfile: {
      id: 'flux2_identity_recommended_16gb',
      label: 'FLUX2 Identity 16GB starter',
      tier: 'recommended_16gb',
      resolutionScale: '0.625',
      sampleResolutionScale: '0.5',
      finalSampleResolutionScale: '0.5',
      maxTrainSteps: 5,
      sampleSteps: 5,
      checkpointSteps: 5,
      saveSamples: true,
      saveCheckpoints: true,
      blockOffload: false,
      runnerScript: 'scripts/run_flux2_smoke.sh',
      timeoutSeconds: 4200,
      caution: 'Native full-size 0.625 inline sample writing OOMed; keep sample scale at 0.5.',
    },
    status: 'ready',
  },
  {
    id: 'flux2-klein-editing',
    name: 'FLUX2 Klein Editing LoRA',
    shortName: 'FLUX2 Editing',
    model: 'black-forest-labs/FLUX.2-klein-4B',
    corePath: 'flux2-klein-edit-self-distill-gt-ref',
    datasetShape: 'reference image(s) plus target image per JSONL row',
    defaultSteps: 3000,
    defaultLr: '2e-5',
    defaultGpus: 1,
    memoryProfile: '16GB profile: 0.5625 train, 0.5 sample',
    productionProfile: {
      id: 'flux2_editing_recommended_16gb',
      label: 'FLUX2 Editing 16GB starter',
      tier: 'recommended_16gb',
      resolutionScale: '0.5625',
      sampleResolutionScale: '0.5',
      maxTrainSteps: 5,
      sampleSteps: 5,
      checkpointSteps: 5,
      saveSamples: true,
      saveCheckpoints: true,
      blockOffload: false,
      runnerScript: 'scripts/run_flux2_editing_smoke.sh',
      timeoutSeconds: 4200,
      caution: 'Higher than 0.5625 and artifact writing beyond five steps need fresh OOM testing.',
    },
    status: 'ready',
  },
];

export function productionProfileForRecipe(recipeId: RecipeId) {
  return recipes.find(recipe => recipe.id === recipeId)?.productionProfile;
}

export const requiredModels = [
  {
    id: 'Tongyi-MAI/Z-Image-Turbo',
    role: 'Z-Image base',
    default: true,
  },
  {
    id: 'Qwen/Qwen3-VL-4B-Instruct',
    role: 'VLM context encoder',
    default: true,
  },
  {
    id: 'black-forest-labs/FLUX.2-klein-4B',
    role: 'FLUX2 Klein base',
    default: true,
  },
  {
    id: 'black-forest-labs/FLUX.2-klein-9B',
    role: 'Experimental larger FLUX2 base',
    default: false,
  },
];
