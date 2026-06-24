import { execFile } from 'child_process';
import { promisify } from 'util';
import { resolveProjectRoot } from '@/lib/project';

export type SettingsPathItem = {
  id: string;
  label: string;
  path: string;
  exists: boolean;
  purpose: string;
};

export type SafetyDefault = {
  id: string;
  label: string;
  value: string | boolean | number;
  reason: string;
};

export type ProductionProfileSummary = {
  id: string;
  recipe_id: string;
  label: string;
  tier: string;
  hardware_profile: string;
  runner_script: string;
  timeout_seconds: number;
  launcher: string;
  low_vram: boolean;
  use_8bit_adam: boolean;
  block_offload: boolean;
  block_offload_num_blocks: number;
  resolution_scale: string;
  sample_resolution_scale: string | null;
  final_sample_resolution_scale: string | null;
  max_train_steps: number;
  sample_steps: number;
  checkpoint_steps: number;
  save_samples: boolean;
  save_checkpoints: boolean;
  artifact_mode: string;
  evidence: string[];
  cautions: string[];
};

export type SettingsSummary = {
  project_root: string;
  backend: {
    host: string;
    preferred: string;
    distro: string;
    venv: string;
    env_script: string;
  };
  hf_token: {
    present: boolean;
    source: string;
    display: string;
  };
  paths: SettingsPathItem[];
  safety_defaults: SafetyDefault[];
  production_profiles: ProductionProfileSummary[];
  runner_policy: {
    backend: string;
    max_active_jobs: number;
    queue_order: string;
    delete_running_jobs: boolean;
    auto_promote_queued: boolean;
  };
};

const execFileAsync = promisify(execFile);

export function projectRoot() {
  return resolveProjectRoot();
}

export async function readSettingsSummary(): Promise<SettingsSummary> {
  const root = projectRoot();
  const { stdout } = await execFileAsync('python', ['scripts/check_runtime.py', 'settings', '--project-root', root], {
    cwd: root,
    timeout: 20000,
    windowsHide: true,
  });
  return JSON.parse(stdout) as SettingsSummary;
}
