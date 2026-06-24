import { execFile, spawn } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { promisify } from 'util';
import { resolveProjectRoot } from '@/lib/project';

export type ModelSpec = {
  model_id: string;
  role: string;
  default: boolean;
  gated_possible: boolean;
  experimental: boolean;
};

export type ModelCacheSummary = {
  model_id: string;
  registered: boolean;
  spec: ModelSpec | null;
  cache_root: string;
  cache_dir: string;
  cached: boolean;
  snapshots: string[];
  snapshot_count: number;
  size_bytes: number;
  primary_snapshot: string | null;
};

export type ModelDownloadResult = {
  ok: boolean;
  model_id: string;
  path?: string;
  cache_root?: string;
  error?: string;
};

export type CustomModelPathEntry = {
  modelId: string;
  path: string;
  exists: boolean;
  updatedAt: string;
};

export type ModelFolderOpenResult = {
  ok: boolean;
  status: number;
  path?: string;
  error?: string;
};

type CustomModelPathsLedger = {
  version?: number;
  paths?: Record<string, { path: string; updatedAt?: string }>;
};

const execFileAsync = promisify(execFile);

export function projectRoot() {
  return resolveProjectRoot();
}

export function modelCacheRoot() {
  return path.join(projectRoot(), 'trainer-data', 'hf-home', 'hub');
}

function customModelPathsFile() {
  return path.join(projectRoot(), 'trainer-data', 'models', 'custom-model-paths.json');
}

function modelCacheDir(modelId: string) {
  return path.join(modelCacheRoot(), `models--${modelId.replace(/\//g, '--')}`);
}

function resolveModelPath(modelPath: string) {
  const trimmed = modelPath.trim();
  if (!trimmed) return '';
  return path.resolve(projectRoot(), trimmed);
}

async function pathExists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readCustomModelPathsLedger(): Promise<CustomModelPathsLedger> {
  try {
    const raw = await fs.readFile(customModelPathsFile(), 'utf-8');
    const parsed = JSON.parse(raw.replace(/^\uFEFF/, '')) as CustomModelPathsLedger;
    return parsed && typeof parsed === 'object' ? parsed : { version: 1, paths: {} };
  } catch {
    return { version: 1, paths: {} };
  }
}

async function writeCustomModelPathsLedger(ledger: CustomModelPathsLedger) {
  const filePath = customModelPathsFile();
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  const tempPath = path.join(path.dirname(filePath), `custom-model-paths.json.tmp-${process.pid}-${Date.now()}`);
  try {
    await fs.writeFile(tempPath, `${JSON.stringify({ version: 1, paths: ledger.paths ?? {} }, null, 2)}\n`, 'utf-8');
    await fs.rename(tempPath, filePath);
  } catch (error) {
    await fs.unlink(tempPath).catch(() => undefined);
    throw error;
  }
}

export async function readCustomModelPaths(): Promise<CustomModelPathEntry[]> {
  const ledger = await readCustomModelPathsLedger();
  const entries = await Promise.all(
    Object.entries(ledger.paths ?? {}).map(async ([modelId, item]) => {
      const modelPath = resolveModelPath(String(item?.path || ''));
      return {
        modelId,
        path: modelPath,
        exists: modelPath ? await pathExists(modelPath) : false,
        updatedAt: item?.updatedAt || '',
      };
    }),
  );
  return entries.sort((a, b) => a.modelId.localeCompare(b.modelId));
}

export async function saveCustomModelPath(modelId: string, modelPath: string): Promise<CustomModelPathEntry | null> {
  const cleanModelId = modelId.trim();
  if (!cleanModelId) throw new Error('modelId is required');

  const ledger = await readCustomModelPathsLedger();
  const paths = { ...(ledger.paths ?? {}) };
  const resolvedPath = resolveModelPath(modelPath || '');
  if (!resolvedPath) {
    delete paths[cleanModelId];
    await writeCustomModelPathsLedger({ version: 1, paths });
    return null;
  }

  paths[cleanModelId] = {
    path: resolvedPath,
    updatedAt: new Date().toISOString(),
  };
  await writeCustomModelPathsLedger({ version: 1, paths });
  const exists = await pathExists(resolvedPath);
  return {
    modelId: cleanModelId,
    path: resolvedPath,
    exists,
    updatedAt: paths[cleanModelId].updatedAt || '',
  };
}

export async function inspectModels(): Promise<ModelCacheSummary[]> {
  const root = projectRoot();
  const { stdout } = await execFileAsync('python', ['scripts/check_runtime.py', 'models', '--cache-root', modelCacheRoot()], {
    cwd: root,
    timeout: 120000,
    windowsHide: true,
  });
  return JSON.parse(stdout) as ModelCacheSummary[];
}

export async function downloadModel(modelId: string): Promise<ModelDownloadResult> {
  const root = projectRoot();
  const { stdout } = await execFileAsync('python', ['scripts/check_runtime.py', 'download-model', modelId, '--cache-root', modelCacheRoot()], {
    cwd: root,
    timeout: 60 * 60 * 1000,
    windowsHide: true,
  });
  return JSON.parse(stdout) as ModelDownloadResult;
}

export async function openModelFolder(modelId: string, requestedPath?: string): Promise<ModelFolderOpenResult> {
  const cleanModelId = modelId.trim();
  if (!cleanModelId) return { ok: false, status: 400, error: 'modelId is required' };

  const customPaths = await readCustomModelPaths();
  const customPath = customPaths.find(item => item.modelId === cleanModelId)?.path;
  const targetPath = resolveModelPath(requestedPath || customPath || modelCacheDir(cleanModelId));
  if (!targetPath) return { ok: false, status: 400, error: 'Model folder path is required' };

  try {
    const stat = await fs.stat(targetPath);
    if (!stat.isDirectory()) {
      return { ok: false, status: 404, error: 'Model folder does not exist' };
    }
  } catch {
    return { ok: false, status: 404, error: 'Model folder does not exist' };
  }

  const platformCommand =
    process.platform === 'win32'
      ? { command: 'explorer.exe', args: [targetPath] }
      : process.platform === 'darwin'
        ? { command: 'open', args: [targetPath] }
        : { command: 'xdg-open', args: [targetPath] };
  const child = spawn(platformCommand.command, platformCommand.args, {
    detached: true,
    stdio: 'ignore',
    windowsHide: true,
  });
  child.unref();

  return { ok: true, status: 200, path: targetPath };
}
