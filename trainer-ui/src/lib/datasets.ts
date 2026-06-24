import crypto from 'crypto';
import { execFile } from 'child_process';
import fs from 'fs/promises';
import path from 'path';
import { promisify } from 'util';
import { resolveProjectRoot } from '@/lib/project';
import type { RecipeId } from '@/lib/recipes';

export type DatasetIssue = {
  row: number;
  severity: 'error' | 'warning';
  message: string;
};

export type DatasetImagePreview = {
  path: string;
  url?: string;
  width?: number;
  height?: number;
};

export type DatasetPreview = {
  row: number;
  class_name?: string | null;
  prompt_key?: string | null;
  prompt: string;
  image_paths: string[];
  image_sizes: Array<[number, number]>;
  images: DatasetImagePreview[];
};

export type DatasetBucket = {
  width: number;
  height: number;
  count: number;
};

export type DatasetSummary = {
  path: string;
  recipe_id: RecipeId;
  rows: number;
  valid_rows: number;
  issues: DatasetIssue[];
  previews: DatasetPreview[];
  bucket_summary: DatasetBucket[];
  ok: boolean;
};

export type ManagedDatasetItem = {
  id: string;
  imagePaths: string[];
  imageSizes?: Array<[number, number]>;
  captionPath?: string;
  prompt: string;
  createdAt: string;
  updatedAt: string;
};

export type DatasetRegistryItem = {
  id: string;
  name: string;
  path: string;
  recipeId: RecipeId;
  shape: string;
  source?: 'bundled' | 'managed' | 'custom' | 'combined';
  managed?: boolean;
  createdAt?: string;
  updatedAt?: string;
  itemCount?: number;
  items?: ManagedDatasetItem[];
};

type ManagedDatasetsLedger = {
  version?: number;
  datasets?: DatasetRegistryItem[];
};

type SavedUploadImage = {
  projectPath: string;
  absolutePath: string;
  stem: string;
  groupKey: string;
  roleRank: number;
};

export type CombinedDatasetSelection = {
  datasetPath: string;
  datasetRows: number;
  datasetValidRows: number;
  datasetIssueCount: number;
  sourcePaths: string[];
};

const execFileAsync = promisify(execFile);
const IMAGE_EXTENSIONS = new Set(['.jpg', '.jpeg', '.png', '.webp', '.bmp']);
const CAPTION_EXTENSIONS = new Set(['.txt', '.caption', '.json']);
const JSONL_IMAGE_KEYS = ['local_path_list', 'local_paths', 'image_path_list', 'image_paths', 'image_path', 'path'];

export const bundledDatasets: DatasetRegistryItem[] = [
  {
    id: 'style-millennium',
    name: 'style_Millennium',
    path: 'z-image-turbo_self-distill-vlm/dataset/style_Millennium/data.jsonl',
    recipeId: 'z-image-turbo-vlm',
    shape: 'single image',
    source: 'bundled',
  },
  {
    id: 'corgi',
    name: 'corgi',
    path: 'flux2-klein_self-distill-edit/dataset/corgi/data.jsonl',
    recipeId: 'flux2-klein-identity',
    shape: 'single image',
    source: 'bundled',
  },
  {
    id: 'interaction',
    name: 'interaction',
    path: 'flux2-klein-edit-self-distill-gt-ref/dataset/interaction/data.jsonl',
    recipeId: 'flux2-klein-editing',
    shape: 'reference + target',
    source: 'bundled',
  },
];

export function projectRoot() {
  return resolveProjectRoot();
}

export function resolveProjectPath(rawPath: string) {
  const root = path.resolve(projectRoot());
  const resolved = path.isAbsolute(rawPath) ? path.resolve(rawPath) : path.resolve(root, rawPath);
  const relative = path.relative(root, resolved);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error('Path must stay inside the project workspace');
  }
  return resolved;
}

function managedDatasetsRoot() {
  return path.join(projectRoot(), 'trainer-data', 'datasets');
}

function managedDatasetStoreRoot() {
  return path.join(managedDatasetsRoot(), 'managed');
}

function combinedDatasetRoot() {
  return path.join(managedDatasetsRoot(), 'selections');
}

function managedRegistryPath() {
  return path.join(managedDatasetsRoot(), 'managed-datasets.json');
}

function toProjectRelative(filePath: string) {
  return path.relative(projectRoot(), filePath).replace(/\\/g, '/');
}

function toSlashPath(filePath: string) {
  return filePath.replace(/\\/g, '/');
}

async function exists(filePath: string) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function isImagePath(filePath: string) {
  return IMAGE_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function isCaptionPath(filePath: string) {
  return CAPTION_EXTENSIONS.has(path.extname(filePath).toLowerCase());
}

function imageUrl(filePath: string) {
  if (!isImagePath(filePath)) return undefined;
  return `/api/datasets/image?path=${encodeURIComponent(toProjectRelative(filePath))}`;
}

function attachImagePreviews(summary: DatasetSummary): DatasetSummary {
  return {
    ...summary,
    previews: summary.previews.map(preview => ({
      ...preview,
      images: preview.image_paths.map((imagePath, index) => {
        const [width, height] = preview.image_sizes[index] ?? [];
        return {
          path: imagePath,
          url: imageUrl(imagePath),
          width,
          height,
        };
      }),
    })),
  };
}

function normalizeManagedDataset(dataset: DatasetRegistryItem): DatasetRegistryItem {
  const items = Array.isArray(dataset.items) ? dataset.items : [];
  return {
    ...dataset,
    source: 'managed',
    managed: true,
    itemCount: items.length,
    items,
  };
}

function shapeForRecipe(recipeId: RecipeId) {
  return recipeId === 'flux2-klein-editing' ? 'reference + target' : 'single image';
}

function sanitizeSegment(value: string, fallback: string) {
  const sanitized = value
    .trim()
    .replace(/[^\w.-]+/g, '_')
    .replace(/^_+|_+$/g, '')
    .slice(0, 80);
  return sanitized || fallback;
}

function fileStem(fileName: string) {
  return path.basename(fileName, path.extname(fileName));
}

function itemGroupKey(stem: string) {
  return stem
    .replace(/(?:[_-])?(ref|reference|source|input|before|target|gt|output|after|edited)$/i, '')
    .replace(/^_+|_+$/g, '') || stem;
}

function itemRoleRank(stem: string) {
  if (/(?:[_-])?(target|gt|output|after|edited)$/i.test(stem)) return 1;
  if (/(?:[_-])?(ref|reference|source|input|before)$/i.test(stem)) return 0;
  return 0.5;
}

function asUploadFile(value: FormDataEntryValue): File | null {
  if (typeof value !== 'object' || value === null) return null;
  const candidate = value as File;
  return typeof candidate.arrayBuffer === 'function' && typeof candidate.name === 'string' ? candidate : null;
}

export function uploadFilesFromForm(formData: FormData, field = 'files') {
  return formData.getAll(field).map(asUploadFile).filter((file): file is File => Boolean(file));
}

export function captionFileByStem(files: File[], stem: string) {
  const lowered = stem.toLowerCase();
  return files.find(file => isCaptionPath(file.name) && fileStem(file.name).toLowerCase() === lowered) ?? null;
}

async function readCaptionFile(file: File) {
  const text = (await file.text()).trim();
  if (path.extname(file.name).toLowerCase() !== '.json') return text;
  try {
    const payload = JSON.parse(text) as Record<string, unknown>;
    return String(payload.caption ?? payload.prompt ?? payload.text ?? text).trim();
  } catch {
    return text;
  }
}

async function uniqueUploadPath(dirPath: string, fileName: string) {
  const parsed = path.parse(sanitizeSegment(fileName, `file-${Date.now()}`));
  let candidate = path.join(dirPath, `${parsed.name}${parsed.ext}`);
  let index = 1;
  while (await exists(candidate)) {
    candidate = path.join(dirPath, `${parsed.name}-${index}${parsed.ext}`);
    index += 1;
  }
  return candidate;
}

async function saveUploadFile(file: File, dirPath: string) {
  await fs.mkdir(dirPath, { recursive: true });
  const filePath = await uniqueUploadPath(dirPath, file.name);
  const buffer = Buffer.from(await file.arrayBuffer());
  await fs.writeFile(filePath, buffer);
  return filePath;
}

async function imageSizeFor(filePath: string): Promise<[number, number]> {
  const script = [
    'import json, sys',
    'from PIL import Image',
    'with Image.open(sys.argv[1]) as im:',
    '    print(json.dumps([im.size[0], im.size[1]]))',
  ].join('\n');
  try {
    const { stdout } = await execFileAsync('python', ['-c', script, filePath], {
      cwd: projectRoot(),
      timeout: 10000,
      windowsHide: true,
    });
    const value = JSON.parse(stdout) as [number, number];
    if (Number.isFinite(value[0]) && Number.isFinite(value[1])) return value;
  } catch {
    // Keep imports non-blocking if PIL cannot read a file yet; validator will still report details.
  }
  return [1024, 1024];
}

async function createManagedItem(imagePaths: string[], prompt: string, captionPath: string | undefined, now: string): Promise<ManagedDatasetItem> {
  const imageSizes = await Promise.all(imagePaths.map(imagePath => imageSizeFor(resolveProjectPath(imagePath))));
  return {
    id: crypto.randomUUID(),
    imagePaths,
    imageSizes,
    captionPath,
    prompt,
    createdAt: now,
    updatedAt: now,
  };
}

async function writeCaptionFile(captionsDir: string, nameStem: string, prompt: string) {
  await fs.mkdir(captionsDir, { recursive: true });
  const safeStem = sanitizeSegment(nameStem, 'caption');
  const captionPath = await uniqueUploadPath(captionsDir, `${safeStem}.txt`);
  await fs.writeFile(captionPath, prompt, 'utf-8');
  return toProjectRelative(captionPath);
}

function groupsForRecipe(recipeId: RecipeId, images: SavedUploadImage[]) {
  if (recipeId !== 'flux2-klein-editing') {
    return images.map(image => [image]);
  }

  const grouped = new Map<string, SavedUploadImage[]>();
  for (const image of images) {
    const items = grouped.get(image.groupKey) ?? [];
    items.push(image);
    grouped.set(image.groupKey, items);
  }
  const groups = Array.from(grouped.values())
    .map(items => items.sort((a, b) => a.roleRank - b.roleRank || a.projectPath.localeCompare(b.projectPath)));
  if (groups.some(group => group.length > 1)) return groups;

  const sorted = [...images].sort((a, b) => a.projectPath.localeCompare(b.projectPath));
  const paired: SavedUploadImage[][] = [];
  for (let index = 0; index < sorted.length; index += 2) {
    paired.push(sorted.slice(index, index + 2));
  }
  return paired;
}

async function writeManagedDatasetJsonl(dataset: DatasetRegistryItem) {
  const jsonlPath = resolveProjectPath(dataset.path);
  const datasetDir = path.dirname(jsonlPath);
  const rows = await Promise.all((dataset.items ?? []).map(async item => {
    const localPaths = item.imagePaths.map(imagePath => toSlashPath(path.relative(datasetDir, resolveProjectPath(imagePath))));
    const imageSizes = item.imageSizes?.length === item.imagePaths.length
      ? item.imageSizes
      : await Promise.all(item.imagePaths.map(imagePath => imageSizeFor(resolveProjectPath(imagePath))));
    const sizeValues = imageSizes.map(([width, height]) => `${height}*${width}`);
    const prompt = item.prompt.trim();
    return {
      class_name: dataset.name,
      local_path_list: localPaths,
      'h*w': sizeValues.length === 1 ? sizeValues[0] : sizeValues,
      short_en: prompt,
      short_zh: prompt,
      user_prompt_en: prompt,
      user_prompt_zh: prompt,
    };
  }));
  await fs.mkdir(datasetDir, { recursive: true });
  await fs.writeFile(jsonlPath, rows.map(row => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : ''), 'utf-8');
}

async function writeManagedDatasets(datasets: DatasetRegistryItem[]) {
  await fs.mkdir(managedDatasetsRoot(), { recursive: true });
  const payload: ManagedDatasetsLedger = { version: 1, datasets: datasets.map(normalizeManagedDataset) };
  await fs.writeFile(managedRegistryPath(), `${JSON.stringify(payload, null, 2)}\n`, 'utf-8');
}

export async function readManagedDatasets() {
  if (!(await exists(managedRegistryPath()))) return [];
  const raw = await fs.readFile(managedRegistryPath(), 'utf-8');
  const payload = JSON.parse(raw.replace(/^\uFEFF/, '')) as ManagedDatasetsLedger;
  return (payload.datasets ?? []).map(normalizeManagedDataset);
}

export async function readRegisteredDatasets() {
  return [...bundledDatasets, ...await readManagedDatasets()];
}

export async function findRegisteredDataset(datasetPathOrId: string | undefined, recipeId?: RecipeId) {
  const datasets = await readRegisteredDatasets();
  if (datasetPathOrId) {
    const found = datasets.find(dataset => dataset.id === datasetPathOrId || dataset.path === datasetPathOrId);
    if (found) return found;
  }
  return recipeId ? datasets.find(dataset => dataset.recipeId === recipeId) : undefined;
}

export async function importManagedDataset({
  name,
  recipeId,
  files,
}: {
  name: string;
  recipeId: RecipeId;
  files: File[];
}) {
  const imageFiles = files.filter(file => isImagePath(file.name));
  const captionFiles = files.filter(file => isCaptionPath(file.name));
  if (!imageFiles.length) {
    throw new Error('At least one image file is required');
  }

  const now = new Date().toISOString();
  const baseName = sanitizeSegment(name, 'dataset');
  const datasetId = `${baseName.toLowerCase()}-${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;
  const datasetDir = path.join(managedDatasetStoreRoot(), datasetId);
  const imagesDir = path.join(datasetDir, 'images');
  const captionsDir = path.join(datasetDir, 'captions');
  const captionTextByStem = new Map<string, string>();

  for (const file of captionFiles) {
    captionTextByStem.set(fileStem(file.name).toLowerCase(), await readCaptionFile(file));
  }

  const savedImages: SavedUploadImage[] = [];
  for (const file of imageFiles) {
    const absolutePath = await saveUploadFile(file, imagesDir);
    const stem = fileStem(file.name);
    savedImages.push({
      projectPath: toProjectRelative(absolutePath),
      absolutePath,
      stem,
      groupKey: itemGroupKey(stem).toLowerCase(),
      roleRank: itemRoleRank(stem),
    });
  }

  const items: ManagedDatasetItem[] = [];
  for (const group of groupsForRecipe(recipeId, savedImages)) {
    const captionKey = group.map(image => image.stem.toLowerCase()).find(stem => captionTextByStem.has(stem)) ?? group[0]?.groupKey;
    const prompt = captionTextByStem.get(captionKey) ?? captionTextByStem.get(group[0]?.groupKey ?? '') ?? '';
    const captionPath = await writeCaptionFile(captionsDir, group[0]?.groupKey ?? group[0]?.stem ?? 'caption', prompt);
    items.push(await createManagedItem(group.map(image => image.projectPath), prompt, captionPath, now));
  }

  const dataset = normalizeManagedDataset({
    id: datasetId,
    name: baseName,
    path: toProjectRelative(path.join(datasetDir, 'data.jsonl')),
    recipeId,
    shape: shapeForRecipe(recipeId),
    createdAt: now,
    updatedAt: now,
    items,
  });

  await writeManagedDatasetJsonl(dataset);
  const datasets = await readManagedDatasets();
  await writeManagedDatasets([dataset, ...datasets.filter(item => item.id !== dataset.id)]);
  return dataset;
}

async function updateManagedDataset(datasetId: string, updater: (dataset: DatasetRegistryItem) => Promise<DatasetRegistryItem>) {
  const datasets = await readManagedDatasets();
  const index = datasets.findIndex(dataset => dataset.id === datasetId);
  if (index < 0) throw new Error('Managed dataset not found');
  const nextDataset = normalizeManagedDataset(await updater(datasets[index]));
  nextDataset.updatedAt = new Date().toISOString();
  await writeManagedDatasetJsonl(nextDataset);
  const nextDatasets = [...datasets];
  nextDatasets[index] = nextDataset;
  await writeManagedDatasets(nextDatasets);
  return nextDataset;
}

export async function addManagedDatasetItem(datasetId: string, files: File[], prompt: string) {
  const imageFiles = files.filter(file => isImagePath(file.name));
  const captionFile = files.find(file => isCaptionPath(file.name));
  if (!imageFiles.length) throw new Error('At least one image file is required');
  const captionText = captionFile ? await readCaptionFile(captionFile) : prompt;

  return updateManagedDataset(datasetId, async dataset => {
    const datasetDir = path.dirname(resolveProjectPath(dataset.path));
    const imagesDir = path.join(datasetDir, 'images');
    const captionsDir = path.join(datasetDir, 'captions');
    const imagePaths = [];
    for (const file of imageFiles) {
      imagePaths.push(toProjectRelative(await saveUploadFile(file, imagesDir)));
    }
    const captionPath = await writeCaptionFile(captionsDir, fileStem(imageFiles[0].name), captionText);
    const now = new Date().toISOString();
    return {
      ...dataset,
      items: [await createManagedItem(imagePaths, captionText, captionPath, now), ...(dataset.items ?? [])],
    };
  });
}

export async function updateManagedDatasetItem(datasetId: string, itemId: string, prompt: string) {
  return updateManagedDataset(datasetId, async dataset => {
    const items = await Promise.all((dataset.items ?? []).map(async item => {
      if (item.id !== itemId) return item;
      if (item.captionPath) {
        await fs.writeFile(resolveProjectPath(item.captionPath), prompt, 'utf-8');
      }
      return { ...item, prompt, updatedAt: new Date().toISOString() };
    }));
    return { ...dataset, items };
  });
}

async function removeManagedFile(projectRelativePath: string | undefined) {
  if (!projectRelativePath) return;
  const filePath = resolveProjectPath(projectRelativePath);
  const root = path.resolve(managedDatasetStoreRoot());
  const relative = path.relative(root, filePath);
  if (relative.startsWith('..') || path.isAbsolute(relative)) return;
  await fs.rm(filePath, { force: true }).catch(() => undefined);
}

export async function deleteManagedDatasetItem(datasetId: string, itemId: string) {
  return updateManagedDataset(datasetId, async dataset => {
    const item = (dataset.items ?? []).find(candidate => candidate.id === itemId);
    if (item) {
      await Promise.all([...item.imagePaths, item.captionPath].map(removeManagedFile));
    }
    return { ...dataset, items: (dataset.items ?? []).filter(candidate => candidate.id !== itemId) };
  });
}

export async function deleteManagedDataset(datasetId: string) {
  const datasets = await readManagedDatasets();
  const dataset = datasets.find(item => item.id === datasetId);
  if (!dataset) throw new Error('Managed dataset not found');
  const datasetDir = path.dirname(resolveProjectPath(dataset.path));
  const root = path.resolve(managedDatasetStoreRoot());
  const relative = path.relative(root, datasetDir);
  if (relative.startsWith('..') || path.isAbsolute(relative)) {
    throw new Error('Managed dataset path is outside the managed store');
  }
  await fs.rm(datasetDir, { recursive: true, force: true });
  await writeManagedDatasets(datasets.filter(item => item.id !== datasetId));
}

export async function validateDataset(datasetPath: string, recipeId: RecipeId): Promise<DatasetSummary> {
  const root = projectRoot();
  const resolvedPath = resolveProjectPath(datasetPath);
  const { stdout, stderr } = await execFileAsync(
    'python',
    ['scripts/check_runtime.py', 'validate-dataset', resolvedPath, '--recipe-id', recipeId, '--project-root', root],
    {
      cwd: root,
      timeout: 20000,
      windowsHide: true,
    },
  );
  if (stderr.trim()) {
    // The Python validator currently writes structured results to stdout. Keep stderr available to callers as an issue.
  }
  return attachImagePreviews(JSON.parse(stdout) as DatasetSummary);
}

export async function validateBundledDatasets() {
  return validateRegisteredDatasets();
}

export async function validateRegisteredDatasets() {
  const datasets = await readRegisteredDatasets();
  return Promise.all(
    datasets.map(async dataset => ({
      ...dataset,
      summary: await validateDataset(dataset.path, dataset.recipeId),
    })),
  );
}

function extractImagePathValues(row: Record<string, unknown>) {
  for (const key of JSONL_IMAGE_KEYS) {
    const value = row[key];
    if (value == null) continue;
    return Array.isArray(value) ? value.map(String).filter(Boolean) : [String(value)].filter(Boolean);
  }
  return [];
}

function recipeRootFor(recipeId: RecipeId) {
  if (recipeId === 'z-image-turbo-vlm') return path.join(projectRoot(), 'z-image-turbo_self-distill-vlm');
  if (recipeId === 'flux2-klein-editing') return path.join(projectRoot(), 'flux2-klein-edit-self-distill-gt-ref');
  return path.join(projectRoot(), 'flux2-klein_self-distill-edit');
}

async function resolveImageForJsonl(rawPath: string, jsonlPath: string, recipeId: RecipeId) {
  const candidate = path.isAbsolute(rawPath) ? path.resolve(rawPath) : '';
  if (candidate && await exists(candidate)) return candidate;
  const bases = [path.dirname(jsonlPath), recipeRootFor(recipeId), projectRoot()];
  for (const base of bases) {
    const resolved = path.resolve(base, rawPath);
    if (await exists(resolved)) return resolved;
  }
  return path.resolve(path.dirname(jsonlPath), rawPath);
}

async function rewriteRowsForCombinedJsonl(datasetPath: string, recipeId: RecipeId, combinedDir: string) {
  const resolvedJsonl = resolveProjectPath(datasetPath);
  const raw = await fs.readFile(resolvedJsonl, 'utf-8');
  const rows = [];
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const row = JSON.parse(line) as Record<string, unknown>;
    const imagePaths = extractImagePathValues(row);
    const resolvedImages = await Promise.all(imagePaths.map(imagePath => resolveImageForJsonl(imagePath, resolvedJsonl, recipeId)));
    for (const key of JSONL_IMAGE_KEYS) {
      delete row[key];
    }
    row.local_path_list = resolvedImages.map(imagePath => toSlashPath(path.relative(combinedDir, imagePath)));
    rows.push(row);
  }
  return rows;
}

export async function combineDatasetSelections(datasetPaths: string[], recipeId: RecipeId): Promise<CombinedDatasetSelection> {
  const uniquePaths = Array.from(new Set(datasetPaths.map(item => item.trim()).filter(Boolean)));
  if (!uniquePaths.length) throw new Error('At least one dataset must be selected');
  const nowId = new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14);
  const combinedDir = path.join(combinedDatasetRoot(), `selection-${nowId}-${crypto.randomUUID().slice(0, 8)}`);
  await fs.mkdir(combinedDir, { recursive: true });

  const sourcePaths: string[] = [];
  const rows: Record<string, unknown>[] = [];
  for (const selectedPath of uniquePaths) {
    const registered = await findRegisteredDataset(selectedPath, recipeId);
    const datasetPath = registered?.path ?? selectedPath;
    const summary = await validateDataset(datasetPath, recipeId);
    if (!summary.ok) {
      throw new Error(`Dataset issues must be fixed before launch: ${datasetPath}`);
    }
    sourcePaths.push(datasetPath);
    rows.push(...await rewriteRowsForCombinedJsonl(datasetPath, recipeId, combinedDir));
  }

  const combinedPath = path.join(combinedDir, 'data.jsonl');
  await fs.writeFile(combinedPath, rows.map(row => JSON.stringify(row)).join('\n') + '\n', 'utf-8');
  const datasetPath = toProjectRelative(combinedPath);
  const summary = await validateDataset(datasetPath, recipeId);
  const datasetIssueCount = summary.issues.filter(issue => issue.severity === 'error').length;
  return {
    datasetPath,
    datasetRows: summary.rows,
    datasetValidRows: summary.valid_rows,
    datasetIssueCount,
    sourcePaths,
  };
}

export async function readProjectImage(rawPath: string) {
  const filePath = resolveProjectPath(rawPath);
  if (!isImagePath(filePath)) {
    return null;
  }
  const data = await fs.readFile(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const contentType = ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : ext === '.webp' ? 'image/webp' : ext === '.bmp' ? 'image/bmp' : 'image/png';
  return { data, contentType };
}
