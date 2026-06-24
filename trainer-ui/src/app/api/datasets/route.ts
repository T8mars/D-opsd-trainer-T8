import { NextResponse } from 'next/server';
import {
  addManagedDatasetItem,
  bundledDatasets,
  combineDatasetSelections,
  deleteManagedDataset,
  deleteManagedDatasetItem,
  importManagedDataset,
  readRegisteredDatasets,
  updateManagedDatasetItem,
  uploadFilesFromForm,
  validateDataset,
  validateRegisteredDatasets,
} from '@/lib/datasets';
import type { RecipeId } from '@/lib/recipes';

export const runtime = 'nodejs';

export async function GET() {
  const datasets = await validateRegisteredDatasets();
  return NextResponse.json({
    ok: true,
    generatedAt: new Date().toISOString(),
    datasets,
  });
}

export async function POST(request: Request) {
  const contentType = request.headers.get('content-type') || '';
  if (contentType.includes('multipart/form-data')) {
    return handleFormAction(request);
  }

  const payload = (await request.json().catch(() => ({}))) as {
    action?: string;
    path?: string;
    recipeId?: RecipeId;
    datasetId?: string;
    itemId?: string;
    prompt?: string;
    datasetPaths?: string[];
  };

  try {
    if (payload.action === 'update-item') {
      if (!payload.datasetId || !payload.itemId) {
        return NextResponse.json({ ok: false, error: 'datasetId and itemId are required' }, { status: 400 });
      }
      const dataset = await updateManagedDatasetItem(payload.datasetId, payload.itemId, payload.prompt ?? '');
      const summary = await validateDataset(dataset.path, dataset.recipeId);
      return NextResponse.json({ ok: true, generatedAt: new Date().toISOString(), dataset: { ...dataset, summary } });
    }

    if (payload.action === 'delete-item') {
      if (!payload.datasetId || !payload.itemId) {
        return NextResponse.json({ ok: false, error: 'datasetId and itemId are required' }, { status: 400 });
      }
      const dataset = await deleteManagedDatasetItem(payload.datasetId, payload.itemId);
      const summary = await validateDataset(dataset.path, dataset.recipeId);
      return NextResponse.json({ ok: true, generatedAt: new Date().toISOString(), dataset: { ...dataset, summary } });
    }

    if (payload.action === 'delete-dataset') {
      if (!payload.datasetId) {
        return NextResponse.json({ ok: false, error: 'datasetId is required' }, { status: 400 });
      }
      await deleteManagedDataset(payload.datasetId);
      return NextResponse.json({ ok: true, generatedAt: new Date().toISOString() });
    }

    if (payload.action === 'combine-selection') {
      if (!payload.recipeId || !payload.datasetPaths?.length) {
        return NextResponse.json({ ok: false, error: 'recipeId and datasetPaths are required' }, { status: 400 });
      }
      const combined = await combineDatasetSelections(payload.datasetPaths, payload.recipeId);
      const summary = await validateDataset(combined.datasetPath, payload.recipeId);
      return NextResponse.json({
        ok: true,
        generatedAt: new Date().toISOString(),
        dataset: {
          id: 'combined-selection',
          name: 'Combined dataset selection',
          path: combined.datasetPath,
          recipeId: payload.recipeId,
          shape: 'combined',
          source: 'combined',
          summary,
        },
      });
    }
  } catch (error: any) {
    return NextResponse.json({ ok: false, error: String(error?.message || error) }, { status: 400 });
  }

  const registeredDatasets = await readRegisteredDatasets();
  const dataset = registeredDatasets.find(item => item.id === payload.path || item.path === payload.path)
    ?? bundledDatasets.find(item => item.id === payload.path || item.path === payload.path);
  const datasetPath = dataset?.path ?? payload.path;
  const recipeId = dataset?.recipeId ?? payload.recipeId;

  if (!datasetPath || !recipeId) {
    return NextResponse.json({ ok: false, error: 'Dataset path and recipeId are required' }, { status: 400 });
  }

  try {
    const summary = await validateDataset(datasetPath, recipeId);
    return NextResponse.json({
      ok: true,
      generatedAt: new Date().toISOString(),
      dataset: dataset ? { ...dataset, summary } : { id: 'custom', name: datasetPath, path: datasetPath, recipeId, shape: 'custom', summary },
    });
  } catch (error: any) {
    return NextResponse.json({ ok: false, error: String(error?.message || error) }, { status: 400 });
  }
}

async function handleFormAction(request: Request) {
  const formData = await request.formData();
  const action = String(formData.get('action') || 'import-files');
  const recipeId = String(formData.get('recipeId') || '') as RecipeId;
  const name = String(formData.get('name') || formData.get('datasetName') || 'dataset');
  const files = uploadFilesFromForm(formData);

  try {
    if (action === 'import-files') {
      if (!recipeId) {
        return NextResponse.json({ ok: false, error: 'recipeId is required' }, { status: 400 });
      }
      const dataset = await importManagedDataset({ name, recipeId, files });
      const summary = await validateDataset(dataset.path, dataset.recipeId);
      return NextResponse.json({ ok: true, generatedAt: new Date().toISOString(), dataset: { ...dataset, summary } }, { status: 201 });
    }

    if (action === 'add-item') {
      const datasetId = String(formData.get('datasetId') || '');
      const prompt = String(formData.get('prompt') || '');
      if (!datasetId) {
        return NextResponse.json({ ok: false, error: 'datasetId is required' }, { status: 400 });
      }
      const dataset = await addManagedDatasetItem(datasetId, files, prompt);
      const summary = await validateDataset(dataset.path, dataset.recipeId);
      return NextResponse.json({ ok: true, generatedAt: new Date().toISOString(), dataset: { ...dataset, summary } }, { status: 201 });
    }

    return NextResponse.json({ ok: false, error: `Unsupported dataset action: ${action}` }, { status: 400 });
  } catch (error: any) {
    return NextResponse.json({ ok: false, error: String(error?.message || error) }, { status: 400 });
  }
}
