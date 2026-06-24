import { NextResponse } from 'next/server';
import {
  downloadModel,
  inspectModels,
  modelCacheRoot,
  openModelFolder,
  readCustomModelPaths,
  saveCustomModelPath,
} from '@/lib/models';

export const runtime = 'nodejs';

type ModelsPostPayload = {
  action?: 'download' | 'save-custom-path' | 'open-folder';
  modelId?: string;
  path?: string;
};

export async function GET() {
  try {
    const [models, customPaths] = await Promise.all([inspectModels(), readCustomModelPaths()]);
    return NextResponse.json({
      ok: true,
      generatedAt: new Date().toISOString(),
      cacheRoot: modelCacheRoot(),
      models,
      customPaths,
    });
  } catch (error: any) {
    return NextResponse.json({
      ok: false,
      generatedAt: new Date().toISOString(),
      cacheRoot: modelCacheRoot(),
      models: [],
      customPaths: [],
      error: String(error?.message || error),
    }, { status: 500 });
  }
}

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as ModelsPostPayload;
  if (!payload.modelId) {
    return NextResponse.json({ ok: false, error: 'modelId is required' }, { status: 400 });
  }

  try {
    if (payload.action === 'save-custom-path') {
      const customPath = await saveCustomModelPath(payload.modelId, payload.path || '');
      const [models, customPaths] = await Promise.all([inspectModels(), readCustomModelPaths()]);
      return NextResponse.json({
        ok: true,
        generatedAt: new Date().toISOString(),
        cacheRoot: modelCacheRoot(),
        customPath,
        customPaths,
        models,
      });
    }

    if (payload.action === 'open-folder') {
      const result = await openModelFolder(payload.modelId, payload.path);
      return NextResponse.json({
        ok: result.ok,
        generatedAt: new Date().toISOString(),
        result,
      }, { status: result.status });
    }

    const result = await downloadModel(payload.modelId);
    const [models, customPaths] = await Promise.all([inspectModels(), readCustomModelPaths()]);
    return NextResponse.json({
      ok: result.ok,
      generatedAt: new Date().toISOString(),
      result,
      models,
      customPaths,
    }, { status: result.ok ? 200 : 400 });
  } catch (error: any) {
    return NextResponse.json({ ok: false, error: String(error?.message || error) }, { status: 500 });
  }
}
