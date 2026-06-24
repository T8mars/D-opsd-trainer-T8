import { NextResponse } from 'next/server';
import { inspectModels } from '@/lib/models';
import { readSettingsSummary, projectRoot } from '@/lib/settings';
import { probeSystem } from '@/lib/system';

export const runtime = 'nodejs';

export async function GET() {
  try {
    const root = projectRoot();
    const [settings, system, models] = await Promise.all([
      readSettingsSummary(),
      probeSystem(root),
      inspectModels(),
    ]);
    return NextResponse.json({
      ok: true,
      generatedAt: new Date().toISOString(),
      settings,
      system,
      models,
    });
  } catch (error: any) {
    return NextResponse.json({
      ok: false,
      generatedAt: new Date().toISOString(),
      settings: null,
      system: null,
      models: [],
      error: String(error?.message || error),
    }, { status: 500 });
  }
}
