import { NextResponse } from 'next/server';
import { createDraftJob, createRunnerProbeJob, readJobs, type TrainingOverrides } from '@/lib/jobs';
import type { RecipeId } from '@/lib/recipes';

export const runtime = 'nodejs';

export async function GET() {
  const jobs = await readJobs();
  return NextResponse.json({
    ok: true,
    generatedAt: new Date().toISOString(),
    jobs,
  });
}

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => ({}))) as {
    probe?: boolean;
    probeDurationSeconds?: number;
    recipeId?: RecipeId;
    datasetPath?: string;
    datasetPaths?: string[];
    trainingOverrides?: TrainingOverrides;
  };
  try {
    const job = payload.probe
      ? await createRunnerProbeJob(payload.probeDurationSeconds)
      : await createDraftJob(payload.recipeId, payload.datasetPath, payload.datasetPaths, payload.trainingOverrides);
    return NextResponse.json({ ok: true, job }, { status: 201 });
  } catch (error: any) {
    return NextResponse.json({ ok: false, error: String(error?.message || error) }, { status: 400 });
  }
}
