import { NextResponse } from 'next/server';
import { cloneJob } from '@/lib/jobs';

export const runtime = 'nodejs';

export async function POST(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const job = await cloneJob(id);
  return NextResponse.json({ ok: Boolean(job), job }, { status: job ? 201 : 404 });
}
