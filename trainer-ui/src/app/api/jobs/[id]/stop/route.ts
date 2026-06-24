import { NextResponse } from 'next/server';
import { stopJob } from '@/lib/jobs';

export const runtime = 'nodejs';

export async function POST(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const result = await stopJob(id);
  return NextResponse.json(result, { status: result.status });
}
