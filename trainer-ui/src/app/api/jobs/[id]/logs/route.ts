import { NextResponse } from 'next/server';
import { getJobLogs } from '@/lib/jobs';

export const runtime = 'nodejs';

export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const result = await getJobLogs(id);
  return NextResponse.json(result, { status: result.status });
}
