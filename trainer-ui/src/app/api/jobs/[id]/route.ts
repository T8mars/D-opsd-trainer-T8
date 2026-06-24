import { NextResponse } from 'next/server';
import { deleteJob } from '@/lib/jobs';

export const runtime = 'nodejs';

export async function DELETE(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const result = await deleteJob(id);
  return NextResponse.json(result, { status: result.status });
}
