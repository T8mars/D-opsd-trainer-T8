import { NextResponse } from 'next/server';
import { probeGpu } from '@/lib/system';

export const runtime = 'nodejs';

export async function GET() {
  return NextResponse.json({
    ok: true,
    generatedAt: new Date().toISOString(),
    gpu: await probeGpu(),
  });
}
