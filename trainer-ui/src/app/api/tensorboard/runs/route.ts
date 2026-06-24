import { NextResponse } from 'next/server';
import { discoverTensorboardRuns } from '@/lib/tensorboard';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    const runs = await discoverTensorboardRuns();
    return NextResponse.json({
      ok: true,
      generatedAt: new Date().toISOString(),
      runs,
    });
  } catch (error: any) {
    return NextResponse.json(
      { ok: false, error: String(error?.message || error) },
      { status: 500 },
    );
  }
}

