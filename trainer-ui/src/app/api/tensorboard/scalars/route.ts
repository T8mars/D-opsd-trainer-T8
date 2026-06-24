import { NextResponse } from 'next/server';
import { readScalarSeries, tensorboardMetricKeys, type TensorboardMetricKey } from '@/lib/tensorboard';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

function splitParam(value: string | null) {
  return (value || '')
    .split(',')
    .map(item => item.trim())
    .filter(Boolean);
}

function parseMetricKeys(value: string | null) {
  const allowed = new Set<string>(tensorboardMetricKeys);
  return splitParam(value).filter((item): item is TensorboardMetricKey => allowed.has(item));
}

function parseNumber(value: string | null) {
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

export async function GET(request: Request) {
  const url = new URL(request.url);
  try {
    const payload = await readScalarSeries({
      runIds: splitParam(url.searchParams.get('runIds')),
      metrics: parseMetricKeys(url.searchParams.get('metrics')),
      limit: parseNumber(url.searchParams.get('limit')),
      stride: parseNumber(url.searchParams.get('stride')),
      smooth: parseNumber(url.searchParams.get('smooth')),
    });

    return NextResponse.json({
      ok: true,
      generatedAt: new Date().toISOString(),
      ...payload,
    });
  } catch (error: any) {
    return NextResponse.json(
      { ok: false, error: String(error?.message || error) },
      { status: 500 },
    );
  }
}

