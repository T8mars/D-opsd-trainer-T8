import { NextResponse } from 'next/server';
import { resolveProjectRoot } from '@/lib/project';
import { probeSystem } from '@/lib/system';

export const runtime = 'nodejs';

export async function GET() {
  const projectRoot = resolveProjectRoot();
  return NextResponse.json(await probeSystem(projectRoot));
}
