import { NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import { resolveProjectRoot } from '@/lib/project';

export const runtime = 'nodejs';

async function readJson(filePath: string) {
  try {
    const raw = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export async function GET() {
  const projectRoot = resolveProjectRoot();
  const [meta, features] = await Promise.all([
    readJson(path.join(projectRoot, 'meta.json')),
    readJson(path.join(projectRoot, 'features.json')),
  ]);

  return NextResponse.json({
    projectRoot,
    meta,
    features,
  });
}
