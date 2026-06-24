import { promises as fs } from 'fs';
import { NextResponse } from 'next/server';
import { getJobArtifactFile } from '@/lib/jobs';

export const runtime = 'nodejs';

export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const { searchParams } = new URL(request.url);
  const artifactPath = searchParams.get('path') || '';
  const result = await getJobArtifactFile(id, artifactPath);

  if (!result.ok) {
    return NextResponse.json(result, { status: result.status });
  }

  const bytes = await fs.readFile(result.filePath);
  return new NextResponse(new Uint8Array(bytes), {
    status: 200,
    headers: {
      'content-type': result.contentType,
      'content-length': String(result.sizeBytes),
      'content-disposition': `inline; filename="${result.name.replace(/"/g, '')}"`,
      'cache-control': 'no-store',
    },
  });
}
