import { NextResponse } from 'next/server';
import { readProjectImage } from '@/lib/datasets';

export const runtime = 'nodejs';

export async function GET(request: Request) {
  const url = new URL(request.url);
  const imagePath = url.searchParams.get('path');
  if (!imagePath) {
    return NextResponse.json({ ok: false, error: 'path is required' }, { status: 400 });
  }

  try {
    const image = await readProjectImage(imagePath);
    if (!image) {
      return NextResponse.json({ ok: false, error: 'image not found or unsupported' }, { status: 404 });
    }
    return new NextResponse(image.data, {
      headers: {
        'content-type': image.contentType,
        'cache-control': 'no-store',
      },
    });
  } catch (error: any) {
    return NextResponse.json({ ok: false, error: String(error?.message || error) }, { status: 400 });
  }
}
