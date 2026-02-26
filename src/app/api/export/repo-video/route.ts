import { NextRequest, NextResponse } from 'next/server';

const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';

/**
 * Proxy direct Video export requests (from repo embeddings, no wiki needed)
 * to the backend.
 */
export async function POST(req: NextRequest) {
  try {
    const requestBody = await req.json();
    const targetUrl = `${TARGET_SERVER_BASE_URL}/export/repo/video`;

    const backendResponse = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!backendResponse.ok) {
      const errorBody = await backendResponse.text();
      return new NextResponse(errorBody, {
        status: backendResponse.status,
        statusText: backendResponse.statusText,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const videoBuffer = await backendResponse.arrayBuffer();

    const responseHeaders = new Headers();
    responseHeaders.set('Content-Type', 'video/mp4');
    responseHeaders.set('Content-Length', videoBuffer.byteLength.toString());

    const contentDisposition = backendResponse.headers.get('Content-Disposition');
    if (contentDisposition) {
      responseHeaders.set('Content-Disposition', contentDisposition);
    }

    return new NextResponse(videoBuffer, {
      status: 200,
      headers: responseHeaders,
    });

  } catch (error) {
    console.error('Error in direct Video export proxy route:', error);
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return new NextResponse(JSON.stringify({ error: errorMessage }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
