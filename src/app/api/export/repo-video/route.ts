import { request as httpRequest, type IncomingHttpHeaders, type IncomingMessage } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { Readable } from 'node:stream';
import { NextRequest, NextResponse } from 'next/server';

const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';
const VIDEO_PROXY_TIMEOUT_MS = 15 * 60 * 1000;

// Video generation is slow (narration + TTS + rendering + ffmpeg composition).
// Allow up to 15 minutes for the backend to respond.
export const maxDuration = 900;
export const runtime = 'nodejs';

type ProxiedResponse = {
  body: IncomingMessage;
  headers: IncomingHttpHeaders;
  statusCode: number;
  statusMessage: string;
};

async function readNodeStreamAsText(stream: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of stream) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString('utf-8');
}

function getSingleHeaderValue(value: string | string[] | undefined): string | undefined {
  if (Array.isArray(value)) {
    return value.join(', ');
  }
  return value;
}

function proxyVideoExportRequest(targetUrl: string, requestBody: string): Promise<ProxiedResponse> {
  const url = new URL(targetUrl);
  const requestImpl = url.protocol === 'https:' ? httpsRequest : httpRequest;

  return new Promise((resolve, reject) => {
    const proxyRequest = requestImpl(
      url,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(requestBody).toString(),
        },
      },
      (response) => {
        resolve({
          body: response,
          headers: response.headers,
          statusCode: response.statusCode ?? 500,
          statusMessage: response.statusMessage ?? 'Unknown Status',
        });
      },
    );

    proxyRequest.setTimeout(VIDEO_PROXY_TIMEOUT_MS, () => {
      proxyRequest.destroy(
        new Error(`Proxy request timed out after ${VIDEO_PROXY_TIMEOUT_MS}ms while waiting for backend response`),
      );
    });
    proxyRequest.on('error', reject);
    proxyRequest.write(requestBody);
    proxyRequest.end();
  });
}

function formatCause(cause: unknown): string | undefined {
  if (!cause) {
    return undefined;
  }

  if (cause instanceof Error) {
    return `${cause.name}: ${cause.message}`;
  }

  if (typeof cause === 'string') {
    return cause;
  }

  if (typeof cause === 'object') {
    const code = 'code' in cause ? String((cause as { code?: unknown }).code ?? '') : '';
    const message = 'message' in cause ? String((cause as { message?: unknown }).message ?? '') : '';
    return [code, message].filter(Boolean).join(': ') || JSON.stringify(cause);
  }

  return String(cause);
}

function buildProxyError(error: unknown, targetUrl: string) {
  if (!(error instanceof Error)) {
    return {
      status: 500,
      body: { error: 'Unknown proxy error', targetUrl },
    };
  }

  const cause = formatCause((error as Error & { cause?: unknown }).cause);
  const searchableText = `${error.name} ${error.message} ${cause ?? ''}`.toLowerCase();
  const isTimeout = searchableText.includes('timeout') || searchableText.includes('timed out');

  return {
    status: isTimeout ? 504 : 502,
    body: {
      error: isTimeout ? 'Backend video export timed out' : 'Backend video export request failed',
      details: `${error.name}: ${error.message}${cause ? ` | cause: ${cause}` : ''}`,
      targetUrl,
    },
  };
}

/**
 * Proxy direct Video export requests to the backend.
 */
export async function POST(req: NextRequest) {
  const targetUrl = `${TARGET_SERVER_BASE_URL}/export/repo/video`;

  try {
    const requestBody = JSON.stringify(await req.json());
    const backendResponse = await proxyVideoExportRequest(targetUrl, requestBody);

    if (backendResponse.statusCode < 200 || backendResponse.statusCode >= 300) {
      const errorBody = await readNodeStreamAsText(backendResponse.body);
      return new NextResponse(errorBody, {
        status: backendResponse.statusCode,
        statusText: backendResponse.statusMessage,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    const responseHeaders = new Headers();
    responseHeaders.set('Content-Type', getSingleHeaderValue(backendResponse.headers['content-type']) || 'video/mp4');

    const contentDisposition = getSingleHeaderValue(backendResponse.headers['content-disposition']);
    if (contentDisposition) {
      responseHeaders.set('Content-Disposition', contentDisposition);
    }

    const contentLength = getSingleHeaderValue(backendResponse.headers['content-length']);
    if (contentLength) {
      responseHeaders.set('Content-Length', contentLength);
    }

    return new NextResponse(Readable.toWeb(backendResponse.body) as ReadableStream<Uint8Array>, {
      status: 200,
      headers: responseHeaders,
    });

  } catch (error) {
    console.error('Error in direct Video export proxy route:', {
      targetUrl,
      error,
    });
    const proxyError = buildProxyError(error, targetUrl);
    return new NextResponse(JSON.stringify(proxyError.body), {
      status: proxyError.status,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
