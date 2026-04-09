import { NextRequest, NextResponse } from 'next/server';

const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || 'http://localhost:8001';

/**
 * Proxy video export progress polling to the backend.
 */
export async function GET(
  _req: NextRequest,
  { params }: { params: Promise<{ jobId: string }> },
) {
  try {
    const { jobId } = await params;
    const res = await fetch(`${TARGET_SERVER_BASE_URL}/export/progress/${jobId}`);
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json({ step: 0, total: 5, message: 'Waiting...', done: false });
  }
}