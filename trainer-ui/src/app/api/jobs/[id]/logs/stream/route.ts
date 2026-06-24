import { NextResponse } from 'next/server';
import { getJobLogs, type JobLogLine } from '@/lib/jobs';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const STREAM_POLL_MS = 1000;
const encoder = new TextEncoder();
const sseEvents = {
  snapshot: 'event: snapshot',
  append: 'event: append',
  heartbeat: 'event: heartbeat',
  error: 'event: error',
} as const;

type StreamEvent = keyof typeof sseEvents;

function formatSseEvent(event: StreamEvent, payload: unknown) {
  return encoder.encode(`${sseEvents[event]}\ndata: ${JSON.stringify(payload)}\n\n`);
}

function sameLogLine(left: JobLogLine, right: JobLogLine) {
  return left.source === right.source && left.line === right.line;
}

function findOverlap(previous: JobLogLine[], next: JobLogLine[]) {
  const maxOverlap = Math.min(previous.length, next.length);
  for (let size = maxOverlap; size > 0; size -= 1) {
    let matches = true;
    for (let index = 0; index < size; index += 1) {
      if (!sameLogLine(previous[previous.length - size + index], next[index])) {
        matches = false;
        break;
      }
    }
    if (matches) return size;
  }
  return 0;
}

function getNewLogLines(previous: JobLogLine[], next: JobLogLine[]) {
  if (!previous.length) return next;
  const overlap = findOverlap(previous, next);
  return next.slice(overlap);
}

function createLogStream(
  id: string,
  signal: AbortSignal,
  initial: { generatedAt: string; truncated: boolean; combined: JobLogLine[] },
) {
  const initialCombined = initial.combined;
  let previousCombined = initialCombined;
  let interval: ReturnType<typeof setInterval> | undefined;
  let closed = false;
  let polling = false;

  return new ReadableStream({
    start(controller) {
      const close = () => {
        if (closed) return;
        closed = true;
        if (interval) clearInterval(interval);
        signal.removeEventListener('abort', close);
        controller.close();
      };

      const send = (event: StreamEvent, payload: unknown) => {
        if (closed) return;
        controller.enqueue(formatSseEvent(event, payload));
      };

      const poll = async () => {
        if (closed || polling) return;
        polling = true;
        try {
          const result = await getJobLogs(id);
          if (!result.ok) {
            send('error', { error: result.error || 'Logs could not be loaded.' });
            close();
            return;
          }

          const combined = result.combined ?? [];
          const lines = getNewLogLines(previousCombined, combined);
          previousCombined = combined;
          if (lines.length) {
            send('append', {
              generatedAt: result.generatedAt,
              truncated: result.truncated,
              lines,
            });
          } else {
            send('heartbeat', {
              generatedAt: result.generatedAt,
              truncated: result.truncated,
            });
          }
        } catch (error) {
          send('error', { error: error instanceof Error ? error.message : 'Log stream failed.' });
        } finally {
          polling = false;
        }
      };

      send('snapshot', {
        generatedAt: initial.generatedAt,
        truncated: initial.truncated,
        combined: initialCombined,
      });
      interval = setInterval(() => {
        void poll();
      }, STREAM_POLL_MS);
      signal.addEventListener('abort', close);
    },
    cancel() {
      closed = true;
      if (interval) clearInterval(interval);
    },
  });
}

export async function GET(request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const initial = await getJobLogs(id);
  if (!initial.ok) {
    return NextResponse.json(initial, { status: initial.status });
  }

  const stream = createLogStream(id, request.signal, {
    generatedAt: initial.generatedAt,
    truncated: initial.truncated,
    combined: initial.combined ?? [],
  });
  return new Response(stream, {
    headers: {
      'content-type': 'text/event-stream; charset=utf-8',
      'cache-control': 'no-cache, no-transform',
      connection: 'keep-alive',
      'x-accel-buffering': 'no',
    },
  });
}
