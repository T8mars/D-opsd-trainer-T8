'use client';

import { useEffect, useState } from 'react';
import { ArrowRight, Clock3 } from 'lucide-react';
import Link from 'next/link';
import StatusPill from '@/components/StatusPill';
import { useI18n, type MessageKey } from '@/lib/i18n';
import type { JobSummary, JobStatus } from '@/lib/jobs';

const statusTone: Record<JobStatus, 'good' | 'warn' | 'bad' | 'neutral'> = {
  draft: 'neutral',
  queued: 'warn',
  running: 'warn',
  completed: 'good',
  failed: 'bad',
  stopped: 'neutral',
};

const statusLabelKeys: Record<JobStatus, MessageKey> = {
  draft: 'draft',
  queued: 'queued',
  running: 'running',
  completed: 'completed',
  failed: 'failed',
  stopped: 'stopped',
};

type Payload = {
  jobs: JobSummary[];
};

export default function JobQueuePreview() {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const { t } = useI18n();

  useEffect(() => {
    let mounted = true;
    fetch('/api/jobs', { cache: 'no-store' })
      .then(res => res.json())
      .then((payload: Payload) => {
        if (mounted) setJobs(payload.jobs?.slice(0, 3) ?? []);
      })
      .catch(() => {
        if (mounted) setJobs([]);
      });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="solid-panel min-w-0 rounded-lg p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-medium text-white">
          <Clock3 className="h-4 w-4 text-aqua-300" />
          {t('queuePreview')}
        </div>
        <Link href="/jobs" className="inline-flex items-center gap-1 text-xs text-aqua-300">
          {t('navJobs')}
          <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>

      <div className="space-y-2">
        {jobs.length === 0 ? (
          <div className="rounded-md border border-white/10 bg-black/20 p-3 text-sm text-ink-400">{t('noJobsTracked')}</div>
        ) : (
          jobs.map(job => (
            <div key={job.id} className="rounded-md border border-white/10 bg-black/20 p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                  <div className="truncate text-sm text-white">{job.name}</div>
                  <div className="text-xs text-ink-400">{job.recipeName}</div>
                </div>
                <StatusPill label={t(statusLabelKeys[job.status])} tone={statusTone[job.status]} />
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-ink-400">
                <div>{job.latestStep} / {job.maxTrainSteps} {t('steps')}</div>
                <div className="text-right">{t('loss')} {job.latestLoss == null ? '-' : job.latestLoss.toFixed(4)}</div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
