'use client';

import { ArrowRight, Database, Gauge, Images, Play, Sparkles } from 'lucide-react';
import Link from 'next/link';
import GpuTelemetryPanel from '@/components/GpuTelemetryPanel';
import JobQueuePreview from '@/components/JobQueuePreview';
import SystemOverview from '@/components/SystemOverview';
import StatusPill from '@/components/StatusPill';
import { useI18n, type MessageKey } from '@/lib/i18n';
import { recipes } from '@/lib/recipes';

export default function DashboardPage() {
  const { t } = useI18n();
  const checkpointKeys: MessageKey[] = [
    'checkpointDatasetValidator',
    'checkpointCommandBuilder',
    'checkpointModelDownloader',
    'checkpointJobRunner',
    'checkpointLossParser',
    'checkpointSampleViewer',
  ];

  return (
    <div className="space-y-6">
      <section className="grid gap-4 xl:grid-cols-[1.25fr_0.75fr]">
        <div className="solid-panel rounded-lg p-5">
          <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight text-white">{t('dashboardTitle')}</h1>
              <p className="mt-1 text-sm text-ink-400">{t('dashboardSubtitle')}</p>
            </div>
            <Link
              href="/jobs/new"
              className="inline-flex items-center gap-2 rounded-md border border-aqua-500/30 bg-aqua-500/[0.14] px-3 py-2 text-sm text-aqua-300 transition hover:bg-aqua-500/20"
            >
              <Play className="h-4 w-4" />
              {t('newTraining')}
            </Link>
          </div>
          <SystemOverview />
        </div>

        <aside className="space-y-4">
          <GpuTelemetryPanel compact />

          <div className="glass rounded-lg p-5">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-aqua-300" />
                <h2 className="text-sm font-semibold text-white">{t('stablePath')}</h2>
              </div>
              <StatusPill label={t('sixteenGbProfile')} tone="good" />
            </div>
            <div className="space-y-3 text-sm text-ink-300">
              <p>{t('stablePathCopy1')}</p>
              <p>{t('stablePathCopy2')}</p>
            </div>
            <Link href="/models" className="mt-5 inline-flex items-center gap-2 text-sm text-aqua-300">
              {t('checkModelCache')}
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </aside>
      </section>

      <section className="grid gap-4 xl:grid-cols-3">
        {recipes.map(recipe => (
          <article key={recipe.id} className="solid-panel rounded-lg p-4">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <h2 className="font-medium text-white">{recipe.shortName}</h2>
                <div className="mt-1 text-xs text-ink-400">{recipe.corePath}</div>
              </div>
              <StatusPill label={recipe.status === 'ready' ? t('ready') : t('advanced')} tone={recipe.status === 'ready' ? 'good' : 'warn'} />
            </div>
            <div className="space-y-2 text-sm text-ink-300">
              <div className="flex items-center gap-2">
                <Gauge className="h-4 w-4 text-aqua-300" />
                {recipe.productionProfile.maxTrainSteps} {t('verifiedSteps')} · {t('scale')} {recipe.productionProfile.resolutionScale}
              </div>
              <div className="flex items-center gap-2">
                <Database className="h-4 w-4 text-mint-500" />
                {recipe.datasetShape}
              </div>
              <div className="flex items-center gap-2">
                <Images className="h-4 w-4 text-amberSoft-500" />
                {recipe.productionProfile.saveSamples ? t('samples') : t('noSamples')} / {recipe.productionProfile.saveCheckpoints ? t('checkpoints') : t('noCheckpoints')}
              </div>
            </div>
          </article>
        ))}
      </section>

      <section className="grid min-w-0 gap-4 lg:grid-cols-[0.8fr_1.2fr]">
        <JobQueuePreview />

        <div className="solid-panel min-w-0 rounded-lg p-4">
          <div className="mb-3 text-sm font-medium text-white">{t('nextImplementationCheckpoints')}</div>
          <div className="grid gap-2 sm:grid-cols-2">
            {checkpointKeys.map(item => (
              <div key={item} className="rounded-md border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-ink-300">
                {t(item)}
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
