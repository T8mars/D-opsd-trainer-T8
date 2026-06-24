'use client';

import { useEffect, useMemo, useState } from 'react';
import { Cpu, HardDrive, KeyRound, MonitorCog, Server, Terminal } from 'lucide-react';
import StatusPill from '@/components/StatusPill';
import { useI18n } from '@/lib/i18n';

type SystemPayload = {
  ok: boolean;
  generatedAt: string;
  projectRoot: string;
  gpu: {
    available: boolean;
    gpus: Array<{
      index: number;
      name: string;
      driver: string;
      memoryTotalMb: number;
      memoryFreeMb: number;
    }>;
    error?: string;
  };
  python: { available: boolean; version?: string; error?: string };
  node: { version: string };
  conda: { available: boolean; paths: string[] };
  wsl: { available: boolean; distributions: string[]; error?: string };
  hfToken: { present: boolean };
  disk: { available: boolean; freeGb?: number; sizeGb?: number; error?: string };
};

const loadingPayload: SystemPayload = {
  ok: false,
  generatedAt: '',
  projectRoot: '',
  gpu: { available: false, gpus: [] },
  python: { available: false },
  node: { version: '' },
  conda: { available: false, paths: [] },
  wsl: { available: false, distributions: [] },
  hfToken: { present: false },
  disk: { available: false },
};

function formatGb(value?: number) {
  if (value == null) return '-';
  return `${value.toFixed(1)} GB`;
}

export default function SystemOverview() {
  const [data, setData] = useState<SystemPayload>(loadingPayload);
  const [loading, setLoading] = useState(true);
  const { t } = useI18n();

  useEffect(() => {
    let mounted = true;
    fetch('/api/system')
      .then(res => res.json())
      .then(payload => {
        if (mounted) setData(payload);
      })
      .catch(() => {
        if (mounted) setData(loadingPayload);
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const gpuLabel = useMemo(() => {
    if (loading) return t('checking');
    if (!data.gpu.available) return t('notFound');
    return `${data.gpu.gpus.length} GPU${data.gpu.gpus.length === 1 ? '' : 's'}`;
  }, [data.gpu.available, data.gpu.gpus.length, loading, t]);

  const cards = [
    {
      title: 'GPU',
      icon: Cpu,
      value: gpuLabel,
      detail: data.gpu.gpus[0]
        ? `${data.gpu.gpus[0].name} · ${Math.round(data.gpu.gpus[0].memoryFreeMb / 1024)} GB ${t('free')}`
        : data.gpu.error || t('gpuDetailMissing'),
      tone: data.gpu.available ? 'good' : 'bad',
    },
    {
      title: 'Python',
      icon: Terminal,
      value: data.python.version || (loading ? t('checking') : t('missing')),
      detail: data.python.available ? t('pythonReady') : data.python.error || t('pythonUnavailable'),
      tone: data.python.available ? 'good' : 'bad',
    },
    {
      title: 'WSL',
      icon: Server,
      value: data.wsl.available ? t('available') : loading ? t('checking') : t('unavailable'),
      detail: data.wsl.distributions.length ? data.wsl.distributions.join(' · ') : data.wsl.error || t('preferredForTraining'),
      tone: data.wsl.available ? 'good' : 'warn',
    },
    {
      title: t('disk'),
      icon: HardDrive,
      value: data.disk.available ? formatGb(data.disk.freeGb) : loading ? t('checking') : t('unknown'),
      detail: data.disk.available ? `${formatGb(data.disk.sizeGb)} ${t('totalOnProjectDrive')}` : data.disk.error || t('diskProbePending'),
      tone: data.disk.available && (data.disk.freeGb ?? 0) > 80 ? 'good' : 'warn',
    },
    {
      title: 'Conda',
      icon: MonitorCog,
      value: data.conda.available ? t('available') : loading ? t('checking') : t('missing'),
      detail: data.conda.paths[0] || t('condaPurpose'),
      tone: data.conda.available ? 'good' : 'warn',
    },
    {
      title: 'HF Token',
      icon: KeyRound,
      value: data.hfToken.present ? t('configured') : t('notSet'),
      detail: data.hfToken.present ? t('hfTokenConfigured') : t('hfTokenMissing'),
      tone: data.hfToken.present ? 'good' : 'warn',
    },
  ] as const;

  return (
    <div className="metric-grid">
      {cards.map(card => {
        const Icon = card.icon;
        return (
          <section key={card.title} className="solid-panel rounded-lg p-4">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm text-ink-300">
                <Icon className="h-4 w-4 text-aqua-300" />
                {card.title}
              </div>
              <StatusPill label={card.tone === 'good' ? t('ok') : card.tone === 'bad' ? t('fix') : t('check')} tone={card.tone} />
            </div>
            <div className="text-xl font-semibold text-white">{card.value}</div>
            <p className="mt-2 min-h-10 text-sm leading-5 text-ink-400">{card.detail}</p>
          </section>
        );
      })}
    </div>
  );
}
