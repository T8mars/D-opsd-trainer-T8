'use client';

import GpuTelemetryPanel from '@/components/GpuTelemetryPanel';
import JobsTable from '@/components/JobsTable';
import { useI18n } from '@/lib/i18n';

export default function JobsPage() {
  const { t } = useI18n();

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-white">{t('jobsPageTitle')}</h1>
        <p className="mt-1 text-sm text-ink-400">{t('jobsPageSubtitle')}</p>
      </div>

      <GpuTelemetryPanel />
      <JobsTable />
    </div>
  );
}
