'use client';

import DatasetValidator from '@/components/DatasetValidator';
import { useI18n } from '@/lib/i18n';

export default function DatasetsPage() {
  const { t } = useI18n();

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-white">{t('datasetsPageTitle')}</h1>
        <p className="mt-1 text-sm text-ink-400">{t('datasetsPageSubtitle')}</p>
      </div>

      <DatasetValidator />
    </div>
  );
}
