'use client';

import ModelManager from '@/components/ModelManager';
import { useI18n } from '@/lib/i18n';

export default function ModelsPage() {
  const { t } = useI18n();

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-white">{t('modelsPageTitle')}</h1>
        <p className="mt-1 text-sm text-ink-400">{t('modelsPageSubtitle')}</p>
      </div>

      <ModelManager />
    </div>
  );
}
