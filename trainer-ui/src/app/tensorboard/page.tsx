'use client';

import TensorBoardConsole from '@/components/TensorBoardConsole';
import { useI18n } from '@/lib/i18n';

export default function TensorBoardPage() {
  const { t } = useI18n();

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-white">{t('tensorboardPageTitle')}</h1>
        <p className="mt-1 text-sm text-ink-400">{t('tensorboardPageSubtitle')}</p>
      </div>

      <TensorBoardConsole />
    </div>
  );
}

