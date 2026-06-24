'use client';

import SettingsConsole from '@/components/SettingsConsole';
import { useI18n } from '@/lib/i18n';

export default function SettingsPage() {
  const { t } = useI18n();

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-white">{t('settingsPageTitle')}</h1>
        <p className="mt-1 text-sm text-ink-400">{t('settingsPageSubtitle')}</p>
      </div>

      <SettingsConsole />
    </div>
  );
}
