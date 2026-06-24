'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import type { ComponentType, ReactNode } from 'react';
import {
  Activity,
  Boxes,
  Database,
  Gauge,
  HardDriveDownload,
  Languages,
  LineChart,
  ListChecks,
  Plus,
  Settings,
} from 'lucide-react';
import { useI18n, type MessageKey } from '@/lib/i18n';

const navItems: Array<{ href: string; labelKey: MessageKey; icon: ComponentType<{ className?: string }> }> = [
  { href: '/', labelKey: 'navDashboard', icon: Gauge },
  { href: '/jobs/new', labelKey: 'navNewJob', icon: Plus },
  { href: '/jobs', labelKey: 'navJobs', icon: Activity },
  { href: '/tensorboard', labelKey: 'navTensorBoard', icon: LineChart },
  { href: '/datasets', labelKey: 'navDatasets', icon: Database },
  { href: '/models', labelKey: 'navModels', icon: HardDriveDownload },
  { href: '/settings', labelKey: 'navSettings', icon: Settings },
];

export default function AppShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const { language, setLanguage, t } = useI18n();

  return (
    <div className="min-h-screen text-ink-100">
      <div className="fixed inset-y-0 left-0 hidden w-[17rem] p-3 lg:block">
        <aside className="glass flex h-full flex-col rounded-lg">
          <div className="border-b border-white/10 px-4 py-4">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-white/20 bg-white/10 shadow-insetGlass">
                <Boxes className="h-5 w-5 text-aqua-300" />
              </div>
              <div>
                <div className="text-sm font-semibold tracking-wide">T8 D-OPSD Tranier</div>
                <div className="text-xs text-ink-400">{t('brandSubtitle')}</div>
              </div>
            </div>
          </div>
          <nav className="flex-1 space-y-1 px-2 py-4">
            {navItems.map(item => {
              const active = pathname === item.href;
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center gap-3 rounded-md px-3 py-2 text-sm transition ${
                    active
                      ? 'bg-white/[0.16] text-white shadow-insetGlass'
                      : 'text-ink-300 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{t(item.labelKey)}</span>
                </Link>
              );
            })}
          </nav>
          <div className="border-t border-white/10 p-3">
            <div className="rounded-md bg-black/20 p-3 text-xs text-ink-300">
              <div className="mb-1 text-ink-100">{t('defaultProfile')}</div>
              <div>RTX 4060 Ti 16GB</div>
              <div>{t('wslTrainingBackend')}</div>
            </div>
          </div>
        </aside>
      </div>

      <div className="lg:pl-[17rem]">
        <header className="sticky top-0 z-30 p-3">
          <div className="glass flex min-h-14 items-center justify-between rounded-lg px-3">
            <div className="flex items-center gap-3 overflow-x-auto no-scrollbar lg:hidden">
              {navItems.map(item => {
                const active = pathname === item.href;
                const Icon = item.icon;
                return (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex h-10 w-10 flex-none items-center justify-center rounded-md transition ${
                      active ? 'bg-white/[0.18] text-white' : 'text-ink-300 hover:bg-white/10'
                    }`}
                    aria-label={t(item.labelKey)}
                    title={t(item.labelKey)}
                  >
                    <Icon className="h-4 w-4" />
                  </Link>
                );
              })}
            </div>
            <div className="hidden items-center gap-2 text-sm text-ink-300 lg:flex">
              <ListChecks className="h-4 w-4 text-mint-500" />
              <span>{t('contextAwareLocalTrainer')}</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                aria-label={language === 'zh' ? t('switchToEnglish') : t('switchToChinese')}
                title={language === 'zh' ? t('switchToEnglish') : t('switchToChinese')}
                onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')}
                className="inline-flex items-center gap-1.5 rounded-md border border-white/15 bg-white/[0.08] px-2 py-1 text-xs text-ink-200 shadow-insetGlass transition hover:bg-white/[0.13] hover:text-white"
              >
                <Languages className="h-3.5 w-3.5 text-aqua-300" />
                <span className={language === 'zh' ? 'text-white' : 'text-ink-400'}>中文</span>
                <span className="text-ink-600">/</span>
                <span className={language === 'en' ? 'text-white' : 'text-ink-400'}>EN</span>
              </button>
              <span className="hidden rounded-md border border-mint-500/25 bg-mint-500/[0.12] px-2 py-1 text-xs text-mint-500 sm:inline-flex">
                {t('singleGpuFirst')}
              </span>
              <span className="hidden rounded-md border border-aqua-500/25 bg-aqua-500/[0.12] px-2 py-1 text-xs text-aqua-300 sm:inline-flex">
                {t('liquidGlass')}
              </span>
            </div>
          </div>
        </header>
        <main className="px-3 pb-8 lg:px-6">{children}</main>
      </div>
    </div>
  );
}
