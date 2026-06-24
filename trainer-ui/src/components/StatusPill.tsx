type StatusTone = 'good' | 'warn' | 'bad' | 'neutral';

const toneClass: Record<StatusTone, string> = {
  good: 'border-mint-500/30 bg-mint-500/[0.12] text-mint-500',
  warn: 'border-amberSoft-500/30 bg-amberSoft-500/[0.12] text-amberSoft-500',
  bad: 'border-roseSoft-500/30 bg-roseSoft-500/[0.12] text-roseSoft-500',
  neutral: 'border-white/15 bg-white/[0.08] text-ink-300',
};

export default function StatusPill({ label, tone = 'neutral' }: { label: string; tone?: StatusTone }) {
  return <span className={`rounded-md border px-2 py-1 text-xs ${toneClass[tone]}`}>{label}</span>;
}
