import path from 'path';

export function resolveProjectRoot() {
  const configuredRoot = process.env.DOPSD_PROJECT_ROOT?.trim();
  if (configuredRoot) return path.resolve(configuredRoot);
  return path.resolve(process.cwd(), '..');
}
