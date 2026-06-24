import { execFile } from 'child_process';
import os from 'os';
import path from 'path';
import { promisify } from 'util';

export type GpuProcess = {
  gpuUuid: string;
  pid: number;
  name: string;
  usedMemoryMb?: number;
};

export type GpuInfo = {
  index: number;
  uuid: string;
  name: string;
  driver: string;
  memoryTotalMb: number;
  memoryFreeMb: number;
  memoryUsedMb: number;
  utilizationGpuPercent?: number;
  temperatureC?: number;
  powerDrawW?: number;
  powerLimitW?: number;
  processes: GpuProcess[];
};

export type SystemPayload = {
  ok: boolean;
  generatedAt: string;
  projectRoot: string;
  gpu: {
    available: boolean;
    gpus: GpuInfo[];
    error?: string;
  };
  python: { available: boolean; version?: string; error?: string };
  node: { version: string };
  conda: { available: boolean; paths: string[] };
  wsl: { available: boolean; distributions: string[]; error?: string };
  hfToken: { present: boolean };
  disk: { available: boolean; freeGb?: number; sizeGb?: number; error?: string };
};

export type TelemetryPayload = {
  ok: boolean;
  generatedAt: string;
  gpu: SystemPayload['gpu'];
};

const execFileAsync = promisify(execFile);

async function run(command: string, args: string[] = [], timeout = 7000) {
  try {
    const { stdout, stderr } = await execFileAsync(command, args, {
      timeout,
      windowsHide: true,
      env: { ...process.env, CUDA_DEVICE_ORDER: 'PCI_BUS_ID' },
    });
    return { ok: true, stdout: String(stdout), stderr: String(stderr) };
  } catch (error: any) {
    return {
      ok: false,
      stdout: String(error?.stdout || ''),
      stderr: String(error?.stderr || error?.message || ''),
    };
  }
}

function maybeNumber(value: string) {
  const cleaned = value.replace(/\[N\/A\]/gi, '').replace(/N\/A/gi, '').trim();
  if (!cleaned) return undefined;
  const numberValue = Number(cleaned);
  return Number.isFinite(numberValue) ? numberValue : undefined;
}

function parseCsvLine(line: string) {
  return line.split(',').map(part => part.trim());
}

async function probeGpuProcesses() {
  const result = await run('nvidia-smi', [
    '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
    '--format=csv,noheader,nounits',
  ]);
  if (!result.ok) return [];

  return result.stdout
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map(line => {
      const [gpuUuid, pid, name, usedMemory] = parseCsvLine(line);
      return {
        gpuUuid,
        pid: Number(pid),
        name,
        usedMemoryMb: maybeNumber(usedMemory),
      };
    })
    .filter(processInfo => processInfo.gpuUuid && Number.isFinite(processInfo.pid));
}

export async function probeGpu() {
  const [gpuResult, processes] = await Promise.all([
    run('nvidia-smi', [
      '--query-gpu=index,uuid,name,driver_version,memory.total,memory.free,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit',
      '--format=csv,noheader,nounits',
    ]),
    probeGpuProcesses(),
  ]);

  if (!gpuResult.ok) {
    return { available: false, gpus: [], error: gpuResult.stderr || 'nvidia-smi failed' };
  }

  const gpus = gpuResult.stdout
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map(line => {
      const [index, uuid, name, driver, total, free, used, utilization, temperature, powerDraw, powerLimit] = parseCsvLine(line);
      const memoryTotalMb = maybeNumber(total) ?? 0;
      const memoryFreeMb = maybeNumber(free) ?? 0;
      const memoryUsedMb = maybeNumber(used) ?? Math.max(0, memoryTotalMb - memoryFreeMb);
      return {
        index: Number(index),
        uuid,
        name,
        driver,
        memoryTotalMb,
        memoryFreeMb,
        memoryUsedMb,
        utilizationGpuPercent: maybeNumber(utilization),
        temperatureC: maybeNumber(temperature),
        powerDrawW: maybeNumber(powerDraw),
        powerLimitW: maybeNumber(powerLimit),
        processes: processes.filter(processInfo => processInfo.gpuUuid === uuid),
      };
    })
    .filter(gpu => Number.isFinite(gpu.index) && gpu.name);

  return { available: gpus.length > 0, gpus };
}

async function probePython() {
  const result = await run('python', ['--version']);
  return {
    available: result.ok,
    version: (result.stdout || result.stderr).trim(),
    error: result.ok ? undefined : result.stderr,
  };
}

async function probeConda() {
  const isWindows = os.platform() === 'win32';
  const result = isWindows ? await run('where.exe', ['conda']) : await run('which', ['conda']);
  const paths = result.stdout
    .trim()
    .split(/\r?\n/)
    .map(item => item.trim())
    .filter(Boolean);
  return { available: result.ok && paths.length > 0, paths };
}

async function probeWsl() {
  if (os.platform() !== 'win32') {
    return { available: false, distributions: [], error: 'WSL probe is Windows-only' };
  }
  const result = await run('wsl.exe', ['-l', '-v']);
  const clean = result.stdout.replace(/\u0000/g, '');
  const distributions = clean
    .split(/\r?\n/)
    .map(line => line.replace(/^\*\s*/, '').trim())
    .filter(line => line && !line.startsWith('NAME'))
    .map(line => line.split(/\s{2,}/)[0])
    .filter(Boolean);
  return {
    available: result.ok && distributions.length > 0,
    distributions,
    error: result.ok ? undefined : result.stderr,
  };
}

async function probeDisk(projectRoot: string) {
  if (os.platform() !== 'win32') {
    const free = os.freemem() / 1024 / 1024 / 1024;
    return { available: true, freeGb: free, sizeGb: undefined };
  }

  const root = path.parse(projectRoot).root.replace(/\\$/, '').replace(':', '');
  const result = await run('powershell.exe', [
    '-NoProfile',
    '-Command',
    `(Get-PSDrive -Name ${root} | Select-Object Used,Free | ConvertTo-Json -Compress)`,
  ]);
  if (!result.ok) {
    return { available: false, error: result.stderr };
  }
  try {
    const parsed = JSON.parse(result.stdout);
    const freeGb = Number(parsed.Free) / 1024 / 1024 / 1024;
    const sizeGb = (Number(parsed.Free) + Number(parsed.Used)) / 1024 / 1024 / 1024;
    return { available: true, freeGb, sizeGb };
  } catch {
    return { available: false, error: 'Could not parse disk data' };
  }
}

export async function probeSystem(projectRoot: string): Promise<SystemPayload> {
  const [gpu, python, conda, wsl, disk] = await Promise.all([
    probeGpu(),
    probePython(),
    probeConda(),
    probeWsl(),
    probeDisk(projectRoot),
  ]);

  return {
    ok: true,
    generatedAt: new Date().toISOString(),
    projectRoot,
    gpu,
    python,
    node: { version: process.version },
    conda,
    wsl,
    hfToken: { present: Boolean(process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN) },
    disk,
  };
}
