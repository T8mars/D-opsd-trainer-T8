const { app, BrowserWindow, dialog } = require('electron');
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const fsp = require('fs/promises');
const http = require('http');
const net = require('net');
const path = require('path');

const PRODUCT_NAME = 'T8 D-OPSD Tranier';
const SMOKE_TEST = process.argv.includes('--smoke-test') || process.env.DOPSD_ELECTRON_SMOKE === '1';
const TEMPLATE_EXCLUDES = new Set([
  '.git',
  '.next',
  'node_modules',
  'release',
  'dist',
  'build',
  'out',
  '__pycache__',
  '.pytest_cache',
  'trainer-data',
]);

let serverProcess = null;
let serverStopping = false;
let currentBaseUrl = null;

function devProjectRoot() {
  return path.resolve(__dirname, '..', '..');
}

function resolveWorkspaceRoot() {
  const configured = process.env.DOPSD_PROJECT_ROOT?.trim();
  if (configured) return path.resolve(configured);
  if (!app.isPackaged) return devProjectRoot();
  return path.join(app.getPath('userData'), 'workspace');
}

function resourcesWorkspaceTemplate() {
  return path.join(process.resourcesPath, 'workspace-template');
}

function electronLogPath() {
  try {
    return path.join(app.getPath('userData'), 'logs', 'electron-main.log');
  } catch {
    return null;
  }
}

async function appendElectronLog(message) {
  const logPath = electronLogPath();
  if (!logPath) return;
  try {
    await fsp.mkdir(path.dirname(logPath), { recursive: true });
    await fsp.appendFile(logPath, message, 'utf-8');
  } catch {
    // Logging must never crash the desktop launcher.
  }
}

function safeStreamWrite(stream, message) {
  try {
    if (!stream || stream.destroyed || stream.writableEnded) return;
    stream.write(message);
  } catch (error) {
    if (error?.code !== 'EPIPE') {
      void appendElectronLog(`[stream-error] ${error?.stack || error}\n`);
    }
  }
}

function attachBrokenPipeGuard(stream) {
  stream?.on('error', error => {
    if (error?.code !== 'EPIPE') {
      void appendElectronLog(`[stream-error] ${error?.stack || error}\n`);
    }
  });
}

process.stdout?.on('error', error => {
  if (error?.code !== 'EPIPE') void appendElectronLog(`[stdout-error] ${error?.stack || error}\n`);
});
process.stderr?.on('error', error => {
  if (error?.code !== 'EPIPE') void appendElectronLog(`[stderr-error] ${error?.stack || error}\n`);
});

async function exists(filePath) {
  try {
    await fsp.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function copyRecursive(source, target) {
  const stat = await fsp.stat(source);
  const name = path.basename(source);
  if (TEMPLATE_EXCLUDES.has(name)) return;

  if (stat.isDirectory()) {
    await fsp.mkdir(target, { recursive: true });
    const entries = await fsp.readdir(source, { withFileTypes: true });
    for (const entry of entries) {
      await copyRecursive(path.join(source, entry.name), path.join(target, entry.name));
    }
    return;
  }

  await fsp.mkdir(path.dirname(target), { recursive: true });
  await fsp.copyFile(source, target);
}

async function copyTemplateIntoWorkspace(workspaceRoot) {
  if (!app.isPackaged) return;
  const templateRoot = resourcesWorkspaceTemplate();
  if (!(await exists(templateRoot))) {
    throw new Error(`Packaged workspace template not found: ${templateRoot}`);
  }
  await fsp.mkdir(workspaceRoot, { recursive: true });
  await copyRecursive(templateRoot, workspaceRoot);
}

function findStandaloneServer(root, depth = 8) {
  if (!fs.existsSync(root) || depth < 0) return null;
  const direct = path.join(root, 'server.js');
  if (fs.existsSync(direct)) return direct;
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const found = findStandaloneServer(path.join(root, entry.name), depth - 1);
    if (found) return found;
  }
  return null;
}

function packagedServerPath() {
  const appRoot = path.join(process.resourcesPath, 'app');
  const candidates = [
    path.join(appRoot, '.next', 'standalone', 'server.js'),
    path.join(appRoot, '.next', 'standalone', 'trainer-ui', 'server.js'),
    path.join(appRoot, 'trainer-ui', '.next', 'standalone', 'server.js'),
    path.join(appRoot, 'trainer-ui', '.next', 'standalone', 'trainer-ui', 'server.js'),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return findStandaloneServer(path.join(appRoot, '.next', 'standalone'))
    || findStandaloneServer(path.join(appRoot, 'trainer-ui', '.next', 'standalone'));
}

function choosePort() {
  const configured = Number(process.env.DOPSD_ELECTRON_PORT);
  if (Number.isInteger(configured) && configured > 0) {
    return Promise.resolve(configured);
  }

  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = typeof address === 'object' && address ? address.port : 0;
      server.close(() => resolve(port));
    });
  });
}

function startNextServer({ port, workspaceRoot }) {
  const env = {
    ...process.env,
    DOPSD_PROJECT_ROOT: workspaceRoot,
    HOSTNAME: '127.0.0.1',
    PORT: String(port),
    NEXT_TELEMETRY_DISABLED: '1',
  };

  if (app.isPackaged) {
    const serverJs = packagedServerPath();
    if (!serverJs) {
      throw new Error('Cannot find packaged Next standalone server.js');
    }
    serverProcess = spawn(process.execPath, [serverJs], {
      cwd: path.dirname(serverJs),
      env: { ...env, ELECTRON_RUN_AS_NODE: '1' },
      stdio: ['ignore', 'pipe', 'pipe'],
      windowsHide: true,
    });
  } else {
    const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
    const uiRoot = path.resolve(__dirname, '..');
    serverProcess = spawn(
      npmCommand,
      ['run', 'dev', '--prefix', uiRoot, '--', '--hostname', '127.0.0.1', '--port', String(port)],
      {
        cwd: workspaceRoot,
        env,
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      },
    );
  }

  attachBrokenPipeGuard(serverProcess.stdout);
  attachBrokenPipeGuard(serverProcess.stderr);
  serverProcess.stdout?.on('data', chunk => {
    const message = `[next] ${chunk}`;
    safeStreamWrite(process.stdout, message);
    void appendElectronLog(message);
  });
  serverProcess.stderr?.on('data', chunk => {
    const message = `[next] ${chunk}`;
    safeStreamWrite(process.stderr, message);
    void appendElectronLog(message);
  });
  serverProcess.once('exit', (code, signal) => {
    if (!serverStopping && !SMOKE_TEST) {
      dialog.showErrorBox(PRODUCT_NAME, `The local UI server stopped unexpectedly: ${code ?? signal}`);
    }
  });
}

function requestProject(url) {
  return new Promise((resolve, reject) => {
    const request = http.get(url, response => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', chunk => { body += chunk; });
      response.on('end', () => {
        if (response.statusCode !== 200) {
          reject(new Error(`HTTP ${response.statusCode}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    request.setTimeout(5000, () => {
      request.destroy(new Error('Timed out waiting for /api/project'));
    });
    request.on('error', reject);
  });
}

async function waitForServer(baseUrl, timeoutMs = 120000) {
  const deadline = Date.now() + timeoutMs;
  let lastError = null;
  while (Date.now() < deadline) {
    try {
      const payload = await requestProject(`${baseUrl}/api/project`);
      if (payload?.meta?.name === 'D-OPSD Trainer') return payload;
      lastError = new Error('Unexpected project metadata');
    } catch (error) {
      lastError = error;
    }
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  throw lastError || new Error('Timed out waiting for local UI server');
}

function createWindow(baseUrl) {
  const window = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 1080,
    minHeight: 720,
    title: PRODUCT_NAME,
    backgroundColor: '#050707',
    show: false,
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  window.once('ready-to-show', () => window.show());
  window.loadURL(baseUrl);
}

function stopServerProcess() {
  if (!serverProcess?.pid) return;
  serverStopping = true;
  if (process.platform === 'win32') {
    spawnSync('taskkill.exe', ['/PID', String(serverProcess.pid), '/T', '/F'], { windowsHide: true });
  } else {
    serverProcess.kill('SIGTERM');
  }
  serverProcess = null;
}

async function boot() {
  const workspaceRoot = resolveWorkspaceRoot();
  await copyTemplateIntoWorkspace(workspaceRoot);
  const port = await choosePort();
  const baseUrl = `http://127.0.0.1:${port}`;
  currentBaseUrl = baseUrl;

  startNextServer({ port, workspaceRoot });
  await waitForServer(baseUrl);

  if (SMOKE_TEST) {
    const payload = { ok: true, url: baseUrl, workspaceRoot };
    const smokeFile = process.env.DOPSD_ELECTRON_SMOKE_FILE?.trim();
    if (smokeFile) {
      const resolvedSmokeFile = path.resolve(smokeFile);
      await fsp.mkdir(path.dirname(resolvedSmokeFile), { recursive: true });
      await fsp.writeFile(resolvedSmokeFile, `${JSON.stringify(payload, null, 2)}\n`, 'utf-8');
    }
    console.log(JSON.stringify(payload));
    app.quit();
    return;
  }

  createWindow(baseUrl);
}

app.whenReady().then(boot).catch(error => {
  const message = `${error?.stack || error}\n`;
  safeStreamWrite(process.stderr, message);
  void appendElectronLog(message);
  if (!SMOKE_TEST) {
    dialog.showErrorBox(PRODUCT_NAME, error.stack || error.message || String(error));
  }
  app.exit(1);
});

app.on('before-quit', stopServerProcess);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0 && serverProcess) {
    if (currentBaseUrl) createWindow(currentBaseUrl);
  }
});
