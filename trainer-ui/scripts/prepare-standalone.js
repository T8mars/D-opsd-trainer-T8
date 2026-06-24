const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');

const uiRoot = path.resolve(__dirname, '..');
const standaloneRoot = path.join(uiRoot, '.next/standalone');
const staticRoot = path.join(uiRoot, '.next/static');
const publicRoot = path.join(uiRoot, 'public');

async function exists(filePath) {
  try {
    await fsp.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function copyRecursive(source, target) {
  if (!(await exists(source))) return;
  const stat = await fsp.stat(source);
  if (stat.isDirectory()) {
    await fsp.mkdir(target, { recursive: true });
    const entries = await fsp.readdir(source, { withFileTypes: true });
    await Promise.all(entries.map(entry => copyRecursive(path.join(source, entry.name), path.join(target, entry.name))));
    return;
  }
  await fsp.mkdir(path.dirname(target), { recursive: true });
  await fsp.copyFile(source, target);
}

function findServerDir(root, depth = 6) {
  if (!fs.existsSync(root) || depth < 0) return null;
  const directServer = path.join(root, 'server.js');
  if (fs.existsSync(directServer)) return root;
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const nested = findServerDir(path.join(root, entry.name), depth - 1);
    if (nested) return nested;
  }
  return null;
}

async function main() {
  const serverDir = findServerDir(standaloneRoot);
  if (!serverDir) {
    throw new Error(`Cannot find standalone server.js under ${standaloneRoot}`);
  }

  await copyRecursive(staticRoot, path.join(serverDir, '.next', 'static'));
  await copyRecursive(publicRoot, path.join(serverDir, 'public'));
  console.log(JSON.stringify({ ok: true, standaloneRoot, serverDir }));
}

main().catch(error => {
  console.error(error);
  process.exit(1);
});
