# T8 D-OPSD Tranier Electron Distribution

## What Is Bundled

The Electron package bundles the Next.js trainer UI, runtime bridge, D-OPSD training scripts, smoke scripts, setup scripts, and project documentation.

The packaged app copies those files into a writable workspace on first launch:

```text
%APPDATA%\d-opsd-trainer-ui\workspace
```

User data such as `trainer-data`, model caches, datasets, jobs, logs, outputs, samples, and checkpoints stays in that writable workspace.

Electron main-process logs are written here:

```text
%APPDATA%\d-opsd-trainer-ui\logs\electron-main.log
```

## Build Commands

From the repository root:

```powershell
npm install --prefix trainer-ui
npm run pack:win --prefix trainer-ui
```

The unpacked app is:

```text
trainer-ui\release\win-unpacked\T8 D-OPSD Tranier.exe
```

To build installer and portable artifacts:

```powershell
npm run dist:win --prefix trainer-ui
```

## Smoke Test

The packaged app supports a no-window smoke test:

```powershell
$env:DOPSD_ELECTRON_PORT = "19075"
$env:DOPSD_ELECTRON_SMOKE_FILE = "$PWD\trainer-data\setup\electron-smoke.json"
Start-Process -FilePath ".\trainer-ui\release\win-unpacked\T8 D-OPSD Tranier.exe" -ArgumentList "--smoke-test" -WindowStyle Hidden -Wait
Get-Content .\trainer-data\setup\electron-smoke.json
```

The smoke test starts the packaged Next.js server, verifies `/api/project`, writes the result file, and exits.

## Fresh Machine Training Setup

Training still requires WSL2 Ubuntu and NVIDIA GPU drivers on the target machine. After opening the packaged app once, run this from the writable workspace or repository root:

```powershell
.\scripts\setup_wsl_trainer.ps1
```

That script:

- verifies `wsl.exe` and the `Ubuntu-22.04` distro,
- converts the Windows project path to a WSL path,
- sources `scripts/dopsd_wsl_env.sh`,
- creates `trainer-data/venvs/dopsd`,
- installs `requirements-trainer.txt`,
- runs `pip check`,
- runs runtime probe/settings checks.

Use `-SkipPipInstall -SkipProbe` only for a quick script-path smoke test.
