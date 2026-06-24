@echo off
setlocal

set "PATH=%Path%"
set "SCRIPT_DIR=%~dp0"
set "LAUNCHER_ARGS=%*"
set "SHOULD_WAIT=1"
if "%~1"=="" (
  set "LAUNCHER_ARGS=-Wait"
  set "SHOULD_WAIT=0"
)

echo(%LAUNCHER_ARGS% | findstr /I /C:"-SmokeTest" >nul 2>nul
if %ERRORLEVEL% EQU 0 set "SHOULD_WAIT=0"
echo(%LAUNCHER_ARGS% | findstr /I /C:"-Wait" >nul 2>nul
if %ERRORLEVEL% EQU 0 set "SHOULD_WAIT=0"
if "%SHOULD_WAIT%"=="1" (
  set "LAUNCHER_ARGS=%LAUNCHER_ARGS% -Wait"
)

where pwsh.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  pwsh.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\start_trainer.ps1" %LAUNCHER_ARGS%
) else (
  where powershell.exe >nul 2>nul
  if %ERRORLEVEL% EQU 0 (
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%scripts\start_trainer.ps1" %LAUNCHER_ARGS%
  ) else (
    echo PowerShell was not found. Install PowerShell or enable Windows PowerShell, then rerun this launcher.
    exit /b 1
  )
)

set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
  echo.
  echo D-OPSD Trainer launcher failed with exit code %EXITCODE%.
  echo Check trainer-data\launcher\launcher.stderr.log for details.
  echo Press any key to close.
  pause >nul
)

exit /b %EXITCODE%
