@echo off
REM ---------------------------------------------------------------------------
REM Windows entry point that works regardless of PowerShell's execution policy.
REM
REM A fresh Windows client defaults to ExecutionPolicy=Restricted, which refuses
REM to run run_experiments.ps1 at all ("выполнение сценариев отключено в этой
REM системе" / "running scripts is disabled on this system"). Batch files are not
REM subject to that policy, so this wrapper launches the PowerShell script with a
REM per-process bypass -- nothing about the machine's policy is changed.
REM
REM   run_experiments.cmd geowalker
REM   run_experiments.cmd gw_test | gw_field | gw_train | gw_infer
REM
REM -NoProfile keeps a user's $PROFILE from injecting anything into the run.
REM ---------------------------------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_experiments.ps1" %*
exit /b %ERRORLEVEL%
