@echo off
REM PLWM resident web explorer - crash-restart loop
REM The architecture roams the real internet (all modalities), learns via the
REM surprise engine, monitors with narrative. No steering, no cut-off.
REM STOP file kills it gracefully: web-resident-STOP

setlocal
cd /d "%~dp0"
set PY=python
set STATUS=%~dp0web-resident-status.json
set CHECKPOINT=%~dp0web-resident-checkpoint.json
set STOPFILE=%~dp0web-resident-STOP
set LOG=%~dp0web-resident-loop.log

:loop
if exist "%STOPFILE%" (
  del "%STOPFILE%"
  goto :eof
)
echo [%date% %time%] starting resident round >> "%LOG%"
"%PY%" -u experiments\wander_resident.py --status "%STATUS%" --checkpoint "%CHECKPOINT%"
echo [%date% %time%] resident exited (code %errorlevel%), restarting in 10s >> "%LOG%"
timeout /t 10 /nobreak >nul
goto loop
