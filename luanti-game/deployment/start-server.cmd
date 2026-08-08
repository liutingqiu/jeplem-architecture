@echo off
setlocal
set "ROOT=%~dp0"
if not exist "%ROOT%logs" mkdir "%ROOT%logs"
"%ROOT%engine\luanti-5.16.1-win64\bin\luanti.exe" --server --world "%ROOT%world" --config "%ROOT%luanti.conf" --logfile "%ROOT%logs\server.log" --color never
endlocal
