@echo off
rem A bench: every model of the library, three images, timed separately
rem (load / 1st image / steady state). Writes bench\REPORT.md and the images.
rem
rem THE GPU MUST BE FREE: close the app before launching this.
rem Re-runnable: --resume skips the models already measured.
rem
rem Options: bench_models.bat --list              (shows the plan, generates nothing)
rem          bench_models.bat --only rayKlein     (a single model; repeatable)
rem          bench_models.bat --size 832x1216     (default 1024x1024)
rem          bench_models.bat --steps 8           (the same steps for every model)
rem          bench_models.bat --resume            (resumes after an interruption)
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe tools\bench_models.py %*
pause
