@echo off
rem Pre-fills the dequantisation cache of the single-file FP8/INT8 checkpoints, so
rem that the conversion is not paid for at the first use.
rem Re-runnable at will: whatever is already cached is skipped in a second.
rem Options: rebuild_cache.bat --list           (shows without converting)
rem          rebuild_cache.bat --cpu            (dequantises without touching the GPU)
rem          rebuild_cache.bat --only rayKlein  (a single model, filtered on the name; repeatable)
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe tools\rebuild_dequant_cache.py %*
pause
