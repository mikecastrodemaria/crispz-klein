@echo off
rem Pre-remplit le cache de dequantification des checkpoints FP8/INT8 single-file,
rem pour ne pas payer la conversion a la premiere utilisation.
rem Relancable a volonte: ce qui est deja en cache est saute en une seconde.
rem Options: rebuild_cache.bat --list           (montre sans convertir)
rem          rebuild_cache.bat --cpu            (dequantification sans toucher au GPU)
rem          rebuild_cache.bat --only rayKlein  (un seul modele, filtre sur le nom; repetable)
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe tools\rebuild_dequant_cache.py %*
pause
