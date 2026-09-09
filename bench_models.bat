@echo off
rem Banc d'essai: chaque modele de la bibliotheque, trois images, temps separes
rem (chargement / 1re image / regime etabli). Ecrit bench\REPORT.md et les images.
rem
rem LE GPU DOIT ETRE LIBRE: ferme l'app avant de lancer.
rem Relancable: --resume saute les modeles deja mesures.
rem
rem Options: bench_models.bat --list              (montre le plan, ne genere rien)
rem          bench_models.bat --only rayKlein     (un seul modele; repetable)
rem          bench_models.bat --size 832x1216     (defaut 1024x1024)
rem          bench_models.bat --steps 8           (meme steps pour tous)
rem          bench_models.bat --resume            (reprend apres une coupure)
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe tools\bench_models.py %*
pause
