@echo off
title crispz-klein - RTX 5090 (local)
cd /d "%~dp0"
echo ============================================
echo  crispz-klein - RTX 5090 (local 127.0.0.1)
echo ============================================
echo.
REM Optimisations CUDA (sans danger, BF16)
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set GRADIO_SERVER_PORT=7860
REM Console UTF-8 (evite les crashs cp1252 sur les barres de progression HF)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM === LOCAL-ONLY: utilise UNIQUEMENT le cache HF, ne RE-telecharge jamais le modele.
REM black-forest-labs/FLUX.2-klein-4B fait ~15 Go et se met en cache au 1er lancement.
REM Mets cette ligne en commentaire (REM) pour autoriser un telechargement (nouveau
REM checkpoint, nouveau LoRA), puis remets-la. ===
set HF_HUB_OFFLINE=1
REM === VRAM (RTX 5090, 32 Go). klein-4B tient ENTIER en VRAM: ~15 Go pour toute la
REM surface (txt2img + edit + inpaint + img2img partagent le meme modele charge).
REM L'offload n'est donc PAS necessaire ici, contrairement aux forks 20B.
REM   - >= 16 Go de VRAM: laisser 'none' (defaut ci-dessous), c'est le plus rapide.
REM   - < 16 Go: mettre 'model' (decharge par sous-module, plus lent mais tient). ===
set CZ_OFFLOAD=none
REM Delegue au run.bat (detection venv + ESRGAN_DIR + lancement)
call "%~dp0run.bat" %*
