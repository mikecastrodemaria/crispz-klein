@echo off
title crispz-klein - RTX 5090 (local)
cd /d "%~dp0"
echo ============================================
echo  crispz-klein - RTX 5090 (local 127.0.0.1)
echo ============================================
echo.
REM CUDA optimisations (harmless, BF16)
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set GRADIO_SERVER_PORT=7860
REM A UTF-8 console (avoids the cp1252 crashes on the HF progress bars)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM === LOCAL-ONLY: uses the HF cache ONLY, never RE-downloads the model.
REM black-forest-labs/FLUX.2-klein-4B weighs ~15 GB and is cached on the 1st launch.
REM Comment this line out (REM) to allow a download (a new checkpoint, a new LoRA),
REM then put it back. ===
set HF_HUB_OFFLINE=1
REM === VRAM (RTX 5090, 32 GB). klein-4B fits ENTIRELY in VRAM: ~15 GB for the whole
REM surface (txt2img + edit + inpaint + img2img share the same loaded model).
REM So the offload is NOT needed here, unlike on the 20B forks.
REM   - >= 16 GB of VRAM: leave 'none' (the default below), it is the fastest.
REM   - < 16 GB: set 'model' (offloaded per submodule, slower but it fits). ===
set CZ_OFFLOAD=none
REM Delegates to run.bat (venv detection + ESRGAN_DIR + launch)
call "%~dp0run.bat" %*
