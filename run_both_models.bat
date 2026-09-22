@echo off
REM ---------------------------------------------------------------------------
REM  Run Model 1 and Model 2 side by side on this laptop.
REM
REM  Model 1 = DINOv3 ViT-S + Faster R-CNN   (accurate, heavy)
REM  Model 2 = DINOv3 ViT-Ti + FCOS (light)  (2.7x faster, batch-exportable)
REM
REM  Each model alarms at ITS OWN operating point from the test evaluation:
REM  the two heads score on different scales, so a shared threshold would be
REM  unfair (Model 2 never scores above ~0.75). See reports\Model1_vs_Model2_Comparison.pdf
REM
REM  Press q in the video window to stop.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
set MODEL1=models\model1_vitS_fasterrcnn.pt
set MODEL2=models\model2_vitTi_fcos_light.pt
set CONF1=0.90
set CONF2=0.55
REM Site footage is confidential and git-ignored; use whichever clip is in Industry\video.
set SITE_VIDEO=
for %%F in ("Industry\video\*.mov" "Industry\video\*.mp4") do if not defined SITE_VIDEO set "SITE_VIDEO=%%~F"

if not exist "%PY%" (
    echo Python environment not found at %PY%
    goto :end
)
if not exist "%MODEL1%" ( echo Missing %MODEL1% & goto :end )
if not exist "%MODEL2%" ( echo Missing %MODEL2% & goto :end )

echo.
echo   1  Webcam - live, both models side by side
echo   2  Client site video - 2 minutes, saved to Industry\demo\
echo   3  Any video file you choose
echo.
set /p CHOICE=Choose 1, 2 or 3:

if "%CHOICE%"=="1" goto :webcam
if "%CHOICE%"=="2" goto :site
if "%CHOICE%"=="3" goto :file
echo Not a valid choice.
goto :end

:webcam
echo Starting webcam. Each frame runs through both models, about 1 frame per second on this CPU.
"%PY%" scripts\compare_models.py ^
    --model "Model 1: ViT-S + Faster R-CNN=%MODEL1%@%CONF1%" ^
    --model "Model 2: ViT-Ti + FCOS (light)=%MODEL2%@%CONF2%" ^
    --video 0 --show --no-save --enter-hits 3 --window 8 --device cpu --output webcam_compare.mp4
goto :end

:site
if not defined SITE_VIDEO ( echo No video found in Industry\video & goto :end )
echo Using "%SITE_VIDEO%"
"%PY%" scripts\compare_models.py ^
    --model "Model 1: ViT-S + Faster R-CNN=%MODEL1%@%CONF1%" ^
    --model "Model 2: ViT-Ti + FCOS (light)=%MODEL2%@%CONF2%" ^
    --video "%SITE_VIDEO%" --start-seconds 60 --max-seconds 120 --stride 5 ^
    --show --device cpu --output Industry\demo\site_compare.mp4
goto :end

:file
set /p VIDEO=Full path to the video file:
"%PY%" scripts\compare_models.py ^
    --model "Model 1: ViT-S + Faster R-CNN=%MODEL1%@%CONF1%" ^
    --model "Model 2: ViT-Ti + FCOS (light)=%MODEL2%@%CONF2%" ^
    --video "%VIDEO%" --max-seconds 120 --stride 5 ^
    --show --device cpu --output demo_compare.mp4
goto :end

:end
echo.
pause
