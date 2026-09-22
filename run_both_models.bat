@echo off
REM ---------------------------------------------------------------------------
REM  Run Model 1 and Model 2 on this laptop - one at a time, or side by side.
REM
REM  Model 1 = DINOv3 ViT-S + Faster R-CNN   (accurate, heavy)
REM  Model 2 = DINOv3 ViT-Ti + FCOS (light)  (about 2.7x faster, batch-exportable)
REM
REM  Each model alarms at ITS OWN operating point from the test evaluation:
REM  the two heads score on different scales, so a shared threshold would be
REM  unfair (Model 2 never scores above about 0.75).
REM  See reports\Model1_vs_Model2_Comparison.pdf
REM
REM  Press q in the video window to stop.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set PY=.venv\Scripts\python.exe
set MODEL1=models\model1_vitS_fasterrcnn.pt
set MODEL2=models\model2_vitTi_fcos_light.pt
set CONF1=0.90
set EXIT1=0.75
set CONF2=0.55
set EXIT2=0.40

REM Site footage is confidential and git-ignored; use whichever clip is in Industry\video.
set SITE_VIDEO=
for %%F in ("Industry\video\*.mov" "Industry\video\*.mp4") do if not defined SITE_VIDEO set "SITE_VIDEO=%%~F"

if not exist "%PY%" ( echo Python environment not found at %PY% & goto :end )
if not exist "%MODEL1%" ( echo Missing %MODEL1% & goto :end )
if not exist "%MODEL2%" ( echo Missing %MODEL2% & goto :end )

:menu
echo.
echo   ONE MODEL AT A TIME
echo     1  Model 1 - webcam
echo     2  Model 2 - webcam
echo     3  Model 1 - site video
echo     4  Model 2 - site video
echo.
echo   BOTH SIDE BY SIDE
echo     5  Both - webcam
echo     6  Both - site video
echo     7  Both - any video file
echo.
echo     0  Exit
echo.
set CHOICE=
set /p CHOICE=Choose:

if "%CHOICE%"=="1" goto :m1_cam
if "%CHOICE%"=="2" goto :m2_cam
if "%CHOICE%"=="3" goto :m1_site
if "%CHOICE%"=="4" goto :m2_site
if "%CHOICE%"=="5" goto :both_cam
if "%CHOICE%"=="6" goto :both_site
if "%CHOICE%"=="7" goto :both_file
if "%CHOICE%"=="0" goto :end
echo Not a valid choice.
goto :menu

REM ---------------- one model at a time ----------------

:m1_cam
"%PY%" scripts\predict_video_dinov3.py --weights "%MODEL1%" --label "Model 1" ^
    --source 0 --show --no-save --conf %CONF1% --exit-conf %EXIT1% ^
    --window 8 --enter-hits 3 --exit-hits 1 --device cpu --events model1_webcam_events.csv
goto :menu

:m2_cam
"%PY%" scripts\predict_video_dinov3.py --weights "%MODEL2%" --label "Model 2 (light)" ^
    --source 0 --show --no-save --conf %CONF2% --exit-conf %EXIT2% ^
    --window 8 --enter-hits 3 --exit-hits 1 --device cpu --events model2_webcam_events.csv
goto :menu

:m1_site
if not defined SITE_VIDEO ( echo No video found in Industry\video & goto :menu )
echo Using "%SITE_VIDEO%"
"%PY%" scripts\predict_video_dinov3.py --weights "%MODEL1%" --label "Model 1" ^
    --source "%SITE_VIDEO%" --show --conf %CONF1% --exit-conf %EXIT1% ^
    --stride 5 --max-frames 3000 --device cpu --output Industry\demo\model1_site.mp4
goto :menu

:m2_site
if not defined SITE_VIDEO ( echo No video found in Industry\video & goto :menu )
echo Using "%SITE_VIDEO%"
"%PY%" scripts\predict_video_dinov3.py --weights "%MODEL2%" --label "Model 2 (light)" ^
    --source "%SITE_VIDEO%" --show --conf %CONF2% --exit-conf %EXIT2% ^
    --stride 5 --max-frames 3000 --device cpu --output Industry\demo\model2_site.mp4
goto :menu

REM ---------------- side by side ----------------

:both_cam
"%PY%" scripts\compare_models.py ^
    --model "Model 1: ViT-S + Faster R-CNN=%MODEL1%@%CONF1%" ^
    --model "Model 2: ViT-Ti + FCOS (light)=%MODEL2%@%CONF2%" ^
    --video 0 --show --no-save --enter-hits 3 --window 8 --device cpu --output webcam_compare.mp4
goto :menu

:both_site
if not defined SITE_VIDEO ( echo No video found in Industry\video & goto :menu )
echo Using "%SITE_VIDEO%"
"%PY%" scripts\compare_models.py ^
    --model "Model 1: ViT-S + Faster R-CNN=%MODEL1%@%CONF1%" ^
    --model "Model 2: ViT-Ti + FCOS (light)=%MODEL2%@%CONF2%" ^
    --video "%SITE_VIDEO%" --start-seconds 60 --max-seconds 120 --stride 5 ^
    --show --device cpu --output Industry\demo\site_compare.mp4
goto :menu

:both_file
set VIDEO=
set /p VIDEO=Full path to the video file:
"%PY%" scripts\compare_models.py ^
    --model "Model 1: ViT-S + Faster R-CNN=%MODEL1%@%CONF1%" ^
    --model "Model 2: ViT-Ti + FCOS (light)=%MODEL2%@%CONF2%" ^
    --video "%VIDEO%" --max-seconds 120 --stride 5 ^
    --show --device cpu --output demo_compare.mp4
goto :menu

:end
endlocal
