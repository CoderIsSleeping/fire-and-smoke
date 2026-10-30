@echo off
REM ============================================================================
REM  sample_run.bat - run the current fire / smoke model on a video
REM
REM  Double-click this file, or run it from the project folder:
REM      sample_run.bat
REM
REM  Model     : Model 7 (DINOv3 + Faster R-CNN, 896 px, negative training)
REM  Alarm     : confidence >= 0.80, confirmed after 6 detections in 15 checks
REM  Speed     : about 2 s per checked frame on this laptop (CPU). The model checks
REM              every 5th frame, so a 10-second clip takes about 2-3 minutes.
REM  Output    : an annotated video and a CSV log of alarms, saved in this folder
REM  Window    : a live window shows the detections; press q to stop early
REM ============================================================================

cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set MODEL=models\model7_dinov3_896_negs.pt
set CONF=0.80
set EXIT=0.65

if not exist "%MODEL%" ( echo Missing %MODEL% & pause & exit /b 1 )

:menu
echo.
echo  1. Short test clip  (10 s, about 2-3 minutes)     Industry\test\fire_test.mp4
echo  2. Long test clip   (101 s, about 17 minutes)     Industry\test\fire_test2.mp4
echo  3. Any video file   (you type the path)
echo  4. Laptop webcam    (live; press q to stop)
echo  5. Quit
set CHOICE=
set /p CHOICE=Choose 1-5:
if "%CHOICE%"=="1" set VIDEO=Industry\test\fire_test.mp4& set NAME=sample_short& goto run
if "%CHOICE%"=="2" set VIDEO=Industry\test\fire_test2.mp4& set NAME=sample_long& goto run
if "%CHOICE%"=="3" goto ask
if "%CHOICE%"=="4" goto webcam
if "%CHOICE%"=="5" exit /b 0
goto menu

:ask
set VIDEO=
set /p VIDEO=Full path to the video file:
set NAME=sample_video
goto run

:run
if not exist "%VIDEO%" ( echo File not found: %VIDEO% & goto menu )
echo Running Model 7 on %VIDEO% ...
"%PY%" scripts\predict_video_dinov3.py --weights "%MODEL%" --label "Model 7" ^
    --source "%VIDEO%" --show --conf %CONF% --exit-conf %EXIT% ^
    --stride 5 --device cpu --output %NAME%_model7.mp4 --events %NAME%_model7_events.csv
echo.
echo Saved %NAME%_model7.mp4 (annotated video) and %NAME%_model7_events.csv (alarm log)
goto menu

:webcam
echo Starting webcam ... press q in the video window to stop.
"%PY%" scripts\predict_video_dinov3.py --weights "%MODEL%" --label "Model 7" ^
    --source 0 --show --conf %CONF% --exit-conf %EXIT% --device cpu ^
    --output webcam_model7.mp4 --events webcam_model7_events.csv
goto menu

REM ============================================================================
REM  The same commands typed by hand (PowerShell, from the project folder):
REM
REM  Short clip:
REM    .venv\Scripts\python.exe scripts\predict_video_dinov3.py --weights models\model7_dinov3_896_negs.pt --label "Model 7" --source Industry\test\fire_test.mp4 --show --conf 0.80 --exit-conf 0.65 --stride 5 --device cpu --output sample_short_model7.mp4 --events sample_short_model7_events.csv
REM
REM  Any video (replace the path):
REM    .venv\Scripts\python.exe scripts\predict_video_dinov3.py --weights models\model7_dinov3_896_negs.pt --label "Model 7" --source "C:\path\to\video.mp4" --show --conf 0.80 --exit-conf 0.65 --stride 5 --device cpu --output sample_video_model7.mp4 --events sample_video_model7_events.csv
REM
REM  Small, distant fires (slower, about 5x): add  --tiles 2
REM  Faster preview (less detail): add  --imgsz 640
REM ============================================================================
