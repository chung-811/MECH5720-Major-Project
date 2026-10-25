@echo off
rem The one file for the lab, on Windows. Double-click it (or run start-windows.bat in a terminal).
rem 1. installs uv if missing (it brings Python and every package)
rem 2. sets up the deblur code and checks it (software\deconvolution\check.py)
rem 3. installs the camera GUI's packages
rem 4. copies the camera server to the Pi, starts it, and opens the camera GUI (Ollie's software\data_capture)
rem Run it again any time: everything already done is skipped in seconds.
setlocal
cd /d "%~dp0"

rem our Pi; password mech5720. To use another: set PI=pi@other.local before running
if not defined PI set PI=pi@eb5234f9picam.local
set PORT=8089
set SSH_OPTS=-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10
set GUI=uv run --no-project --python 3.13 --with-requirements requirements.txt

echo.
echo == 1/4 uv
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (set "MSG=could not install uv: check the internet connection" & goto :fail)
)
uv --version
if errorlevel 1 (set "MSG=uv is not working: close this window and run the file again" & goto :fail)

echo.
echo == 2/4 deblur code (first time: about 1 GB of packages and 131 MB of model weights)
pushd software\deconvolution
uv sync -q
if errorlevel 1 (popd & set "MSG=the deblur packages did not install: check the internet connection" & goto :fail)
uv run python check.py
if errorlevel 1 (popd & set "MSG=the deblur code is not ready: see the line above" & goto :fail)
popd

echo.
echo == 3/4 camera GUI packages
%GUI% python -c "import imgui_bundle, glfw, cv2, requests" 2>nul
if errorlevel 1 (set "MSG=the camera GUI packages did not install: run this again with the internet on" & goto :fail)
echo ok

echo.
echo == 4/4 camera
set "CAM=g"
set /p "CAM=Connect to which camera? [g] global shutter, [r] rolling shutter, [s] skip  (Enter = g): "
if /i "%CAM%"=="g" (set FOLDER=global) else if /i "%CAM%"=="r" (set FOLDER=rolling) else (
    echo.
    echo Set up and checked. Not connecting to a camera.
    goto :end
)

echo Copying the %FOLDER% camera server to the Pi (%PI%). Password: mech5720
tar -C software\data_capture -cf - %FOLDER% coded | ssh %SSH_OPTS% %PI% "chmod -R u+w ~/%FOLDER% ~/coded 2>/dev/null; tar -xf - -C ~ && chmod -R u+w ~/%FOLDER% ~/coded"
if errorlevel 1 (set "MSG=could not reach the Pi: is it on, plugged into this computer by ethernet, and is %PI% right?" & goto :fail)

echo.
echo Starting the camera server on the Pi in a NEW WINDOW: type the Pi password (mech5720) there.
echo Leave that window open while you use the camera; closing it stops the server.
start "Pi camera server" ssh %SSH_OPTS% -tt -o ExitOnForwardFailure=yes -L %PORT%:localhost:8000 %PI% "pkill -f '[s]tream.py'; sleep 1; cd ~/%FOLDER% && exec python3 stream.py"

echo Waiting for the camera (up to 2 minutes, while you type the password)...
for /l %%i in (1,1,120) do (
    curl -s --max-time 5 -o nul http://localhost:%PORT%/ && goto :up
    timeout /t 1 /nobreak >nul
)
set "MSG=the camera server did not come up: check the server window for errors (is the %FOLDER% camera the one plugged into the Pi?)"
goto :fail

:up
echo Camera is up. Opening the camera GUI: captures go to output\. Close its window when done.
%GUI% python software\data_capture\%FOLDER%\camview_capture.py --host localhost --port %PORT%
echo.
echo Camera GUI closed. Close the "Pi camera server" window to stop the server.
goto :end

:fail
echo.
echo *** %MSG%
pause
exit /b 1

:end
pause
exit /b 0
