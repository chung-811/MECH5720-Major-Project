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
rem OneDrive marks folders read-only, and tar carries that over, so the Pi's old copy is removed and a fresh one
rem unpacked (the Pi keeps nothing of its own in these folders). The Pi's clock is often behind: hide those warnings.
tar -cf - --exclude __pycache__ -C software\data_capture %FOLDER% coded | ssh %SSH_OPTS% %PI% "chmod -R u+w ~/%FOLDER% ~/coded 2>/dev/null; rm -rf ~/%FOLDER% ~/coded && tar -xf - -C ~ --warning=no-timestamp --warning=no-unknown-keyword && chmod -R u+w ~/%FOLDER% ~/coded"
if errorlevel 255 (set "MSG=could not reach the Pi: is it on, plugged into this computer by ethernet, and is %PI% right?" & goto :fail)
if errorlevel 1 (set "MSG=reached the Pi but could not replace ~/%FOLDER% and ~/coded there: see the lines above" & goto :fail)

rem The tunnel needs port %PORT% free on this computer (an ssh -L window left open from the README steps holds it).
netstat -an | findstr /r /c:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (set "MSG=port %PORT% is already in use on this computer: close any other ssh -L %PORT%:... window (or Pi camera server window) and run this again" & goto :fail)

echo.
echo Starting the camera server on the Pi in a NEW WINDOW: type the Pi password (mech5720) there.
echo Leave that window open while you use the camera; closing it stops the server.
rem On the Pi: stop any server left from an earlier run (the pattern must start at the python path, or it would match
rem this shell's own command line); use the first Python that has flask and picamera2 (the system one, or a venv
rem in the home folder such as ~/mech5720 from the lab handout); and if the server stops (an error, or Ctrl+C),
rem wait for Enter so the error can be read.
set "STOP=pkill -f '^[^ ]*python3? .*stream[.]py'; sleep 1"
set "FINDPY=for p in python3 ~/*/bin/python3; do $p -c 'import flask, picamera2' 2>/dev/null && break; p=; done"
set "NOPY=echo '*** No Python on the Pi has flask and picamera2. On the Pi run: pip install flask --break-system-packages'"
set "SERVER=cd ~/%FOLDER% && echo Using $p && $p stream.py"
set "WAIT=echo; echo '*** The camera server stopped: see the lines above. Press Enter to close.'; read x"
start "Pi camera server" ssh %SSH_OPTS% -tt -o ExitOnForwardFailure=yes -L %PORT%:localhost:8000 %PI% "%STOP%; %FINDPY%; if [ $p ]; then %SERVER%; else %NOPY%; fi; %WAIT%"

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
