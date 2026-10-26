#!/bin/bash
# The one file for the lab, on a Mac. Double-click it in Finder (or run ./start-mac.command).
# 1. installs uv if missing (it brings Python and every package)
# 2. sets up the deblur code and checks it (software/deconvolution/check.py)
# 3. installs the camera GUI's packages
# 4. copies the camera server to the Pi, starts it, and opens the camera GUI (Ollie's software/data_capture)
# Run it again any time: everything already done is skipped in seconds.
set -euo pipefail
cd "$(dirname "$0")"

PI="${PI:-pi@eb5234f9picam.local}"     # our Pi; password mech5720. Override: PI=pi@other.local ./start-mac.command
PORT=8089                             # this computer's end of the tunnel to the Pi's port 8000
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=10)
GUI=(uv run --no-project --python 3.13 --with-requirements requirements.txt)

step() { printf '\n== %s\n' "$1"; }
fail() { printf '\n*** %s\n' "$1"; read -r -p "Press Enter to close. " _; exit 1; }

step "1/4 uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "could not install uv: check the internet connection"
fi
uv --version

step "2/4 deblur code (first time: about 1 GB of packages and 131 MB of model weights)"
(cd software/deconvolution && uv sync -q && uv run python check.py) || fail "the deblur code is not ready: see the lines above"

step "3/4 camera GUI packages"
"${GUI[@]}" python -c "import imgui_bundle, glfw, cv2, requests" 2>/dev/null \
    || fail "the camera GUI packages did not install: run this again with the internet on"
echo "ok"

step "4/4 camera"
read -r -p "Connect to which camera? [g] global shutter, [r] rolling shutter, [s] skip  (Enter = g): " cam
case "${cam:-g}" in
    g|G) folder=global ;;
    r|R) folder=rolling ;;
    *) printf '\nSet up and checked. Not connecting to a camera.\n'; exit 0 ;;
esac

pkill -f "$PORT:localhost:8000" 2>/dev/null || true                   # an old tunnel from a last run
echo "Copying the $folder camera server to the Pi ($PI). Password: mech5720"
# The Pi's old copy is removed first so stale read-only folders can't block the unpack (the Pi keeps nothing of its own there).
COPYFILE_DISABLE=1 tar -cf - --exclude __pycache__ -C software/data_capture "$folder" coded \
    | ssh "${SSH_OPTS[@]}" "$PI" "chmod -R u+w ~/$folder ~/coded 2>/dev/null; rm -rf ~/$folder ~/coded && tar -xf - -C ~ --warning=no-timestamp --warning=no-unknown-keyword && chmod -R u+w ~/$folder ~/coded" \
    || fail "could not reach the Pi: is it on, plugged into this computer by ethernet, and is $PI right?"

echo "Starting the camera server on the Pi. Password again: mech5720"
# -f: background after the password; -tt: the server gets a terminal, so it stops when the tunnel closes.
# Any server left from an earlier run is stopped first. The pattern is anchored to the python path at the start,
# so it can't match the Pi's own shell running this command line (which also contains "stream.py").
# Then the first Python with flask and picamera2 runs it: the system one, or a venv in the home folder (e.g. ~/mech5720).
ssh "${SSH_OPTS[@]}" -f -tt -o ExitOnForwardFailure=yes -L "$PORT:localhost:8000" "$PI" \
    "pkill -f '^[^ ]*python3? .*stream[.]py'; sleep 1
     for p in python3 ~/*/bin/python3; do \$p -c 'import flask, picamera2' 2>/dev/null && break; p=; done
     cd ~/$folder && exec \${p:-python3} stream.py" \
    || fail "could not start the camera server on the Pi"

printf 'Waiting for the camera'
for _ in $(seq 60); do
    curl -s --max-time 5 -o /dev/null "http://localhost:$PORT/" && break
    printf '.'; sleep 1
done
curl -s --max-time 5 -o /dev/null "http://localhost:$PORT/" \
    || fail "the camera server did not come up in 60 s: is the $folder camera the one plugged into the Pi?"
printf ' up\n\nOpening the camera GUI. Captures go to output/. Close its window when done.\n'

"${GUI[@]}" python "software/data_capture/$folder/camview_capture.py" --host localhost --port "$PORT" || true
pkill -f "$PORT:localhost:8000" 2>/dev/null || true                   # closes the tunnel, which stops the server
echo "Camera closed."
