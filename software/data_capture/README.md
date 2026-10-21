# Data capture

Camera server/client code for capturing images from the Raspberry Pi, split by camera type:

```
software/data_capture/
├── rolling/              # Rolling shutter camera (OV5647)
│   ├── stream.py         # server, runs on the Pi
│   └── camview_capture.py  # client, runs on the development computer
└── global/               # Global Shutter camera (Sony IMX296)
    ├── stream.py
    └── camview_capture.py
```

Each folder has the same pair of scripts: `stream.py` runs **on the Pi** and serves the camera over HTTP; `camview_capture.py` runs **on the development computer** and shows the live feed with controls for capturing frames. Captured DNGs can be viewed with [software/view_raw.py](../view_raw.py).

All commands below are run from the repo root (`MECH5720-Major-Project/`) unless stated otherwise.

## Environment setup (development computer)

The lab handout suggests creating the venv with `python -m venv --system-site-packages mech5720`, but the `--system-site-packages` flag caused issues. A plain `.venv` at the repo root, installed from [requirements.txt](../../requirements.txt), works better:

```
python -m venv .venv
./.venv/Scripts/Activate.ps1   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

To activate the environment in later sessions:

```
./.venv/Scripts/Activate.ps1   # macOS/Linux: source .venv/bin/activate
```

Quick check that everything installed correctly:

```
python -c "import numpy, skimage, rawpy, matplotlib, imgui_bundle; print('Environment OK')"
```

## Pi setup

The Pi needs `picamera2` and `flask` (usually already installed on the lab image):

```
sudo apt install -y python3-picamera2
pip install flask --break-system-packages
```

Username: `pi`, password: `mech5720`. Our Pi is `eb5234f9picam.local` (in general, `<ID_STRING>.local`).

Copy both server folders to the Pi. Run this on the development computer:

```
scp -r software/data_capture/rolling software/data_capture/global pi@eb5234f9picam.local:~/
```

This puts them at `~/rolling/` and `~/global/` on the Pi. Re-run it whenever you change either `stream.py`.

## Capturing images

Capturing works the same way for both cameras; only the folder changes. Make sure the right camera is plugged into the Pi.

**1. SSH into the Pi with port forwarding.** Connect to the Pi via ethernet (or WiFi), then in a new terminal on the development computer run the following. Leave it open while you're streaming:

```
ssh -L 8089:localhost:8000 pi@eb5234f9picam.local
```

**2. Start the server on the Pi.** In the SSH session from step 1:

Rolling shutter:

```
cd ~/rolling
python stream.py               # serves on port 8000
```

Global shutter:

```
rpicam-hello --list-cameras    # should list imx296 at 1456x1088
cd ~/global
python stream.py               # serves on port 8000 (change with --port)
```

For the global shutter camera, the startup log should show `Camera model: imx296` and a raw stream size of `(1456, 1088)`. Stop the server with `Ctrl+C`.

**3. Start the client.** In a second terminal on the development computer, `cd` into the folder where you want captures saved (files are written to the current working directory), activate the venv, and run the matching client:

```
# Rolling shutter
python software/data_capture/rolling/camview_capture.py --host localhost --port 8089

# Global shutter
python software/data_capture/global/camview_capture.py --host localhost --port 8089
```

(Adjust the script path if you've `cd`'d somewhere other than the repo root.) As a fallback, you can view the stream in a browser at `localhost:8089`.

**4. Capture.** Use **Capture DNG** in the Stream Settings window to save a raw frame (see [controls](#camview_capturepy-controls) below).

**5. View a captured DNG.** With the venv active:

```
python software/view_raw.py captured_frame_<timestamp>.dng       # rolling
python software/view_raw.py gs_captured_frame_<timestamp>.dng    # global
```

## `camview_capture.py` controls

The client opens two windows: a **Pi Camera Feed** view and a **Stream Settings** control panel. Settings in the panel are:

- **Camera Settings** (sent to the Pi, changes what the sensor actually captures):
  - **Resolution** — rolling: `640x480`, `1280x720`, `1920x1080` (or whatever the server reports at connect time); global: see [below](#global-shutter-differences).
  - **Format** — `RGB888`, `BGR888`, `RGB565`, `YUV420`.
  - Changing either posts to the Pi's `/config` endpoint and restarts the stream at the new setting.

- **Display Processing** (local only, doesn't affect what's captured/saved on the Pi):
  - **Color Order** — `RGB` (correct colours) or `BGR` (channels swapped, useful for spotting a wrong RGB/BGR assumption).
  - **Rotation** — rotate the displayed feed 90° left/right, or reset to 0°.

- **Advanced Camera Controls** (collapsible section): dynamically fetched from the Pi's `/controls` endpoint (e.g. exposure, gain, white balance — whatever `stream.py` exposes). Renders as checkboxes (bool), sliders (numeric with a small range), or numeric input boxes (numeric with a large/no range). Click **Refresh Controls** to re-fetch the current values from the Pi. Edits are pushed back to the Pi as you change them.

- **Actions**:
  - **Capture Debug Data** — downloads one raw sensor frame (via `/capture_raw`) and saves it as `captured_frame_<timestamp>.raw` (plain bytes, shape/dtype printed to the console).
  - **Capture DNG** — downloads one raw frame as a proper `.dng` file (via `/capture_dng`), saved as `captured_frame_<timestamp>.dng`; openable directly in `rawpy`, Lightroom, darktable, etc.
  - **Save Current** — saves the currently displayed (post color-order/rotation) frame as a PNG, `frame_<timestamp>.png`.

- **Status bar**: shows stream connection status (Connected/Disconnected), the last activity/error message, and UI vs. stream FPS.

### Global shutter differences

The `global/` scripts are adapted from the rolling shutter ones for the Raspberry Pi Global Shutter camera (IMX296):

- The raw stream is locked to the sensor's single native mode (1456x1088), so DNGs are always full resolution. The streamed output defaults to 728x544; the resolution menu offers `728x544`, `1456x1088` and `640x480` (sizes larger than the sensor are rejected by the server).
- The lens is focused by hand. The control panel shows a **Focus score** (variance of Laplacian) with its peak; turn the focus ring to maximise it. **Reset Peak** restarts the peak tracking.
- Saved files are prefixed `gs_` (`gs_captured_frame_<timestamp>.dng`, `gs_frame_<timestamp>.png`, ...) so they can't be confused with rolling shutter captures. `view_raw.py` opens them unchanged.
