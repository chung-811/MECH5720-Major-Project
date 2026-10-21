"""
Pi-side coded (flutter shutter) capture, used by rolling/stream.py and
global/stream.py through their POST /capture_coded endpoint.

One coded capture is a single long exposure that covers the whole light code
shown on the Pi's monitor by coded_display.py:

    1. the caller stops the stream; we configure a full-resolution still
       (raw + processed main stream) with AE off and
       ExposureTime = FrameDurationLimits = T
    2. coded_display.py opens fullscreen black and says READY
    3. wait for a frame to finish (the next one starts now), start grabbing
       that next frame, and tell the display GO
    4. the grabbed frame integrates every bit; it is returned as DNG + PNG

Exposure time:
    T = total_duration_s + readout_s + padding_s

The camera delivers a frame once it has been read out, so when the code
starts the next exposure has already been running for about one readout
time. On a rolling shutter (OV5647) that is also the delay between the top
and bottom rows starting. Adding readout_s means every row (rolling) or the
whole frame (global) sees the full code, with black before and after it.
padding_s absorbs scheduling jitter. Keep the room dark: anything lit
during the padding adds to the image.
"""

import io
import os
import sys
import json
import time
import queue
import tempfile
import threading
import subprocess

DISPLAY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coded_display.py")
CAMERA_BUFFER_COUNT = 2         # small on purpose: shared GPU/display memory
DEFAULT_PADDING_S = 0.05
# Set by the coded capture itself, so ignored if the GUI has set them.
FORCED_CONTROLS = ("AeEnable", "ExposureTime", "FrameDurationLimits")


class CodedAborted(Exception):
    """The run was stopped from the illumination screen (ESC)."""


# ======================================================================
# Sensor information
# ======================================================================

def sensor_info(picam2):
    """Describe the full-resolution sensor mode used for coded capture.

    Call this before the camera is started: probing sensor modes
    reconfigures the camera. readout_s is estimated as 1 / max fps of the
    mode, which slightly over-estimates the true readout (it includes the
    minimum blanking) - that is the safe direction for padding.
    """
    info = {
        "model": picam2.camera_properties.get("Model"),
        "sensor_size": list(picam2.sensor_resolution),
        "fps": None,
        "readout_s": None,
        "max_exposure_us": None,
    }
    try:
        modes = picam2.sensor_modes
        mode = max(modes, key=lambda m: m["size"][0] * m["size"][1])
        info["mode_size"] = list(mode["size"])
        fps = mode.get("fps")
        if fps:
            info["fps"] = float(fps)
            info["readout_s"] = 1.0 / float(fps)
        limits = mode.get("exposure_limits") or ()
        if len(limits) > 1 and limits[1]:
            info["max_exposure_us"] = int(limits[1])
    except Exception as err:
        info["error"] = "could not read sensor modes: %s" % err
    return info


# ======================================================================
# Display subprocess
# ======================================================================

class DisplayProcess:
    """Runs coded_display.py and talks to it over stdin/stdout lines."""

    def __init__(self, code):
        job = {k: code[k] for k in ("bits", "bit_duration_s", "pre_roll_s", "post_roll_s")}
        fd, self.job_path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(job, f)

        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")         # the Pi's own monitor
        self.proc = subprocess.Popen(
            [sys.executable, DISPLAY_SCRIPT, self.job_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1, env=env)

        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)                     # process exited

    def _expect(self, prefix, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("the illumination display did not send %s in time" % prefix)
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                self.proc.wait()
                raise RuntimeError(
                    "the illumination display exited early (code %s). Check the "
                    "server terminal; is a monitor attached, DISPLAY right and "
                    "python3-tk installed?" % self.proc.returncode)
            if line.startswith(prefix):
                return line[len(prefix):].strip()

    def _send(self, cmd):
        try:
            self.proc.stdin.write(cmd + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def wait_ready(self, timeout):
        self._expect("READY", timeout)

    def go(self):
        self._send("GO")

    def abort(self):
        self._send("ABORT")

    def result(self, timeout):
        return json.loads(self._expect("RESULT", timeout))

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        try:
            os.remove(self.job_path)
        except OSError:
            pass


# ======================================================================
# Camera timing
# ======================================================================

def sync_to_frame_boundary(cam, exposure_us, max_tries=8):
    """Wait for a frame to finish, so the NEXT frame starts right now.

    The camera free-runs with back-to-back frames of length ~exposure, so
    the instant a frame is delivered is (to within a few ms) the instant the
    next exposure begins. Starting the sequence at that instant makes the
    next frame cover the whole sequence.

    Also checks the exposure the camera actually used. If it settles below
    the request, the camera is clamping the exposure, and this raises before
    anything is flashed.
    """
    frame_s = exposure_us / 1e6
    prev = None
    for _ in range(max_tries):
        t = time.perf_counter()
        req = cam.capture_request()
        try:
            got = req.get_metadata().get("ExposureTime", 0)
        finally:
            req.release()
        blocked = time.perf_counter() - t

        if got >= 0.98 * exposure_us:
            if blocked >= 0.5 * frame_s:      # a live boundary, not a stale queued frame
                return got
        elif prev is not None and got == prev:
            raise RuntimeError(
                "the camera is limiting the exposure to %.3f s but %.3f s was "
                "requested. Shorten total_duration_s or the padding."
                % (got / 1e6, frame_s))
        prev = got
    raise RuntimeError("could not synchronise to the camera's frame boundary")


def start_frame_grabber(cam):
    """Collect the next completed frame on a background thread.

    Returns (thread, box); box gets "raw" (DNG-ready buffer), "image" (the
    ISP-processed main stream as a PIL image) and "metadata", or "error".
    Both come from the same request, so they are the same exposure.
    """
    box = {}

    def run():
        try:
            req = cam.capture_request()
            try:
                box["raw"] = req.make_buffer("raw")
                box["image"] = req.make_image("main")
                box["metadata"] = req.get_metadata()
            finally:
                req.release()
        except Exception as err:
            box["error"] = str(err)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th, box


# ======================================================================
# One coded capture
# ======================================================================

def _configure_still(picam2, controls):
    """Full-resolution raw + processed main stream. Falls back to a
    half-size main stream if there is not enough memory for full size."""
    size = tuple(picam2.sensor_resolution)
    for main_size in (size, (size[0] // 2 & ~1, size[1] // 2 & ~1)):
        try:
            cfg = picam2.create_still_configuration(
                main={"format": "RGB888", "size": main_size},
                raw={"size": size},
                buffer_count=CAMERA_BUFFER_COUNT,
                controls=controls)
            picam2.configure(cfg)
            return list(main_size)
        except Exception as err:
            last_err = err
    raise RuntimeError("could not configure the camera for coded capture: %s" % last_err)


def run_coded_capture(picam2, code, camera, info, shutter, manual_controls):
    """Run one coded capture. The camera must already be stopped; it is left
    stopped, and the caller restores its streaming configuration.

    code             output of coded_params.parse_code_params
    camera           GUI settings: {"padding_s": ..., "readout_s": ...}
    info             output of sensor_info (readout_s default, max exposure)
    shutter          "rolling" or "global" (recorded in the log)
    manual_controls  the stream's current controls (gain, colour gains, ...)

    Returns (log, dng_bytes, png_bytes). Raises CodedAborted or RuntimeError.
    """
    padding_s = float(camera.get("padding_s", DEFAULT_PADDING_S))
    readout_s = camera.get("readout_s")
    readout_s = float(info.get("readout_s") or 0.0) if readout_s is None else float(readout_s)
    if padding_s < 0 or readout_s < 0:
        raise ValueError("padding_s and readout_s must be >= 0")

    exposure_s = code["total_duration_s"] + readout_s + padding_s
    exposure_us = int(round(exposure_s * 1e6))
    if info.get("max_exposure_us") and exposure_us > info["max_exposure_us"]:
        raise RuntimeError("an exposure of %.3f s is longer than this camera's maximum (%.3f s). "
                           "Shorten total_duration_s or the padding."
                           % (exposure_s, info["max_exposure_us"] / 1e6))

    controls = {k: v for k, v in manual_controls.items() if k not in FORCED_CONTROLS}
    controls.setdefault("AnalogueGain", 1.0)
    controls.update({"AeEnable": False,
                     "ExposureTime": exposure_us,
                     "FrameDurationLimits": (exposure_us, exposure_us)})

    main_size = _configure_still(picam2, controls)
    raw_config = picam2.camera_configuration()["raw"]
    picam2.start()

    log = {}
    grabber = None
    try:
        display = DisplayProcess(code)
        try:
            display.wait_ready(timeout=code["pre_roll_s"] + 20)
            print("Coded capture: waiting for a camera frame boundary...")
            sync_exposure_us = sync_to_frame_boundary(picam2, exposure_us)
            boundary_ns = time.monotonic_ns()
            grabber = start_frame_grabber(picam2)  # this frame will cover the code
            go_ns = time.monotonic_ns()
            display.go()
            print("Coded capture: playing %d bits over %.3f s"
                  % (len(code["bits"]), code["total_duration_s"]))
            log = display.result(timeout=code["total_duration_s"] + code["post_roll_s"] + 20)
        except Exception:
            display.abort()
            raise
        finally:
            display.close()

        th, box = grabber
        th.join(timeout=exposure_s + 30)           # exposure ends ~now; allow readout
    finally:
        picam2.stop()

    if log.get("aborted"):
        raise CodedAborted("aborted from the illumination screen")
    if "raw" not in box:
        raise RuntimeError("no frame was captured: %s" % box.get("error", "timed out"))

    meta = box["metadata"]
    fd, dng_path = tempfile.mkstemp(suffix=".dng")
    os.close(fd)
    try:
        picam2.helpers.save_dng(box["raw"], meta, raw_config, dng_path)
        with open(dng_path, "rb") as f:
            dng_bytes = f.read()
    finally:
        os.remove(dng_path)

    png = io.BytesIO()
    box["image"].save(png, format="PNG")

    log.update({
        "camera_model": info.get("model"),
        "shutter": shutter,
        "sensor_size": info.get("sensor_size"),
        "png_size": main_size,
        "code": {k: v for k, v in code.items() if k != "warnings"},
        "warnings": code["warnings"],
        "readout_s": readout_s,
        "padding_s": padding_s,
        "requested_exposure_us": exposure_us,
        "sync_exposure_us": sync_exposure_us,
        "actual_exposure_us": int(meta.get("ExposureTime", 0)),
        "actual_gain": float(meta.get("AnalogueGain", 0)),
        "colour_gains": list(meta.get("ColourGains", [])) or None,
        "sensor_timestamp_ns": meta.get("SensorTimestamp"),
        "frame_boundary_monotonic_ns": boundary_ns,
        "go_sent_monotonic_ns": go_ns,
        "controls": {k: (list(v) if isinstance(v, tuple) else v) for k, v in controls.items()},
        "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    return log, dng_bytes, png.getvalue()


# ======================================================================
# Results waiting to be downloaded
# ======================================================================

_results = {}
_results_lock = threading.Lock()
RESULT_TTL_S = 300


def store_result(dng_bytes, png_bytes):
    """Keep one capture's files until the client downloads them. Returns an id."""
    rid = "%d" % time.time_ns()
    with _results_lock:
        now = time.time()
        for k in [k for k, v in _results.items() if now - v["t"] > RESULT_TTL_S]:
            del _results[k]
        _results[rid] = {"t": now, "dng": dng_bytes, "png": png_bytes}
    return rid


def fetch_result(rid, kind):
    """Return the 'dng' or 'png' bytes for a result (or None); the entry is
    dropped once both have been fetched."""
    with _results_lock:
        entry = _results.get(rid)
        if entry is None or entry.get(kind) is None:
            return None
        data = entry[kind]
        entry[kind] = None
        if entry["dng"] is None and entry["png"] is None:
            del _results[rid]
        return data
