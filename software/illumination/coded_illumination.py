#!/usr/bin/env python3
"""
coded_illumination.py

Flash the whole screen black / white in a binary sequence (coded
illumination). Every parameter comes from a JSON file:

    python coded_illumination.py params1.json

Over SSH, tell it which display to draw on:

    DISPLAY=:0 python3 coded_illumination.py params1.json

Parameters (all optional except "sequence" and "total_duration_s")
------------------------------------------------------------------
    sequence            the code. A string such as "1011001110" (spaces,
                        commas and underscores are ignored) or a list
                        [1,0,1,1,...]. 1 = white, 0 = black.      [required]
    total_duration_s    how long the whole run of bits is displayed, in
                        seconds, counting every repeat. Each bit is shown
                        for total_duration_s / (number of bits).   [required]
    repeat              how many times the sequence is played.          [1]
    invert              true = swap black and white.                [false]
    pre_roll_s          seconds of black before the sequence.         [1.0]
    post_roll_s         seconds of black after the sequence.          [1.0]
    end_color           "black" or "white", held after the run.   ["black"]
    wait_for_start      true = show a prompt and wait for ENTER (on the
                        screen, or in the SSH terminal; 'q' + ENTER
                        aborts). false = start immediately.          [true]

    capture             true = take ONE raw photo whose exposure covers the
                        whole sequence. false = display only.       [false]
    out_path            folder for the DNG and its run log.  ["data/coded"]
    analogue_gain       fixed analogue gain. Keep it low: the exposure is as
                        long as the whole sequence.                   [1.0]
    exposure_padding_s  extra exposure added on top of total_duration_s, to
                        absorb small start/stop sync errors.          [0.0]

Camera exposure
---------------
With capture on, the exposure time is set automatically to
total_duration_s + exposure_padding_s, so the single frame integrates every
flash in the sequence. The camera free-runs, so the script first waits for a
frame boundary, then starts the sequence straight away; the exposure begins
within a few milliseconds of the first bit. If the camera cannot do an
exposure that long, the script stops before flashing anything and says what
the camera actually delivered.

Timing notes
------------
Bit changes are scheduled against one start time (t0 + i * duration), so
errors do not accumulate. The screen only changes on a display refresh, so
the real resolution is one refresh period (about 16.7 ms at 60 Hz).

Press ESC on the illumination screen, or Ctrl-C in the terminal, to stop a
run early.

SAFETY: rapid black/white flashing (more than about 3 flashes per second) can
trigger seizures in people with photosensitive epilepsy. Warn anyone who may
be in the room and do not use the screen where people might look at it.
"""

import os
import sys
import json
import time
import queue
import argparse
import threading

import tkinter as tk


# ======================================================================
# Parameters
# ======================================================================

DEFAULTS = {
    "sequence": None,
    "total_duration_s": None,
    "repeat": 1,
    "invert": False,
    "pre_roll_s": 1.0,
    "post_roll_s": 1.0,
    "end_color": "black",
    "wait_for_start": True,
    "capture": False,
    "out_path": "data/coded",
    "analogue_gain": 1.0,
    "exposure_padding_s": 0.0,
}

WARN_FLASH_HZ = 3.0        # photosensitivity warning threshold
REFRESH_S = 1.0 / 60.0     # assumed display refresh period
CAMERA_BUFFER_COUNT = 2    # small on purpose: shared GPU/display memory


def parse_sequence(seq):
    """Turn a '1011...' string or a [1, 0, 1, ...] list into a list of ints."""
    if seq is None:
        raise ValueError('"sequence" is required')

    if isinstance(seq, str):
        items = [c for c in seq if c not in " ,_\t\r\n"]
    elif isinstance(seq, (list, tuple)):
        items = list(seq)
    else:
        raise ValueError('"sequence" must be a string like "1011" or a list of 0/1')

    bits = []
    for item in items:
        if item in (0, 1, False, True, "0", "1"):
            bits.append(int(item))
        else:
            raise ValueError('"sequence" may only contain 0 and 1 (found %r)' % (item,))
    if not bits:
        raise ValueError('"sequence" is empty')
    return bits


def load_params(path):
    """Read the JSON file, apply defaults, validate, return a dict."""
    with open(path, "r") as f:
        user = json.load(f)
    if not isinstance(user, dict):
        raise ValueError("the JSON file must contain an object { ... }")

    unknown = sorted(set(user) - set(DEFAULTS))
    if unknown:
        print("Warning: ignoring unknown parameter(s): %s" % ", ".join(unknown))

    p = dict(DEFAULTS)
    p.update({k: v for k, v in user.items() if k in DEFAULTS})

    p["bits"] = parse_sequence(p["sequence"])

    def need(name, kind, low, strict=False):
        v = p[name]
        if v is None:
            raise ValueError('"%s" is required' % name)
        if isinstance(v, bool) or not isinstance(v, kind):
            raise ValueError('"%s" must be a number' % name)
        if v < low or (strict and v == low):
            raise ValueError('"%s" must be %s %s' % (name, ">" if strict else ">=", low))

    need("total_duration_s", (int, float), 0, strict=True)
    need("repeat", int, 1)
    need("pre_roll_s", (int, float), 0)
    need("post_roll_s", (int, float), 0)
    need("analogue_gain", (int, float), 0, strict=True)
    need("exposure_padding_s", (int, float), 0)

    if str(p["end_color"]).lower() not in ("black", "white"):
        raise ValueError('"end_color" must be "black" or "white"')
    p["end_color"] = str(p["end_color"]).lower()

    # Derived values.
    p["n_bits"] = len(p["bits"]) * p["repeat"]
    p["bit_duration_s"] = float(p["total_duration_s"]) / p["n_bits"]
    p["exposure_us"] = int(round((p["total_duration_s"] + p["exposure_padding_s"]) * 1e6))

    return p


# ======================================================================
# Illumination display
# ======================================================================

class LightStage:
    """A fullscreen window that is entirely black or entirely white."""

    def __init__(self):
        self.aborted = False

        self.root = tk.Tk()
        self.root.title("Coded illumination")
        self.root.configure(bg="black")
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        self.root.geometry("%dx%d+0+0" % (w, h))
        self.root.attributes("-fullscreen", True)
        self.root.config(cursor="none")

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.root.update()

        # Message text, hidden during the run: anything drawn here is light
        # falling on the scene, so it must be off while the code plays.
        self.text = self.canvas.create_text(
            self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
            text="", fill="#808080", font=("DejaVu Sans", 28),
            justify="center", state="hidden")

        self.root.focus_force()

    def show(self, on):
        """Fill the whole screen white (truthy) or black (falsy)."""
        self.canvas.configure(bg="white" if on else "black")
        self.root.update()

    def message(self, text):
        self.canvas.configure(bg="black")
        self.canvas.itemconfig(self.text, text=text, state="normal")
        self.canvas.tag_raise(self.text)
        self.root.update()

    def clear_message(self):
        self.canvas.itemconfig(self.text, state="hidden")
        self.root.update()

    def enable_escape_abort(self):
        def _abort(_event=None):
            self.aborted = True
        self.root.bind("<Escape>", _abort)

    def wait_for_key(self, prompt):
        """Block until ENTER is pressed on the screen or in the terminal.

        Returns False if aborted (ESC on screen, or 'q' + ENTER in terminal).
        The terminal is read on a daemon thread and handed to Tk through a
        queue, because Tk objects must only be touched from the Tk thread.
        """
        state = {"go": False}
        done = {"flag": False}
        stdin_q = queue.Queue()

        def start(_event=None):
            if done["flag"]:
                return
            done["flag"] = True
            state["go"] = True
            self.root.quit()

        def abort(_event=None):
            if done["flag"]:
                return
            done["flag"] = True
            self.root.quit()

        def read_stdin():
            try:
                for line in sys.stdin:
                    stdin_q.put(line.strip())
                    if done["flag"]:
                        return
            except Exception:
                pass

        def poll_stdin():
            if done["flag"]:
                return
            try:
                line = stdin_q.get_nowait()
            except queue.Empty:
                self.root.after(100, poll_stdin)
                return
            if line.lower() in ("q", "esc", "abort", "escape"):
                abort()
            else:
                start()

        self.message(prompt)
        self.root.bind("<Return>", start)
        self.root.bind("<KP_Enter>", start)
        self.root.bind("<Escape>", abort)
        self.root.focus_force()

        threading.Thread(target=read_stdin, daemon=True).start()
        self.root.after(100, poll_stdin)

        self.root.mainloop()
        self.root.unbind("<Return>")
        self.root.unbind("<KP_Enter>")
        self.root.unbind("<Escape>")
        self.clear_message()
        return state["go"]

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# ======================================================================
# Camera (optional)
# ======================================================================

def open_camera(exposure_us, gain):
    """Open the Pi camera for raw capture with a fixed, long exposure.

    Auto-exposure and auto-white-balance are off. FrameDurationLimits is set
    equal to the exposure, because a frame cannot be shorter than its
    exposure, and the default limits would cap the exposure far below the
    length of a typical sequence.

    Returns None if no camera is available (the code is then just displayed).
    """
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
    except Exception as err:
        print("\n*** No camera available: %s" % err)
        print("*** The code will be displayed, but nothing will be saved.\n")
        return None

    cfg = cam.create_still_configuration(
        raw={"format": "SBGGR10", "size": cam.sensor_resolution},
        buffer_count=CAMERA_BUFFER_COUNT,
        controls={
            "AeEnable": False,
            "AwbEnable": False,
            "AnalogueGain": float(gain),
            "ExposureTime": int(exposure_us),
            "FrameDurationLimits": (int(exposure_us), int(exposure_us)),
        })
    cam.configure(cfg)
    cam.start()
    return cam


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
                "requested. Shorten total_duration_s (and exposure_padding_s)."
                % (got / 1e6, frame_s))
        prev = got
    raise RuntimeError("could not synchronise to the camera's frame boundary")


def start_frame_grabber(cam):
    """Collect the next completed frame on a background thread.

    Returns (thread, box); box gets "array" and "metadata", or "error".
    The request is released as soon as the pixels are copied out.
    """
    box = {}

    def run():
        try:
            req = cam.capture_request()
            try:
                box["array"] = req.make_array("raw").copy()
                box["metadata"] = req.get_metadata()
            finally:
                req.release()
        except Exception as err:
            box["error"] = str(err)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    return th, box


# ======================================================================
# Playback
# ======================================================================

def wait_until(stage, t):
    """Wait until perf_counter() reaches t, keeping Tk responsive (for ESC)."""
    while True:
        remaining = t - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 0.006:
            stage.root.update()
            time.sleep(min(remaining - 0.005, 0.02))
        else:
            time.sleep(0)    # final stretch: spin for accuracy


def play(stage, cam, params):
    """Display the sequence, recording one long exposure if a camera is open.

    Returns (log, frame) where frame is {"array", "metadata"} or None.
    """
    bits = params["bits"] * params["repeat"]
    if params["invert"]:
        bits = [1 - b for b in bits]
    dur = params["bit_duration_s"]
    n = len(bits)

    log = {"bits": [], "late_bits": 0}
    grabber = None

    stage.enable_escape_abort()
    stage.show(0)
    time.sleep(params["pre_roll_s"])

    if cam is not None:
        print("  waiting for a camera frame boundary...")
        used = sync_to_frame_boundary(cam, params["exposure_us"])
        log["sync_exposure_us"] = used
        grabber = start_frame_grabber(cam)      # this frame will cover the sequence

    log["wall_clock_start"] = time.time()
    t0 = time.perf_counter() + 0.02              # small lead so bit 0 is not late

    for i, bit in enumerate(bits):
        planned = t0 + i * dur
        wait_until(stage, planned)
        if stage.aborted:
            print("  aborted from the illumination screen.")
            break

        t_show = time.perf_counter()
        stage.show(bit)
        late = t_show - planned
        if late > 0.010:
            log["late_bits"] += 1
        log["bits"].append({"index": i, "value": bit,
                            "shown_at_s": round(t_show - t0, 4),
                            "late_s": round(max(late, 0.0), 4)})

        if (i + 1) % 10 == 0 or i + 1 == n:
            print("  %4d / %4d bits shown  (%.1f s elapsed)"
                  % (i + 1, n, time.perf_counter() - t0))
    else:
        wait_until(stage, t0 + n * dur)         # last bit gets its full duration

    stage.show(0)
    log["aborted"] = stage.aborted

    frame = None
    if grabber is not None and not stage.aborted:
        th, box = grabber
        th.join(timeout=30.0)                    # exposure ends ~now; allow readout
        if "array" in box:
            frame = box
        else:
            print("\n*** No frame was captured: %s" % box.get("error", "timed out"))

    time.sleep(params["post_roll_s"])
    return log, frame


def save_run(cam, params, log, frame):
    """Write the DNG and a JSON log describing the run."""
    os.makedirs(params["out_path"], exist_ok=True)
    stem = time.strftime("coded_%Y%m%d_%H%M%S")
    dng_path = os.path.join(params["out_path"], stem + ".dng")
    log_path = os.path.join(params["out_path"], stem + ".json")

    cam.helpers.save_dng(frame["array"], frame["metadata"],
                         cam.camera_configuration()["raw"], dng_path)

    meta = frame["metadata"]
    log["actual_exposure_us"] = int(meta.get("ExposureTime", 0))
    log["actual_gain"] = float(meta.get("AnalogueGain", 0))
    log["params"] = {k: v for k, v in params.items() if k != "bits"}
    log["sequence_played"] = [b["value"] for b in log["bits"]]
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    return dng_path, log_path


# ======================================================================
# Main
# ======================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Flash the screen black/white in a binary sequence "
                    "defined by a JSON parameter file.")
    ap.add_argument("params", help="JSON parameter file, e.g. params1.json")
    args = ap.parse_args()

    try:
        params = load_params(args.params)
    except (OSError, ValueError) as err:
        print("Error in %s: %s" % (args.params, err))
        return 2

    bits = params["bits"] * params["repeat"]
    dur = params["bit_duration_s"]
    total_s = params["total_duration_s"]
    shown = [1 - b for b in bits] if params["invert"] else bits

    print("Parameter file   %s" % args.params)
    print("Sequence         %s%s" % ("".join(map(str, params["bits"])),
                                      "  x%d" % params["repeat"] if params["repeat"] > 1 else ""))
    print("Bits shown       %d%s" % (len(bits), "  (inverted)" if params["invert"] else ""))
    print("Total duration   %.3f s  ->  %.4f s per bit (%.2f bits/s)"
          % (total_s, dur, 1.0 / dur))
    print("Pre/post roll    %.1f s / %.1f s" % (params["pre_roll_s"], params["post_roll_s"]))
    if params["capture"]:
        print("Camera           one exposure of %.3f s, gain %.1f -> %s"
              % (params["exposure_us"] / 1e6, params["analogue_gain"], params["out_path"]))
    else:
        print("Camera           off")

    if dur < REFRESH_S:
        print("\n*** WARNING: %.1f ms per bit is shorter than one refresh of a 60 Hz"
              % (dur * 1000))
        print("*** screen (%.1f ms). Bits will be dropped. Lengthen total_duration_s."
              % (REFRESH_S * 1000))

    flashes = sum(1 for a, b in zip([0] + shown, shown) if b == 1 and a == 0)
    if flashes and flashes / total_s > WARN_FLASH_HZ:
        print("\n*** WARNING: about %.1f white flashes per second on average."
              % (flashes / total_s))
        print("*** Rapid flashing can trigger photosensitive seizures. Make sure")
        print("*** nobody who may be affected can see the screen.")

    cam = open_camera(params["exposure_us"], params["analogue_gain"]) if params["capture"] else None

    stage = LightStage()
    log, frame = None, None
    try:
        if params["wait_for_start"]:
            print("\nCover the rig, then press ENTER here in the terminal")
            print("(or on the illumination screen) to start. Type 'q' + ENTER to abort.")
            ok = stage.wait_for_key(
                "Press ENTER -- on this screen, or in your\n"
                "SSH terminal -- to start.\n\n"
                "%d bits, %.1f seconds.\n"
                "Type 'q' + ENTER in the terminal, or press ESC here, to abort."
                % (len(bits), total_s))
            if not ok:
                print("Aborted.")
                return 1

        print("\n--- running ---")
        log, frame = play(stage, cam, params)

        stage.show(1 if params["end_color"] == "white" else 0)
        time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    except RuntimeError as err:
        print("\n*** %s" % err)
        return 1
    finally:
        stage.close()
        if log is None and cam is not None:     # leaving early: release the camera
            cam.stop()

    try:
        if log["late_bits"]:
            print("\n*** %d bit(s) were shown more than 10 ms late." % log["late_bits"])

        if cam is not None and frame is not None:
            dng_path, log_path = save_run(cam, params, log, frame)
            got = int(frame["metadata"].get("ExposureTime", 0))
            print("\nExposure used: %.3f s (requested %.3f s)"
                  % (got / 1e6, params["exposure_us"] / 1e6))
            print("Wrote %s\n      %s" % (dng_path, log_path))
        elif params["capture"] and cam is None:
            print("\nNo camera, so nothing was saved.")
    finally:
        if cam is not None:
            cam.stop()

    print("Stopped early." if log["aborted"] else "Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
