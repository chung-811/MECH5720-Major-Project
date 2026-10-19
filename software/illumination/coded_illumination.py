#!/usr/bin/env python3
"""
coded_illumination.py

Flash the whole screen black / white in a binary sequence (coded
illumination). Every parameter comes from a JSON file:

    python coded_illumination.py params1.json

Over SSH, tell it which display to draw on:

    DISPLAY=:0 python3 coded_illumination.py params1.json

Parameters (all optional except "sequence"; defaults in brackets)
-----------------------------------------------------------------
    sequence          the code. A string such as "1011001110" (spaces, commas
                      and underscores are ignored) or a list [1,0,1,1,...].
                      1 = white, 0 = black.                        [required]
    bit_duration_s    how long each bit is displayed, in seconds.       [0.5]
    repeat            how many times the whole sequence is played.        [1]
    invert            true = swap black and white.                    [false]
    pre_roll_s        seconds of black before the sequence starts.      [1.0]
    post_roll_s       seconds of black after the sequence ends.         [1.0]
    end_color         "black" or "white", held after the run ends.  ["black"]
    wait_for_start    true = show a prompt and wait for ENTER (on the
                      screen, or in the SSH terminal; 'q' + ENTER
                      aborts). false = start immediately.             [true]

    capture           true = record raw frames from the Pi camera while the
                      sequence plays. false = display only.          [false]
    out_path          folder for the captured DNG files and run_log.json.
                                                           ["data/coded"]
    frames_per_bit    frames recorded for each bit (capture only).        [1]
    exposure_us       fixed exposure time in microseconds.          [20000]
    analogue_gain     fixed analogue gain.                            [16.0]
    settle_s          wait after each screen change before recording
                      (capture only).                                [0.05]
    discard_frames    frames thrown away after each change, in case the
                      camera still shows the previous bit.            [2]

Timing notes
------------
Bit changes are scheduled against a single start time (t0 + i * duration), so
timing errors do not accumulate. The screen itself can only change on a
display refresh, so the real resolution is one refresh period (about 16.7 ms
at 60 Hz): use durations that are a whole number of refreshes. If capture is
on, settle_s + frames_per_bit * (frame time) must fit inside bit_duration_s,
otherwise bits are shown late; the script counts and reports this.

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
    "bit_duration_s": 0.5,
    "repeat": 1,
    "invert": False,
    "pre_roll_s": 1.0,
    "post_roll_s": 1.0,
    "end_color": "black",
    "wait_for_start": True,
    "capture": False,
    "out_path": "data/coded",
    "frames_per_bit": 1,
    "exposure_us": 20000,
    "analogue_gain": 16.0,
    "settle_s": 0.05,
    "discard_frames": 2,
}

# Photosensitivity warning threshold: flashes (white pulses) per second.
WARN_FLASH_HZ = 3.0

# Camera buffers. Kept small on purpose: they come from a small pool of
# GPU/display-shared memory on the Pi, which the fullscreen window also needs.
CAMERA_BUFFER_COUNT = 2
WRITE_WORKERS = 3
WRITE_QUEUE_DEPTH = 4


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
        if isinstance(v, bool) or not isinstance(v, kind):
            raise ValueError('"%s" must be a number' % name)
        if v < low or (strict and v == low):
            raise ValueError('"%s" must be %s %s' % (name, ">" if strict else ">=", low))

    need("bit_duration_s", (int, float), 0, strict=True)
    need("repeat", int, 1)
    need("pre_roll_s", (int, float), 0)
    need("post_roll_s", (int, float), 0)
    need("frames_per_bit", int, 1)
    need("exposure_us", int, 1)
    need("analogue_gain", (int, float), 0, strict=True)
    need("settle_s", (int, float), 0)
    need("discard_frames", int, 0)

    if str(p["end_color"]).lower() not in ("black", "white"):
        raise ValueError('"end_color" must be "black" or "white"')
    p["end_color"] = str(p["end_color"]).lower()

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
    """Open the Pi camera for raw capture with everything held fixed.

    Auto-exposure and auto-white-balance are off: anything that adapts
    between frames breaks the linear relationship that decoding relies on.
    Returns None if no camera is available (the script then just displays
    the code).
    """
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
    except Exception as err:
        print("\n*** No camera available: %s" % err)
        print("*** The code will be displayed, but no frames will be saved.\n")
        return None

    cfg = cam.create_still_configuration(
        raw={"format": "SBGGR10", "size": cam.sensor_resolution},
        buffer_count=CAMERA_BUFFER_COUNT)
    cam.configure(cfg)
    cam.set_controls({
        "AeEnable": False,
        "AwbEnable": False,
        "AnalogueGain": float(gain),
        "ExposureTime": int(exposure_us),
    })
    cam.start()
    time.sleep(2.0)          # let the pipeline settle on the fixed values
    return cam


def grab_capture(cam, discard):
    """Capture one raw frame as (array, metadata), releasing the request
    straight away so camera buffers are never held during a slow disk write.
    """
    if cam is None:
        return None
    for _ in range(discard):
        cam.capture_array("raw")
    request = cam.capture_request()
    try:
        array = request.make_array("raw").copy()
        metadata = request.get_metadata()
    finally:
        request.release()
    return array, metadata


def save_frame(cam, array, metadata, raw_config, out_path, name, index):
    """Write one frame as a DNG (untouched sensor raw plus its metadata)."""
    cam.helpers.save_dng(
        array, metadata, raw_config,
        os.path.join(out_path, "%s%04d.dng" % (name, index)))


class AsyncFrameWriter:
    """Background DNG writer so the flashing never waits on disk I/O."""

    def __init__(self, cam, raw_config, out_path,
                 n_workers=WRITE_WORKERS, depth=WRITE_QUEUE_DEPTH):
        os.makedirs(out_path, exist_ok=True)
        self.cam = cam
        self.raw_config = raw_config
        self.out_path = out_path
        self.q = queue.Queue(maxsize=depth)
        self.errors = []
        self._lock = threading.Lock()
        self.workers = [threading.Thread(target=self._run, daemon=True)
                        for _ in range(n_workers)]
        for w in self.workers:
            w.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            array, metadata, name, index = item
            try:
                save_frame(self.cam, array, metadata, self.raw_config,
                           self.out_path, name, index)
            except Exception as err:
                with self._lock:
                    self.errors.append("%s%04d: %s" % (name, index, err))
            finally:
                self.q.task_done()

    def put(self, array, metadata, name, index):
        self.q.put((array, metadata, name, index))

    def pending(self):
        return self.q.qsize()

    def close(self):
        self.q.join()
        for _ in self.workers:
            self.q.put(None)
        for w in self.workers:
            w.join()
        if self.errors:
            print("\n  *** %d frame(s) failed to write:" % len(self.errors))
            for e in self.errors[:10]:
                print("      %s" % e)


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
    """Display the sequence (and record it, if a camera is open).

    Returns a log dict describing what was shown and when.
    """
    bits = params["bits"] * params["repeat"]
    if params["invert"]:
        bits = [1 - b for b in bits]
    dur = float(params["bit_duration_s"])
    n = len(bits)

    writer = None
    raw_config = None
    if cam is not None:
        raw_config = cam.camera_configuration()["raw"]
        writer = AsyncFrameWriter(cam, raw_config, params["out_path"])

    log = {"bits": [], "frames": [], "late_bits": 0}
    frame_no = 0

    stage.enable_escape_abort()
    stage.show(0)
    time.sleep(params["pre_roll_s"])

    wall_start = time.time()
    t0 = time.perf_counter() + 0.05          # small lead so bit 0 is not late
    log["wall_clock_start"] = wall_start

    try:
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

            if writer is not None:
                wait_until(stage, t_show + params["settle_s"])
                for _ in range(params["frames_per_bit"]):
                    t_cap = time.perf_counter()
                    captured = grab_capture(cam, params["discard_frames"])
                    array, metadata = captured
                    frame_no += 1
                    writer.put(array, metadata, "Coded", frame_no)
                    log["frames"].append({"frame": frame_no, "bit_index": i,
                                          "bit_value": bit,
                                          "captured_at_s": round(t_cap - t0, 4)})

            if (i + 1) % 10 == 0 or i + 1 == n:
                print("  %4d / %4d bits shown  (%.1f s elapsed)"
                      % (i + 1, n, time.perf_counter() - t0))

        else:
            # Let the final bit be displayed for its full duration.
            wait_until(stage, t0 + n * dur)

        stage.show(0)
        time.sleep(params["post_roll_s"])
    finally:
        if writer is not None:
            t1 = time.time()
            print("  flushing %d queued frame(s) to disk..." % writer.pending())
            writer.close()
            print("  writing finished in %.1f s" % (time.time() - t1))

    log["aborted"] = stage.aborted
    return log


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
    total_s = len(bits) * dur
    shown = [1 - b for b in bits] if params["invert"] else bits

    print("Parameter file   %s" % args.params)
    print("Sequence         %s%s" % ("".join(map(str, params["bits"])),
                                      "  x%d" % params["repeat"] if params["repeat"] > 1 else ""))
    print("Bits shown       %d%s" % (len(bits), "  (inverted)" if params["invert"] else ""))
    print("Bit duration     %.4f s  (%.2f bits/s)" % (dur, 1.0 / dur))
    print("Run time         %.1f s (+ %.1f s pre/post roll)"
          % (total_s, params["pre_roll_s"] + params["post_roll_s"]))
    print("Camera capture   %s" % ("on -> %s" % params["out_path"] if params["capture"] else "off"))

    # Worst-case flash rate: white pulses per second.
    flashes = sum(1 for a, b in zip([0] + shown, shown) if b == 1 and a == 0)
    if flashes and flashes / total_s > WARN_FLASH_HZ:
        print("\n*** WARNING: up to %.1f white flashes per second on average."
              % (flashes / total_s))
        print("*** Rapid flashing can trigger photosensitive seizures. Make sure")
        print("*** nobody who may be affected can see the screen.")

    cam = None
    if params["capture"]:
        cam = open_camera(params["exposure_us"], params["analogue_gain"])

    stage = LightStage()
    try:
        if params["wait_for_start"]:
            print("\nCover the rig, then press ENTER here in the terminal")
            print("(or on the illumination screen) to start. Type 'q' + ENTER to abort.")
            ok = stage.wait_for_key(
                "Press ENTER -- on this screen, or in your\n"
                "SSH terminal -- to start.\n\n"
                "%d bits, about %.0f seconds.\n"
                "Type 'q' + ENTER in the terminal, or press ESC here, to abort."
                % (len(bits), total_s))
            if not ok:
                print("Aborted.")
                return 1

        print("\n--- running ---")
        log = play(stage, cam, params)

        stage.show(1 if params["end_color"] == "white" else 0)
        time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 1
    finally:
        stage.close()
        if cam is not None:
            cam.stop()

    if log["late_bits"]:
        print("\n*** %d bit(s) were shown more than 10 ms late. Increase"
              % log["late_bits"])
        print("*** bit_duration_s, or reduce frames_per_bit / discard_frames.")

    if cam is not None:
        log["params"] = {k: v for k, v in params.items() if k != "bits"}
        log["sequence_played"] = shown
        os.makedirs(params["out_path"], exist_ok=True)
        log_path = os.path.join(params["out_path"], "run_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        print("\nWrote %d frame(s) and %s" % (len(log["frames"]), log_path))
    elif params["capture"]:
        print("\nNo camera, so nothing was saved.")

    print("Done." if not log["aborted"] else "Stopped early.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
