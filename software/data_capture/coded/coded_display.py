#!/usr/bin/env python3
"""
Fullscreen black/white display for one coded illumination run.

Started by stream.py (as a subprocess, on the Pi's monitor via DISPLAY=:0)
during a coded capture. It runs in its own process because Tk needs to own
its main loop, which Flask already owns in the server process.

    python coded_display.py job.json

job.json holds {"bits": [...], "bit_duration_s": ..., "pre_roll_s": ...,
"post_roll_s": ...} (see coded_params.parse_code_params).

Protocol, one line each:
    stdout  READY            screen is black and pre-roll is done
    stdin   GO | ABORT       play the code now / quit without playing
    stdout  RESULT {json}    per-bit timing log, then the process exits

Press ESC on the illumination screen to stop a run early. To try it on its
own, run it from a terminal and type GO + ENTER.
"""

import sys
import json
import time
import queue
import threading

import tkinter as tk


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
        self.root.bind("<Escape>", self._abort)
        self.root.update()
        self.root.focus_force()

    def _abort(self, _event=None):
        self.aborted = True

    def show(self, on):
        """Fill the whole screen white (truthy) or black (falsy)."""
        self.canvas.configure(bg="white" if on else "black")
        self.root.update()

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


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


def wait_for_command(stage):
    """Block until GO or ABORT arrives on stdin, keeping the window alive."""
    lines = queue.Queue()

    def read_stdin():
        for line in sys.stdin:
            lines.put(line.strip().upper())
        lines.put("ABORT")                      # stdin closed: parent went away

    threading.Thread(target=read_stdin, daemon=True).start()
    while True:
        try:
            cmd = lines.get(timeout=0.02)
        except queue.Empty:
            stage.root.update()
            if stage.aborted:
                return "ABORT"
            continue
        if cmd in ("GO", "ABORT"):
            return cmd


def play(stage, bits, dur):
    """Show each bit for dur seconds. Returns the per-bit timing log."""
    log = {"bits": [], "late_bits": 0}
    n = len(bits)

    t0 = time.perf_counter() + 0.02              # small lead so bit 0 is not late
    log["t0_monotonic_ns"] = time.monotonic_ns() + 20_000_000

    for i, bit in enumerate(bits):
        planned = t0 + i * dur
        wait_until(stage, planned)
        if stage.aborted:
            break

        t_show = time.perf_counter()
        stage.show(bit)
        late = t_show - planned
        if late > 0.010:
            log["late_bits"] += 1
        log["bits"].append({"index": i, "value": bit,
                            "shown_at_s": round(t_show - t0, 4),
                            "late_s": round(max(late, 0.0), 4)})
    else:
        wait_until(stage, t0 + n * dur)         # last bit gets its full duration

    stage.show(0)
    log["end_s"] = round(time.perf_counter() - t0, 4)
    log["aborted"] = stage.aborted
    return log


def emit(line):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def main():
    if len(sys.argv) != 2:
        print("Usage: python coded_display.py job.json", file=sys.stderr)
        return 2
    with open(sys.argv[1], "r") as f:
        job = json.load(f)

    stage = LightStage()
    try:
        stage.show(0)
        wait_until(stage, time.perf_counter() + float(job.get("pre_roll_s", 0)))
        emit("READY")

        if wait_for_command(stage) != "GO":
            emit("RESULT " + json.dumps({"aborted": True, "bits": [], "late_bits": 0}))
            return 1

        log = play(stage, job["bits"], float(job["bit_duration_s"]))
        wait_until(stage, time.perf_counter() + float(job.get("post_roll_s", 0)))
        emit("RESULT " + json.dumps(log))
        return 0
    finally:
        stage.close()


if __name__ == "__main__":
    sys.exit(main())
