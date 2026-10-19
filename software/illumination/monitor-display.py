#!/usr/bin/env python3
"""Flash the screen black/white from a binary sequence in a JSON parameters file.

Usage: python monitor-display.py PARAMETERS_FILE.json
Press Escape or Q to quit at any time.

Parameters (JSON):
  sequence         string of 0s and 1s, e.g. "000011101011101010"  (0 = black, 1 = white)
  bit_duration_ms  how long each bit is shown, in milliseconds
    -or-
  frequency_hz     bits per second (alternative to bit_duration_ms)
  loop             optional, default true: repeat the sequence
  duration_s       optional: stop after this many seconds
"""
import json
import sys
import time
import tkinter as tk

COLORS = {"0": "#000000", "1": "#FFFFFF"}


def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_steps(params):
    """Turn the parameters into a list of (bit, duration_ms) steps.
    Consecutive identical bits are merged so the screen only redraws on a change."""
    seq = params.get("sequence")
    if seq is None:
        sys.exit("Error: parameters file needs a 'sequence' of 0s and 1s.")
    if isinstance(seq, list):
        seq = "".join(str(b) for b in seq)
    if not seq or any(b not in "01" for b in seq):
        sys.exit("Error: 'sequence' may only contain the characters 0 and 1.")

    if "bit_duration_ms" in params:
        bit_ms = float(params["bit_duration_ms"])
    elif "frequency_hz" in params:
        bit_ms = 1000.0 / float(params["frequency_hz"])
    else:
        sys.exit("Error: need 'bit_duration_ms' or 'frequency_hz'.")
    if bit_ms <= 0:
        sys.exit("Error: bit duration must be positive.")

    steps = []
    for b in seq:
        if steps and steps[-1][0] == b:
            steps[-1][1] += bit_ms
        else:
            steps.append([b, bit_ms])
    return [(b, ms) for b, ms in steps]


class Flasher:
    def __init__(self, steps, duration_s, loop):
        self.steps = steps
        self.duration_s = duration_s
        self.loop = loop
        self.index = 0

        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=COLORS["0"], cursor="none")
        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("q", lambda e: self.root.destroy())

    def run(self):
        self.start = time.perf_counter()
        self.deadline = self.start
        self.root.after(0, self.next_step)
        self.root.mainloop()

    def next_step(self):
        now = time.perf_counter()
        if self.duration_s and now - self.start >= self.duration_s:
            self.root.destroy()
            return
        if self.index >= len(self.steps):
            if not self.loop:
                self.root.destroy()
                return
            self.index = 0

        bit, ms = self.steps[self.index]
        self.root.configure(bg=COLORS[bit])
        self.root.update_idletasks()
        self.index += 1

        # Schedule against an absolute deadline so timing errors don't accumulate
        self.deadline += ms / 1000.0
        delay_ms = max(0, int((self.deadline - time.perf_counter()) * 1000))
        self.root.after(delay_ms, self.next_step)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python monitor-display.py PARAMETERS_FILE.json")

    params = load_params(sys.argv[1])
    steps = build_steps(params)

    print("WARNING: this program flashes the screen rapidly and can trigger")
    print("seizures in people with photosensitive epilepsy.")
    input("Press Enter to start (Escape or Q quits once running)... ")

    Flasher(steps, params.get("duration_s"), params.get("loop", True)).run()


if __name__ == "__main__":
    main()
