"""
"Coded Capture" section of the camview_capture.py control panel (dev
computer side), shared by the rolling and global shutter clients.

Pick a code file (light sequence only, see coded_params.py), set the camera
side in the GUI (gain etc. via Advanced Camera Controls, plus padding and
readout time here), and click Run Coded Capture. The Pi flashes the code on
its monitor during one long exposure; the DNG, PNG and a JSON run log are
saved to the current working directory as <prefix><timestamp>.{dng,png,json}.
"""

import glob
import json
import os
import threading
import time

import requests
from imgui_bundle import imgui

import coded_params

DEFAULT_PARAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params")
DEFAULT_PADDING_S = 0.05

WARN = (0.95, 0.65, 0.2, 1.0)
BAD = (0.9, 0.2, 0.2, 1.0)
GOOD = (0.2, 0.8, 0.2, 1.0)
BUSY = (0.9, 0.6, 0.2, 1.0)


class CodedCaptureUI:
    def __init__(self, base_url, prefix, params_dir=None):
        self.base_url = base_url
        self.prefix = prefix                     # e.g. "coded_gs_" / "coded_rs_"
        self.params_dir = params_dir or DEFAULT_PARAMS_DIR
        self.files = []
        self.file_idx = 0
        self.code_raw = None                     # the JSON object, sent as-is
        self.code = None                         # parsed, for the summary
        self.code_error = None
        self.info = None                         # from /coded_info
        self.padding_s = DEFAULT_PADDING_S
        self.readout_s = 0.0
        self.running = False
        self.reload_files()
        threading.Thread(target=self.fetch_info, daemon=True).start()

    # --- data ---------------------------------------------------------

    def reload_files(self):
        self.files = sorted(glob.glob(os.path.join(self.params_dir, "*.json")))
        self.file_idx = min(self.file_idx, max(len(self.files) - 1, 0))
        self.load_selected()

    def load_selected(self):
        self.code_raw = self.code = self.code_error = None
        if not self.files:
            self.code_error = "no .json code files in %s" % self.params_dir
            return
        try:
            with open(self.files[self.file_idx], "r") as f:
                self.code_raw = json.load(f)
            self.code = coded_params.parse_code_params(self.code_raw)
        except (OSError, ValueError) as err:
            self.code_error = str(err)

    def fetch_info(self):
        try:
            r = requests.get(f"{self.base_url}/coded_info", timeout=5)
            r.raise_for_status()
            self.info = r.json()
            if self.info.get("readout_s"):
                self.readout_s = float(self.info["readout_s"])
        except Exception as err:
            print("Failed to fetch /coded_info:", err)

    def exposure_s(self):
        if self.code is None:
            return None
        return self.code["total_duration_s"] + self.readout_s + self.padding_s

    # --- capture ------------------------------------------------------

    def start_capture(self, set_status):
        if self.running or self.code is None:
            return
        self.running = True
        threading.Thread(target=self._capture, args=(set_status,), daemon=True).start()

    def _capture(self, set_status):
        code_file = os.path.basename(self.files[self.file_idx])
        try:
            set_status("Coded capture: flashing %s on the Pi..." % code_file, BUSY)
            run_s = self.code["pre_roll_s"] + self.exposure_s() + self.code["post_roll_s"]
            r = requests.post(
                f"{self.base_url}/capture_coded",
                json={"code": self.code_raw,
                      "camera": {"padding_s": self.padding_s, "readout_s": self.readout_s}},
                timeout=run_s * 3 + 60)          # sync waits a few frames first
            if r.status_code != 200:
                try:
                    msg = r.json().get("error", r.text)
                except ValueError:
                    msg = r.text
                set_status("Coded capture failed: %s" % msg, BAD)
                return
            log = r.json()
            log["code_file"] = code_file

            set_status("Coded capture: downloading...", BUSY)
            stem = self.prefix + time.strftime("%Y%m%d_%H%M%S")
            for kind in ("dng", "png"):
                d = requests.get(f"{self.base_url}/coded_result/{log['id']}/{kind}", timeout=30)
                d.raise_for_status()
                with open(stem + "." + kind, "wb") as f:
                    f.write(d.content)
            with open(stem + ".json", "w") as f:
                json.dump(log, f, indent=2)

            got, req = log.get("actual_exposure_us", 0), log.get("requested_exposure_us", 0)
            msg = "Saved %s.dng/.png/.json (exposure %.3f s of %.3f s" % (stem, got / 1e6, req / 1e6)
            msg += ", %d late bit(s))" % log["late_bits"] if log.get("late_bits") else ")"
            set_status(msg, WARN if log.get("late_bits") else GOOD)
            print(msg)
        except Exception as err:
            set_status("Coded capture error: %s" % err, BAD)
        finally:
            self.running = False

    # --- GUI ----------------------------------------------------------

    def draw(self, set_status, camera_controls):
        """Draw the section; call once per frame inside the control panel."""
        if not imgui.collapsing_header("Coded Capture"):
            return

        names = [os.path.basename(p) for p in self.files] or ["(none)"]
        changed, idx = imgui.combo("Code file", self.file_idx, names)
        if changed and self.files:
            self.file_idx = idx
            self.load_selected()
        imgui.same_line()
        if imgui.button("Reload"):
            self.reload_files()

        if self.code_error:
            imgui.text_colored(BAD, self.code_error)
        elif self.code is not None:
            c = self.code
            imgui.text("%d bits over %.3f s  (%.1f ms/bit, %.1f flashes/s)"
                       % (len(c["bits"]), c["total_duration_s"],
                          c["bit_duration_s"] * 1000, c["flashes_per_s"]))
            for w in c["warnings"]:
                imgui.text_colored(WARN, "! " + w)

        imgui.spacing()
        if self.info is None:
            imgui.text_colored(WARN, "Camera info not loaded")
            imgui.same_line()
            if imgui.button("Retry"):
                threading.Thread(target=self.fetch_info, daemon=True).start()
        else:
            imgui.text("%s, %s shutter" % (self.info.get("model"), self.info.get("shutter")))

        gain = camera_controls.get("AnalogueGain", {}).get("current")
        imgui.text("AnalogueGain: %s (set in Advanced Camera Controls)"
                   % (gain if gain is not None else "1.0 default"))
        _, self.padding_s = imgui.input_float("Padding (s)", self.padding_s, 0.01, 0.1, "%.3f")
        _, self.readout_s = imgui.input_float("Readout time (s)", self.readout_s, 0.001, 0.01, "%.4f")
        self.padding_s = max(self.padding_s, 0.0)
        self.readout_s = max(self.readout_s, 0.0)

        exp = self.exposure_s()
        max_us = (self.info or {}).get("max_exposure_us")
        if exp is not None:
            too_long = bool(max_us) and exp * 1e6 > max_us
            text = "Exposure: %.3f s = code + readout + padding" % exp
            if max_us:
                text += "  (max %.3f s)" % (max_us / 1e6)
            imgui.text_colored(BAD if too_long else GOOD, text)

        imgui.begin_disabled(self.running or self.code is None)
        if imgui.button("Running..." if self.running else "Run Coded Capture"):
            self.start_capture(set_status)
        imgui.end_disabled()
        imgui.text_colored(WARN, "Flashing light: warn anyone who can see the Pi's screen.")
