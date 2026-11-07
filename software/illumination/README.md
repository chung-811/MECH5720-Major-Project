# Coded illumination display

`coded_illumination.py` flashes the whole of the Raspberry Pi's monitor black and white in a binary
code, and can take one raw photo whose exposure covers the whole code. This is the standalone
prototype we used to bring up coded illumination and test the monitor. In the final system the same
job is done from the camera GUI's **Coded Capture** panel (`software/data_capture/coded`), which uses the
same JSON code files.

> **Photosensitivity warning:** flashing black and white faster than about 3 times a second can
> trigger seizures in people with photosensitive epilepsy. Warn anyone in the room.

## Run

On the Pi (over SSH, `DISPLAY=:0` picks the Pi's own screen):

```bash
DISPLAY=:0 python3 coded_illumination.py params_raskar52.json
```

Press ESC on the screen, or Ctrl-C in the terminal, to stop early.

## Code files

| File | Code |
| --- | --- |
| `params_raskar52.json` | the 52-bit code from Raskar et al. (2006), one bit per 60 Hz refresh |
| `params_constant52.json` | steady light for the same 0.87 s, for comparison |
| `params1.json`, `params_light_only.json` | a 16-bit test code over 1 s |
| `params_white.json` | a white screen for 5 minutes, for framing the camera |

Each file sets `sequence` (1 = white, 0 = black) and `total_duration_s`; the optional keys (repeat,
pre- and post-roll, `capture`, `analogue_gain`, `exposure_padding_s`, ...) are documented at the top of
`coded_illumination.py`. The screen only changes on a refresh, so the smallest bit is one refresh
(16.7 ms at 60 Hz). Bit changes are scheduled from one start time, so timing errors do not add up.

`data/coded/` holds two example captures from the first tests (19 Oct), each a `.dng` with the
`.json` log of when every bit was shown.
