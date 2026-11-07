# Motor control

Arduino Uno sketch that spins the fan at a set speed with a 28BYJ-48 stepper motor through a ULN2003
driver board. No libraries needed. Because the motor steps at a known rate, the commanded speed is the
ground truth that the deblurring and the speed estimate are scored against.

## Wiring

| ULN2003 | Arduino Uno |
| --- | --- |
| IN1, IN2, IN3, IN4 | pins 8, 9, 10, 11 |
| + | 5 V (an external 5 V, 1 A supply is better at speed) |
| − | GND (share it with the external supply, if used) |

Optional: a switch between pin 2 and GND selects half-step (closed) or full-step (open).

## Use

Open `motor_control.ino` in the Arduino IDE, upload it, then open the Serial Monitor at
**9600 baud** with line ending **Newline**.

| Command | Does |
| --- | --- |
| `v 3` | set the speed to 3 rpm |
| `r 5` | turn 5 revolutions |
| `s 400` | turn 400 steps (negative reverses) |
| `d 90` | turn 90 degrees |
| `a 200` | acceleration ramp length (0 for none) |
| `m f` / `m h` | full-step / half-step |
| `?` | help |

## Note on steps per revolution

The sketch assumes 2048 full steps per revolution, the figure usually quoted for the 28BYJ-48. The real
gear ratio gives **2038**, so the fan turns about 0.5% faster than commanded. The deblurring code uses
2038 (`--steps-per-rev 2038`).

On the rig: `s 200` once to take up gear slack, `s 400` between the three calibration stills, then
`v 3` and `r 5` while capturing. The full lab protocol is in
[`software/deconvolution/README.md`](../deconvolution/README.md).
