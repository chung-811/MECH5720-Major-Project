# Code design

`code_design.py` compares flashing codes by how invertible the blur they make is. For each code it builds
the blur as a matrix and reports:

- **Noise amplification** (dB): how much deblurring magnifies sensor noise, the metric of
  Raskar et al. (2006). Lower is better.
- **Frequency response** of the blur: steady light (a box) has exact zeros, where detail is lost for
  good; a good code stays well above zero at every frequency.

It also models what a real monitor does to a code: a first-order lag smooths every on/off edge, so a
code designed for an ideal shutter loses some of its advantage.

```bash
python code_design.py      # prints the table, saves code_spectrum.png
```

The codes the rig actually used are JSON files in
[`software/data_capture/coded/params`](../data_capture/coded/params): `code_16bit_best_4refresh.json`
(16 bits, four 60 Hz refreshes each, 1.07 s) and `flat_16bit_4refresh.json` (steady light for the same
time). The monitor's measured response (44 ms to brighten, dims at once) is modelled in
`software/deconvolution/real.py`.
