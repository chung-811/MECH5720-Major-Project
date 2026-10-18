# MECH5720 Major Project: Flutter Shutter/Coded Illumination

## Team and Roles
- Jack Thompson (490386475): Coded sequence design and deblurring pipeline
- Oliver Turner (520419913): Camera control and calibration in setup
- Krish Singh (530258892): Deconvolution & velocity estimation pipeline
- Daniel Chung (530585097): Monitor animation, velocity estimation algorithm
- Ben Ritchie (520479733): Test rig design and implementation

## Goal
Our system will compare three methods for flutter shutter photography with coded illumination: rolling shutter
deconvolved on a per-frame basis, rolling shutter deconvolved per-row and global shutter deconvolved per-frame.
Flutter shutter cameras break the light/motion blur trade-off that conventional cameras face when imaging moving
objects. Global shutter cameras overcome the artefacts rolling shutters exhibit from their row-by-row capture. The
rolling shutter artefacts can also be accounted for by applying the deconvolution on a row-by-row basis.
The recovered sharp images from flutter shutter cameras are useful for high speed applications. This project will
emulate the flutter shutter method with coded illumination to image fast-moving subjects and compare the performance
of the three different methods mentioned above.

## Approach
A fan with characters on the blades will be spinning at a controllable rate and imaged by a coded illumination rig
including a monitor and a camera. Within one exposure on the camera, the monitor will flash in a coded binary
sequence, producing a known blurring effect in the recorded image. Both a global shutter and a rolling shutter camera
will be used to image the subject and the images will be put through a processing pipeline which de-convolves the
image using the known shutter sequence to undo the motion blur and estimate a velocity for each of the 3 methods
discussed above. The results can then be compared.

## Success Criteria
All metrics are reported as mean ± std over ≥10 repeated captures per speed; ground-truth speed from the motor
step rate. Three-way comparison:
1. For each proposed method, we plot image quality against a photo of the stationary fan (PSNR/SSIM) for each
band of rows from top to bottom of the image. Target: per-row rolling shutter within 1 dB of global shutter across
all rows, while per-frame rolling shutter degrades away from the rows its code was timed for.
2. Fan legibility vs. speed: fraction of fan blade characters read correctly by OCR for coded light then deblurred,
constant light then deblurred, and a very short exposure with brightness boosted. Target: ≥90% read correctly
with coded light at speeds where constant light gives less than 50%.
3. Blind speed estimate: rotation speed estimated from the length of the blur arc, plotted against the true speed.
Target: within 5%.
Does the system behave as designed? We photograph a bright dot on a moving blade to see the blur pattern directly,
and check that it matches our flashing sequence and keeps the fine detail that constant light loses.

## Fallback
If a global shutter camera cannot be sourced and mounted by Week 10, we run the pipeline on the kit rolling shutter
camera alone and compare coded against constant monitor illumination. If per-row deconvolution fails, we deconvolve
each rolling shutter frame with a single kernel, keeping the full code inside the window where all rows are exposing. If
the stepper rig cannot hold a constant speed, we switch to a slow fan target with its speed measured independently
by high-frame-rate video. If blind speed estimation fails, we deblur using the commanded stepper speed and report
PSNR/SSIM against blur length. At minimum, we show that the measured kernel of a point target on the rig matches
the designed code and has no spectral nulls, unlike the box kernel.
6. Hardware
We will build: A coded illumination rig using a monitor displaying black/white screens in a designed binary sequence
while the global shutter camera and rolling shutter camera hold one continuous exposure, so the target’s motion is
convolved with a broadband code instead of a simple box blur. This will be tested primarily with a spinning fan
controlled via a stepper motor, so that the speed can be controlled and recorded.
