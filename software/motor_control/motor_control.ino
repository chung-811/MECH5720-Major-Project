/*
  ULN2003 Driver Board + 28BYJ-48 Stepper Motor -- Arduino Uno
  ------------------------------------------------------------
  No libraries needed. Switchable HALF-step / FULL-step drive with an
  acceleration/deceleration ramp.

    Full-step: 2048 steps/turn, faster top speed, a bit less smooth.
    Half-step: 4096 steps/turn, smoother and quieter, lower top speed.

  WIRING
    ULN2003 IN1  -> Arduino pin 8
    ULN2003 IN2  -> Arduino pin 9
    ULN2003 IN3  -> Arduino pin 10
    ULN2003 IN4  -> Arduino pin 11
    ULN2003 +    -> 5V  (an external 5V 1A+ supply is recommended for speed)
    ULN2003 -    -> GND (if using an external supply, connect its GND to Arduino GND too)
    Motor plug   -> white 5-pin socket on the ULN2003 board

    OPTIONAL MODE SWITCH (toggle or slide switch)
    Switch between Arduino pin 2 and GND.
      switch OPEN   -> full-step
      switch CLOSED -> half-step
    No resistor needed (internal pull-up is used). If you don't wire a switch,
    pin 2 just stays at "open" and you can use the serial command instead.
    Whichever you change last (switch or serial) wins.

  SERIAL CONTROL (Serial Monitor at 9600 baud, line ending = "Newline")
    s 2048   -> move 2048 steps        (negative = reverse)
    d 90     -> move 90 degrees        (negative = reverse)
    r 1.5    -> move 1.5 revolutions   (negative = reverse)
    v 15     -> set top speed in RPM (limited by MAX_STEP_HZ below)
    a 200    -> set ramp length (in full-step units; 0 = no ramp)
    m h      -> half-step mode
    m f      -> full-step mode
    m        -> show current mode
    ?        -> show help
*/

// ---- Pins ----
const int IN1 = 8;
const int IN2 = 9;
const int IN3 = 10;
const int IN4 = 11;
const int MODE_PIN = 2;   // optional half/full-step switch to GND

// ---- Motor / speed limits ----
// 28BYJ-48: ~2048 full-steps or ~4096 half-steps per output turn
const float MIN_RPM = 1.0;

// Highest step rate (steps per second) we allow. The datasheet gives >600 Hz for
// starting from standstill and >1000 Hz for running, both with NO load.
// 600 is conservative. Try raising it gradually (700, 800, 850) with the ramp on.
// Max RPM = MAX_STEP_HZ * 60 / stepsPerRev, so half-step tops out at half the RPM.
const float MAX_STEP_HZ = 2000.0;

// ---- Acceleration settings ----
const float START_HZ = 100.0;   // step rate the motor starts/finishes at (datasheet rated frequency)
long rampSteps = 200;           // ramp length in FULL-step units (doubled automatically in half-step)

// Half-step coil sequence. Bit 0 = IN1, bit 1 = IN2, bit 2 = IN3, bit 3 = IN4
// Full-step mode uses every second entry (the odd indices: two coils on at a time).
const uint8_t SEQUENCE[8] = {
  0b0001, 0b0011, 0b0010, 0b0110,
  0b0100, 0b1100, 0b1000, 0b1001
};

// ---- State ----
bool halfStep = false;           // false = full-step, true = half-step
long stepsPerRev = 2048;
int seqIndex = 1;                // odd = valid full-step position
float requestedRpm = 15.0;       // what you asked for
float currentRpm = 15.0;         // what's actually used (after the step-rate limit)
unsigned long stepDelayUs = 0;   // delay per step at top speed
int lastSwitchState = HIGH;

// ---- Low-level helpers ----
void writeCoils(uint8_t pattern) {
  digitalWrite(IN1, pattern & 0b0001);
  digitalWrite(IN2, pattern & 0b0010);
  digitalWrite(IN3, pattern & 0b0100);
  digitalWrite(IN4, pattern & 0b1000);
}

// Turn all coils off so the motor and driver don't heat up while idle
void releaseMotor() {
  writeCoils(0);
}

// delayMicroseconds() is only accurate up to ~16 ms, so split long delays
void waitUs(unsigned long us) {
  if (us > 16000UL) {
    delay(us / 1000UL);
    delayMicroseconds(us % 1000UL);
  } else {
    delayMicroseconds(us);
  }
}

// Recompute speed and delay for the current mode
void updateTiming() {
  float maxRpm = MAX_STEP_HZ * 60.0 / stepsPerRev;
  currentRpm = constrain(requestedRpm, MIN_RPM, maxRpm);
  stepDelayUs = (unsigned long)(60000000.0 / (stepsPerRev * currentRpm));
}

void setRpm(float rpm) {
  requestedRpm = max(rpm, MIN_RPM);
  updateTiming();
  Serial.print(F("Speed set to "));
  Serial.print(currentRpm);
  Serial.print(F(" RPM"));
  if (currentRpm < requestedRpm) {
    Serial.print(F(" (limited by MAX_STEP_HZ in this mode)"));
  }
  Serial.println();
}

void setMode(bool half) {
  if (half == halfStep) return;
  halfStep = half;
  stepsPerRev = halfStep ? 4096 : 2048;
  if (!halfStep && (seqIndex % 2 == 0)) {
    seqIndex = (seqIndex + 1) % 8;   // full-step only uses odd positions
  }
  updateTiming();
  Serial.print(F("Mode: "));
  Serial.print(halfStep ? F("half-step (4096 steps/rev)") : F("full-step (2048 steps/rev)"));
  Serial.print(F(", speed now "));
  Serial.print(currentRpm);
  Serial.println(F(" RPM"));
}

// If the physical switch was flipped since last time, apply it
void checkSwitch() {
  int s = digitalRead(MODE_PIN);
  if (s != lastSwitchState) {
    lastSwitchState = s;
    setMode(s == LOW);   // closed to GND = half-step
  }
}

// ---- Motion functions ----
// Positive = one direction, negative = the other.
// The step delay starts long (slow), shrinks to the top-speed delay over the
// ramp, and grows again over the last part of the move. Short moves get a
// triangular profile (accelerate to the halfway point, then slow down).
void moveSteps(long steps) {
  checkSwitch();   // pick up any switch change before moving

  int dir = (steps >= 0) ? 1 : -1;
  int stepInc = halfStep ? 1 : 2;
  long count = labs(steps);

  unsigned long startDelay = (unsigned long)(1000000.0 / START_HZ);
  long effectiveRamp = rampSteps * (stepsPerRev / 2048);
  long ramp = min(effectiveRamp, count / 2);
  bool useRamp = (ramp > 0) && (startDelay > stepDelayUs);

  for (long i = 0; i < count; i++) {
    unsigned long d = stepDelayUs;

    if (useRamp) {
      long fromEdge = min(i, count - 1 - i);   // distance from nearest end of the move
      if (fromEdge < ramp) {
        d = stepDelayUs + (startDelay - stepDelayUs) * (unsigned long)(ramp - fromEdge) / ramp;
      }
    }

    seqIndex = (seqIndex + dir * stepInc + 8) % 8;
    writeCoils(SEQUENCE[seqIndex]);
    waitUs(d);
  }
  releaseMotor();
}

void moveDegrees(float degrees) {
  moveSteps(lround(degrees * stepsPerRev / 360.0));
}

void moveRevolutions(float revs) {
  moveSteps(lround(revs * stepsPerRev));
}

// ---- Serial interface ----
void printHelp() {
  Serial.println(F("Commands:"));
  Serial.println(F("  s <steps>   move steps (e.g. s 2048, s -500)"));
  Serial.println(F("  d <degrees> move degrees (e.g. d 90, d -45)"));
  Serial.println(F("  r <revs>    move revolutions (e.g. r 1, r -0.5)"));
  Serial.println(F("  v <rpm>     set top speed in RPM (e.g. v 15)"));
  Serial.println(F("  a <steps>   set ramp length, 0 = off (e.g. a 200)"));
  Serial.println(F("  m h | m f   half-step / full-step mode (m alone shows mode)"));
  Serial.println(F("  ?           this help"));
  Serial.print(F("Mode: "));
  Serial.print(halfStep ? F("half-step") : F("full-step"));
  Serial.print(F(", speed: "));
  Serial.print(currentRpm);
  Serial.print(F(" RPM, ramp: "));
  Serial.print(rampSteps);
  Serial.println(F(" steps"));
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  char cmd = tolower(line.charAt(0));
  String arg = line.substring(1);
  arg.trim();
  float value = arg.toFloat();

  switch (cmd) {
    case 's':
      Serial.print(F("Moving steps: "));
      Serial.println((long)value);
      moveSteps((long)value);
      break;
    case 'd':
      Serial.print(F("Moving degrees: "));
      Serial.println(value);
      moveDegrees(value);
      break;
    case 'r':
      Serial.print(F("Moving revolutions: "));
      Serial.println(value);
      moveRevolutions(value);
      break;
    case 'v':
      setRpm(value);
      return;
    case 'a':
      rampSteps = max(0L, (long)value);
      Serial.print(F("Ramp set to "));
      Serial.print(rampSteps);
      Serial.println(F(" steps (full-step units)"));
      return;
    case 'm': {
      char m = (arg.length() > 0) ? tolower(arg.charAt(0)) : ' ';
      if (m == 'h') {
        setMode(true);
      } else if (m == 'f') {
        setMode(false);
      } else {
        Serial.print(F("Mode: "));
        Serial.println(halfStep ? F("half-step") : F("full-step"));
      }
      return;
    }
    case '?':
      printHelp();
      return;
    default:
      Serial.println(F("Unknown command. Type ? for help."));
      return;
  }
  Serial.println(F("Done."));
}

// ---- Arduino entry points ----
void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(MODE_PIN, INPUT_PULLUP);
  releaseMotor();

  Serial.begin(9600);

  // Start in the mode the switch says (open = full-step, closed = half-step)
  lastSwitchState = digitalRead(MODE_PIN);
  halfStep = (lastSwitchState == LOW);
  stepsPerRev = halfStep ? 4096 : 2048;
  seqIndex = halfStep ? 0 : 1;
  updateTiming();

  // Startup demo: one turn each way
  Serial.println(F("ULN2003 stepper ready. Running startup demo..."));
  moveRevolutions(1);
  delay(500);
  moveRevolutions(-1);

  printHelp();
}

void loop() {
  checkSwitch();
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }
}