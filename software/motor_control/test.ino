/*
  ULN2003 Driver Board + 28BYJ-48 Stepper Motor -- Arduino Uno
  ------------------------------------------------------------
  No libraries needed. Uses half-stepping for smooth, quiet motion
  and the most torque.

  WIRING
    ULN2003 IN1  -> Arduino pin 8
    ULN2003 IN2  -> Arduino pin 9
    ULN2003 IN3  -> Arduino pin 10
    ULN2003 IN4  -> Arduino pin 11
    ULN2003 +    -> 5V  (Arduino 5V is OK for one motor; external 5V supply is better)
    ULN2003 -    -> GND (if using an external supply, connect its GND to Arduino GND too)
    Motor plug   -> white 5-pin socket on the ULN2003 board

  SERIAL CONTROL (Serial Monitor at 9600 baud, line ending = "Newline")
    s 2048   -> move 2048 steps        (negative = reverse)
    d 90     -> move 90 degrees        (negative = reverse)
    r 1.5    -> move 1.5 revolutions   (negative = reverse)
    v 12     -> set speed in RPM (1 to 15)
    ?        -> show help
*/

// ---- Pins ----
const int IN1 = 8;
const int IN2 = 9;
const int IN3 = 10;
const int IN4 = 11;

// ---- Motor constants ----
// 28BYJ-48: 64 half-steps per motor turn x ~64:1 gearbox = 4096 half-steps per output turn
// (the true gear ratio is 63.68:1, so ~4076 is slightly more precise if you need it)
const long STEPS_PER_REV = 4096;
const float MIN_RPM = 1.0;
const float MAX_RPM = 15.0;   // above ~15 RPM the 28BYJ-48 tends to stall

// Half-step coil sequence. Bit 0 = IN1, bit 1 = IN2, bit 2 = IN3, bit 3 = IN4
const uint8_t SEQUENCE[8] = {
  0b0001, 0b0011, 0b0010, 0b0110,
  0b0100, 0b1100, 0b1000, 0b1001
};

int seqIndex = 0;
unsigned long stepDelayUs = 0;
float currentRpm = 10.0;

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

void setRpm(float rpm) {
  currentRpm = constrain(rpm, MIN_RPM, MAX_RPM);
  stepDelayUs = (unsigned long)(60000000.0 / (STEPS_PER_REV * currentRpm));
}

// ---- Motion functions ----
// Positive = one direction, negative = the other
void moveSteps(long steps) {
  int dir = (steps >= 0) ? 1 : -1;
  long count = labs(steps);

  for (long i = 0; i < count; i++) {
    seqIndex = (seqIndex + dir + 8) % 8;
    writeCoils(SEQUENCE[seqIndex]);
    delayMicroseconds(stepDelayUs);
  }
  releaseMotor();
}

void moveDegrees(float degrees) {
  moveSteps(lround(degrees * STEPS_PER_REV / 360.0));
}

void moveRevolutions(float revs) {
  moveSteps(lround(revs * STEPS_PER_REV));
}

// ---- Serial interface ----
void printHelp() {
  Serial.println(F("Commands:"));
  Serial.println(F("  s <steps>   move steps (e.g. s 2048, s -500)"));
  Serial.println(F("  d <degrees> move degrees (e.g. d 90, d -45)"));
  Serial.println(F("  r <revs>    move revolutions (e.g. r 1, r -0.5)"));
  Serial.println(F("  v <rpm>     set speed 1-15 RPM (e.g. v 12)"));
  Serial.println(F("  ?           this help"));
  Serial.print(F("Current speed: "));
  Serial.print(currentRpm);
  Serial.println(F(" RPM"));
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  char cmd = tolower(line.charAt(0));
  float value = line.substring(1).toFloat();

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
      Serial.print(F("Speed set to "));
      Serial.print(currentRpm);
      Serial.println(F(" RPM"));
      return;
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
  releaseMotor();

  Serial.begin(9600);
  setRpm(currentRpm);

  // Startup demo: one turn each way
  Serial.println(F("ULN2003 stepper ready. Running startup demo..."));
  moveRevolutions(1);
  delay(500);
  moveRevolutions(-1);

  printHelp();
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }
}
