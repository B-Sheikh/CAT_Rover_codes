/*
  car_controller.ino
  =====================================================================
  Runs on Arduino MEGA 2560. Talks to a Raspberry Pi 5 over USB serial.

  RESPONSIBILITIES:
    - Drive 4 DC motors (skid-steer) via 2x L298N motor drivers
    - Count ticks from all 4 wheel encoders using hardware interrupts (Pins 2, 3, 18, 19)
    - Ping 2 ultrasonic sensors (HC-SR04) for front obstacle detection
    - Optional IMU MPU-6050 read over I2C (Pins 20 SDA / 21 SCL) for pitch/roll elevation
    - Stream sensor data to Pi at 20 Hz, execute motor PWM commands from Pi
    - Fail-safe: stop motors if serial command stream goes silent (> 500 ms)

  SERIAL PROTOCOL (115200 baud, newline-terminated ASCII):
    Pi -> Arduino:
      "M,<leftSpeed>,<rightSpeed>\n"   left/right in -255..255 (negative = reverse)
      "S\n"                           emergency stop immediately

    Arduino -> Pi (sent automatically every 50ms / 20 Hz):
      "D,<us1_cm>,<us2_cm>,<encFL>,<encRL>,<encFR>,<encRR>,<pitch>,<roll>\n"

  PIN MAP (Arduino Mega 2560):
    Encoders (Dedicated Hardware Interrupts):
      ENC_FL = Pin 2  (INT4)
      ENC_RL = Pin 3  (INT5)
      ENC_FR = Pin 18 (INT3)
      ENC_RR = Pin 19 (INT2)

    L298N #1 -> LEFT Motors:
      Front-Left: ENA = 5 (PWM),  IN1 = 22, IN2 = 23
      Rear-Left:  ENB = 6 (PWM),  IN3 = 24, IN4 = 25

    L298N #2 -> RIGHT Motors:
      Front-Right: ENA = 9 (PWM), IN1 = 26, IN2 = 27
      Rear-Right:  ENB = 10 (PWM),IN3 = 28, IN4 = 29

    Ultrasonic Sensors:
      US1 (Left-Front):  TRIG = 30, ECHO = 31
      US2 (Right-Front): TRIG = 32, ECHO = 33

    IMU (MPU-6050) over I2C:
      SDA = Pin 20 (SDA)
      SCL = Pin 21 (SCL)
      VCC = 5V (or 3.3V)
      GND = GND
  =====================================================================
*/

#include <Arduino.h>
#include <Wire.h>

// ---------- Pin definitions ----------
const uint8_t ENC_FL_PIN = 2;
const uint8_t ENC_RL_PIN = 3;
const uint8_t ENC_FR_PIN = 18;
const uint8_t ENC_RR_PIN = 19;

const uint8_t FL_ENA = 5,  FL_IN1 = 22, FL_IN2 = 23;
const uint8_t RL_ENB = 6,  RL_IN3 = 24, RL_IN4 = 25;
const uint8_t FR_ENA = 9,  FR_IN1 = 26, FR_IN2 = 27;
const uint8_t RR_ENB = 10, RR_IN3 = 28, RR_IN4 = 29;

const uint8_t US1_TRIG = 30, US1_ECHO = 31;
const uint8_t US2_TRIG = 32, US2_ECHO = 33;

const uint8_t MPU_ADDR = 0x68;

// ---------- Timing ----------
const unsigned long SENSOR_PERIOD_MS = 50;   // ~20 Hz sensor stream
const unsigned long COMMAND_TIMEOUT_MS = 500; // fail-safe stop window
const unsigned long US_TIMEOUT_US = 25000UL;  // ~4m max range timeout

// ---------- State ----------
volatile unsigned long encFLTicks = 0;
volatile unsigned long encRLTicks = 0;
volatile unsigned long encFRTicks = 0;
volatile unsigned long encRRTicks = 0;

int cmdLeftSpeed = 0;   // -255..255, last commanded speed
int cmdRightSpeed = 0;

unsigned long lastSensorSend = 0;
unsigned long lastCommandTime = 0;

String rxBuffer = "";
bool imuAvailable = false;
float currentPitchDeg = 0.0;
float currentRollDeg = 0.0;

// ---------- Encoder ISRs ----------
void encFLISR() { encFLTicks++; }
void encRLISR() { encRLTicks++; }
void encFRISR() { encFRTicks++; }
void encRRISR() { encRRTicks++; }

// ---------- Motor helpers ----------
void setMotor(uint8_t ena, uint8_t in1, uint8_t in2, int speed) {
  speed = constrain(speed, -255, 255);
  bool forward = speed >= 0;
  uint8_t pwm = (uint8_t)abs(speed);
  digitalWrite(in1, forward ? HIGH : LOW);
  digitalWrite(in2, forward ? LOW : HIGH);
  analogWrite(ena, pwm);
}

void applyMotorCommand(int leftSpeed, int rightSpeed) {
  setMotor(FL_ENA, FL_IN1, FL_IN2, leftSpeed);
  setMotor(RL_ENB, RL_IN3, RL_IN4, leftSpeed);
  setMotor(FR_ENA, FR_IN1, FR_IN2, rightSpeed);
  setMotor(RR_ENB, RR_IN3, RR_IN4, rightSpeed);
}

void stopMotors() {
  cmdLeftSpeed = 0;
  cmdRightSpeed = 0;
  applyMotorCommand(0, 0);
}

// ---------- Ultrasonic ----------
float readUltrasonicCm(uint8_t trigPin, uint8_t echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  unsigned long duration = pulseIn(echoPin, HIGH, US_TIMEOUT_US);
  if (duration == 0) return -1.0;
  return duration / 58.0; // standard HC-SR04 conversion
}

// ---------- IMU (MPU-6050) ----------
void initIMU() {
  Wire.begin();
  Wire.setClock(400000); // 400 kHz fast I2C
  Wire.beginTransmission(MPU_ADDR);
  if (Wire.endTransmission() == 0) {
    // Wake up MPU-6050
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x6B); // PWR_MGMT_1 register
    Wire.write(0);    // set to zero (wakes up the MPU-6050)
    Wire.endTransmission(true);
    imuAvailable = true;
  } else {
    imuAvailable = false;
  }
}

void readIMU() {
  if (!imuAvailable) {
    currentPitchDeg = 0.0;
    currentRollDeg = 0.0;
    return;
  }

  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B); // starting with register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)6, (uint8_t)true);

  if (Wire.available() >= 6) {
    int16_t axRaw = (Wire.read() << 8) | Wire.read();
    int16_t ayRaw = (Wire.read() << 8) | Wire.read();
    int16_t azRaw = (Wire.read() << 8) | Wire.read();

    float ax = (float)axRaw;
    float ay = (float)ayRaw;
    float az = (float)azRaw;

    // Pitch: angle around Y-axis (+ uphill, - downhill)
    // Roll: angle around X-axis
    float pitch = atan2(ax, sqrt(ay * ay + az * az)) * 180.0 / PI;
    float roll = atan2(ay, sqrt(ax * ax + az * az)) * 180.0 / PI;

    currentPitchDeg = pitch;
    currentRollDeg = roll;
  }
}

// ---------- Serial command parsing ----------
void handleLine(const String &line) {
  if (line.length() == 0) return;

  if (line[0] == 'M') {
    // Format: M,left,right
    int firstComma = line.indexOf(',');
    int secondComma = line.indexOf(',', firstComma + 1);
    if (firstComma < 0 || secondComma < 0) return;

    int left = line.substring(firstComma + 1, secondComma).toInt();
    int right = line.substring(secondComma + 1).toInt();

    cmdLeftSpeed = constrain(left, -255, 255);
    cmdRightSpeed = constrain(right, -255, 255);
    applyMotorCommand(cmdLeftSpeed, cmdRightSpeed);
    lastCommandTime = millis();
  } else if (line[0] == 'S') {
    stopMotors();
    lastCommandTime = millis();
  }
}

void pollSerial() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      handleLine(rxBuffer);
      rxBuffer = "";
    } else if (c != '\r') {
      rxBuffer += c;
      if (rxBuffer.length() > 64) rxBuffer = "";
    }
  }
}

void sendSensorPacket() {
  float d1 = readUltrasonicCm(US1_TRIG, US1_ECHO);
  float d2 = readUltrasonicCm(US2_TRIG, US2_ECHO);
  readIMU();

  unsigned long fl, rl, fr, rr;
  noInterrupts();
  fl = encFLTicks;
  rl = encRLTicks;
  fr = encFRTicks;
  rr = encRRTicks;
  interrupts();

  Serial.print("D,");
  Serial.print(d1, 1);
  Serial.print(',');
  Serial.print(d2, 1);
  Serial.print(',');
  Serial.print(fl);
  Serial.print(',');
  Serial.print(rl);
  Serial.print(',');
  Serial.print(fr);
  Serial.print(',');
  Serial.print(rr);
  Serial.print(',');
  Serial.print(currentPitchDeg, 1);
  Serial.print(',');
  Serial.println(currentRollDeg, 1);
}

void setup() {
  pinMode(FL_ENA, OUTPUT); pinMode(FL_IN1, OUTPUT); pinMode(FL_IN2, OUTPUT);
  pinMode(RL_ENB, OUTPUT); pinMode(RL_IN3, OUTPUT); pinMode(RL_IN4, OUTPUT);
  pinMode(FR_ENA, OUTPUT); pinMode(FR_IN1, OUTPUT); pinMode(FR_IN2, OUTPUT);
  pinMode(RR_ENB, OUTPUT); pinMode(RR_IN3, OUTPUT); pinMode(RR_IN4, OUTPUT);

  pinMode(US1_TRIG, OUTPUT); pinMode(US1_ECHO, INPUT);
  pinMode(US2_TRIG, OUTPUT); pinMode(US2_ECHO, INPUT);

  pinMode(ENC_FL_PIN, INPUT_PULLUP);
  pinMode(ENC_RL_PIN, INPUT_PULLUP);
  pinMode(ENC_FR_PIN, INPUT_PULLUP);
  pinMode(ENC_RR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENC_FL_PIN), encFLISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_RL_PIN), encRLISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_FR_PIN), encFRISR, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_RR_PIN), encRRISR, RISING);

  stopMotors();
  initIMU();

  Serial.begin(115200);
  lastCommandTime = millis();
}

void loop() {
  pollSerial();

  // Fail-safe: no command from Pi for too long -> stop.
  if (millis() - lastCommandTime > COMMAND_TIMEOUT_MS) {
    if (cmdLeftSpeed != 0 || cmdRightSpeed != 0) stopMotors();
  }

  if (millis() - lastSensorSend >= SENSOR_PERIOD_MS) {
    lastSensorSend = millis();
    sendSensorPacket();
  }
}
