#include <AFMotor.h>
#include <SoftwareSerial.h>

SoftwareSerial BT(2, 13);

AF_DCMotor M1(1);
AF_DCMotor M2(2);
AF_DCMotor M3(3);
AF_DCMotor M4(4);

int speedValue = 200;

void setup()
{
  BT.begin(9600);

  M1.setSpeed(speedValue);
  M2.setSpeed(speedValue);
  M3.setSpeed(speedValue);
  M4.setSpeed(speedValue);

  stopCar();
}

void loop()
{
  if (BT.available())
  {
    char command = BT.read();

    if (command == 'F')
      forward();

    else if (command == 'B')
      backward();

    else if (command == 'L')
      left();

    else if (command == 'R')
      right();

    else if (command == 'S')
      stopCar();

    else if (command >= '0' && command <= '9')
    {
      speedValue = map(command - '0', 0, 9, 0, 255);

      M1.setSpeed(speedValue);
      M2.setSpeed(speedValue);
      M3.setSpeed(speedValue);
      M4.setSpeed(speedValue);
    }
  }
}

void forward()
{
  M1.run(FORWARD);
  M2.run(FORWARD);
  M3.run(FORWARD);
  M4.run(FORWARD);
}

void backward()
{
  M1.run(BACKWARD);
  M2.run(BACKWARD);
  M3.run(BACKWARD);
  M4.run(BACKWARD);
}

void left()
{
  M1.run(BACKWARD);
  M2.run(BACKWARD);

  M3.run(FORWARD);
  M4.run(FORWARD);
}

void right()
{
  M1.run(FORWARD);
  M2.run(FORWARD);

  M3.run(BACKWARD);
  M4.run(BACKWARD);
}

void stopCar()
{
  M1.run(RELEASE);
  M2.run(RELEASE);
  M3.run(RELEASE);
  M4.run(RELEASE);
}