const int sensorPin = A0;
int sensorVal = 0;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); // open a serial port
  // 9600 bits per second is te speed at which the Genuino will communicate
}

void loop() {
  sensorVal = analogRead(sensorPin);
  
  Serial.println(sensorVal);
  //}
  // wait - 10 seconds 
  //delay(10000);
  
}
