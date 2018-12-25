const int sensorPin_temp = A0;
const int sensorPin_light_sens = A1;

int sensorVal_temp;
int sensorVal_light_sens;


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600); // open a serial port
  // 9600 bits per second is te speed at which the Genuino will communicate
}

void loop() {
  sensorVal_temp = analogRead(sensorPin_temp);
  sensorVal_light_sens = analogRead(sensorPin_light_sens);

  Serial.print(sensorVal_temp);
  Serial.print(", ");
  Serial.println(sensorVal_light_sens);
  // wait: 10 seconds 
  //delay(10000);  
}
