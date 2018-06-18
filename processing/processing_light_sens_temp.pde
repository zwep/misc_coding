import processing.serial.*;

Serial myPort;  // Create object from Serial class
String val;     // Data received from the serial port
String[] val_list;
String path = "C:\\Users\\C35612.LAUNCHER\\Production_data\\Light_sensor_temp";

int numReadings = 1; //keeps track of how many readings you'd like to take before writing the file. 
int readingCounter = 0; //counts each reading to compare to numReadings. 
 
String fileName;

Table datatable = new Table(); // Define a table where we store the results

void setup()
{
  // Get the right port from te arduino
  String portName = Serial.list()[3]; //change the 0 to a 1 or 2 etc. to match your port
  println(portName);
  myPort = new Serial(this, portName, 9600);
  
  datatable.addColumn("id"); //This column stores a unique identifier for each record. We will just count up from 0 - so your first reading will be ID 0, your second will be ID 1, etc. 
  
  //the following adds columns for time. You can also add milliseconds. See the Time/Date functions for Processing: https://www.processing.org/reference/ 
  datatable.addColumn("year");
  datatable.addColumn("month");
  datatable.addColumn("day");
  datatable.addColumn("hour");
  datatable.addColumn("minute");
  datatable.addColumn("second");
  
  //the following are dummy columns for each data value. Add as many columns as you have data values. Customize the names as needed. Make sure they are in the same order as the order that Arduino is sending them!
  datatable.addColumn("sensorVal");
  datatable.addColumn("sensorVal_1");
}


void serialEvent(Serial myPort){
  try {
    val = myPort.readStringUntil('\n'); //The newline separator separates each Arduino loopfeature_BAG_sel. We will parse the data by each newline separator.
    if (val!= null) { //We have a reading! Record it.
      val_list = split(trim(val),","); //gets rid of any whitespace or Unicode nonbreakable space
      print(val_list[0]);
      println(val_list[1]); //Optional, useful for debugging. If you see this, you know data is being sent. Delete if you like.
       
      TableRow newRow = datatable.addRow(); //add a row for this new reading
      newRow.setInt("id", datatable.lastRowIndex());//record a unique identifier (the row’s index)
      
      //record time stamp
      newRow.setInt("year", year());
      newRow.setInt("month", month());
      newRow.setInt("day", day());
      newRow.setInt("hour", hour());
      newRow.setInt("minute", minute());
      newRow.setInt("second", second());
      
      //record sensor information. Customize the names so they match your sensor column names.	
      newRow.setString("sensorVal", val_list[0]);
	    newRow.setString("sensorVal_1", val_list[1]);
       
      readingCounter++; //optional, use if you’d like to write your file every numReadings reading cycles
    
      //saves the table as a csv in the same folder as the sketch every numReadings.
      if (readingCounter % numReadings ==0)//The % is a modulus, a math operator that signifies remainder after division. The if statement checks if readingCounter is a multiple of numReadings (the remainder of readingCounter/numReadings is 0)
      {
        fileName = path + str(year()) + str(month()) + str(day()) + ".csv"; //this filename is of the form year+month+day+readingCounter
        saveTable(datatable, fileName); //Woo! save it to your computer. It is ready for all your spreadsheet dreams.
      }
    }
  }
  catch(RuntimeException e) {
  e.printStackTrace();
  }
}
 

void draw()
{ 
   //visualize your sensor data in real time here! In the future we hope to add some cool and useful graphic displays that can be tuned to different ranges of values. 
}