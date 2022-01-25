#include <Servo.h>

/* 親指だけ1900～1500くらい */
/* 他の指は2000～1450くらい */

Servo thumb;
Servo index;
Servo middle;
Servo ring;
Servo pinky;

String myString;

int split(String data, char delimiter, String *dst){
    int index = 0;
    int arraySize = (sizeof(data)/sizeof((data)[0]));  
    int datalength = data.length();
    for (int i = 0; i < datalength; i++) {
        char tmp = data.charAt(i);
        if ( tmp == delimiter ) {
            index++;
            if ( index > (arraySize - 1)) return -1;
        }
        else dst[index] += tmp;
    }
    return (index + 1);
}

void setup(){
    Serial.begin(115200);
    thumb.attach(7);
    index.attach(8);
    middle.attach(9);
    ring.attach(10);
    pinky.attach(11);
    
    thumb.writeMicroseconds(1900);
    index.writeMicroseconds(2000);
    middle.writeMicroseconds(2000);
    ring.writeMicroseconds(2000);
    pinky.writeMicroseconds(2000);
    
    delay(2000);
}


void loop(){
    if(Serial.available()){
        String cmds[6] = {"\n"};
        myString = Serial.readStringUntil('\n');
        //Serial.println(myString);
        int ind = split(myString, ',', cmds);
        
        index.writeMicroseconds(process(cmds[0]));
        //Serial.println(cmds[0]);
        middle.writeMicroseconds(process(cmds[1]));
        ring.writeMicroseconds(process(cmds[2]));
        pinky.writeMicroseconds(process(cmds[3]));
        //thumb.writeMicroseconds(process(cmds[5]));
        delay(100); 
    }
    
}
int process(String str){
  Serial.println(map(str.toInt(), 0, 12, 2000, 1450));
  return map(str.toInt(), 0, 12, 2000, 1450);
}
