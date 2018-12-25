import gab.opencv.*;
import processing.video.*;
import java.awt.*;

PImage getFace;
Capture video;
OpenCV opencv;

// Used to limit the amount of face-captures
float s = 0.0;
float t = 0.0;
int count_face = 0;

// Sobel kernel, used for convolution
// See https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html
// for explanation on the working of such a kernel
float[][] kernel = {{ -1, -1, -1}, 
                    { -1,  9, -1}, 
                    { -1, -1, -1}};


// Need to use this in settings(), somehow setup() did not like this
void settings() {
  size(640, 480);
}


// Could actually make the size to be defined by a variable
void setup() {
  video = new Capture(this, 640/2, 480/2);
  opencv = new OpenCV(this, 640/2, 480/2);
  opencv.loadCascade(OpenCV.CASCADE_FRONTALFACE); 
  video.start();
}


// Drawing online what we need
void draw() {   
  scale(2);
  opencv.loadImage(video);
  image(video, 0, 0);  
  Rectangle[] faces = opencv.detect();  // Here we detect the face(s)...
     
  noFill();
  stroke(255, 0, 0);
  strokeWeight(2);
  loadPixels();
  
  // If the rectange faces contains multiple objects, we loop over them and draw a rectangle
  for (int i = 0; i < faces.length; i++) {
    //println(faces[i].x + "," + faces[i].y);
    rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);
  }
  
  // If we do find a face.. we can change the content of the first face (i=0) by applying the Sobel Kernel  
  if(faces.length > 0){
    int i = 0;
    PImage getFace = video.get(faces[i].x,faces[i].y,faces[i].width,faces[i].height);
    PImage img = getFace;
    img.loadPixels();      
    PImage edgeImg = createImage(img.width, img.height, RGB);
    
    // Loop through every pixel in the image.
    for (int y = 1; y < img.height-1; y++) {  // Skip top and bottom edges
      for (int x = 1; x < img.width-1; x++) {  // Skip left and right edges
        float sum = 0;  // Kernel sum for this pixel
        for (int ky = -1; ky <= 1; ky++) {
          for (int kx = -1; kx <= 1; kx++) {
            // Calculate the adjacent pixel for this kernel point
            int pos = (y + ky)*img.width + (x + kx);
            // Image is grayscale, red/green/blue are identical
            float val = red(img.pixels[pos]);
            // Multiply adjacent pixels based on the kernel values
            sum += kernel[ky+1][kx+1] * val;
          }
        }
        // For this pixel in the new image, set the gray value
        // based on the sum from the kernel
        edgeImg.pixels[y*img.width + x] = color(sum, sum, sum);
      }
    }
    // State that there are changes to edgeImg.pixels[]
    edgeImg.updatePixels();
    image(edgeImg, faces[i].x,faces[i].y,faces[i].width,faces[i].height); // Draw the new image on a specific location


    if(s-t > 10){
      println("we can save a picture");
      edgeImg.save("D:/webcam_face_" + count_face + t +".jpg");  // save one face...
      count_face = count_face + 1;
      t = second();  // With this we have a timestamp of the last picture taken
    }   
  }  
  
  s = second();  // With this we have a timestamp of the 'now'
  
  if (mousePressed) {
    video.save("D:/webcam_shot.jpg");  // Messing around with the mousePressed function
  }  
}

void captureEvent(Capture c) {
  c.read();
}