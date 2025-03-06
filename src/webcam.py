#import of the module for detection the hand in the stream video
#the code below works only with some version of the library, in my case i use OpenCV 4.5.5, CVZone 1.4.1 and mediapipe 0.8.8
from cvzone.HandTrackingModule import HandDetector
import cvzone
import cv2

#init webcam resolution and HandDetector object for hand tracking
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
detector = HandDetector(detectionCon=0.75)

#get class names from the train generator
classNames = list(train_generator.class_indices.keys())

#parameters to optimize the code and improve the prediction
confVal = 30
flag = True

with tf.device("/device:GPU:0"):
    while True:
        #get current frame from the webcam
        success, img = cap.read() 

        #detect the hand in the current frame
        img = detector.findHands(img, draw=False)
        lmList, bbox = detector.findPosition(img, draw=False)

        #whether there's a hand in the frame
        if bbox:
            
            #get coords of the bounded box
            x,y,w,h = bbox["bbox"]

            #using confVal to get a bigger portion of the images and maing sure i get all the pixel of the hand
            x = x - (confVal * 2); y = y - confVal
            h = h + (confVal * 3); w = w + (confVal * 3)

            #if the previous istruction don't identify some pixels outside the curren frame
            if(x >= 0 and y >= 0 and (x+w) < 640 and (y+h) < 480):
                #update the bounded box coords          	
                bbox["bbox"] = x,y,w,h

                #slicing operation with which i can obtain the portion of the images that encloses the hand
                input_image = img[y:y+h, x:x+w] 
                cvzone.cornerRect(img, bbox["bbox"])

		#if the flag used to optimize the code is True
                if flag:
                    #pre-processing of the image so that it can match the input size of the net
                    input_image = cv2.resize(input_image, dsize=(single_size, single_size))
                    if color_mode != "rgb":
                        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                    input_image = np.expand_dims(input_image, axis=0)

                    #classification
                    output = newModel.predict(input_image)
                    output = np.argmax(output, axis=1)

                #negation of the flag and print the class name
                flag = not flag
                cv2.putText(img, classNames[output[0]], (x - 15, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),  thickness=4)
        
        #show the current frame
        cv2.imshow("Image", img)

        #press 'q' to close the application
        if cv2.waitKey(1) == ord('q'):
            break; 

#release all the resources
cap.release()
cv2.destroyAllWindows()
