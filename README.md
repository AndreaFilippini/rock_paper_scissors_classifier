### Real-time classification of hand gestures of "rock, paper, scissors" game
This project contains the code to implement a CNN that can classify the gestures of "rock, paper, scissors" game in real-time.  
The optimal structure was find after 33 epochs, with a train accuracy of 0.9835 and a validation loss of 0.0164.   
The classification is done using the webcam of the user.  
![alt text](https://github.com/AndreaFilippini/rock_paper_scissors_classifier/blob/main/final_result/result.gif?raw=true)

### Dependencies
[Tensorflow](https://www.tensorflow.org/)  
[Keras](https://keras.io/)  
[OpenCV](https://opencv.org/)  
[CVZone](https://www.computervision.zone/)  

### Dataset
The net was trained on a costum dataset that conatains 36084 images, in which each one has a size of 300x300x3 and a different background.  
The images were splitted in a train set, validation test and test set with a 28764, 5856 and 1464 images respectively.  
The reason why each images has a different bg is beacause the net is able to generalize better, recognizing the shape of the hand regardless of the background.
