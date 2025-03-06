import cv2
import os
import sys
from PIL import Image
import numpy as np
import random

start = False
count = 0

num_samples = 10

x, y = 100, 100
rect_size = 300
label_name = 'paper'

BG_CLASS_PATH = 'C:/Users/___/Desktop/background/'
IMG_SOURCE_PATH = 'C:/Users/___/Desktop/dataset/'
IMG_SAVE_CLASS_PATH = os.path.join(IMG_SOURCE_PATH, label_name)

# folders that need to be created
new_dir = [BG_CLASS_PATH , IMG_SOURCE_PATH, IMG_SAVE_CLASS_PATH]

# check for the existence of each folder in new_dir
for folder in new_dir:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print("FAILED TO CREATE FOLDER {}".format(folder))

# check the existence of files inside the background folder
bg_files = [os.path.isfile(os.path.join(BG_CLASS_PATH, image)) for image in os.listdir(BG_CLASS_PATH)]
num_bg = np.count_nonzero(bg_files)

if(num_bg <= 0):
    print("No background available")
    sys.exit()

# init camera video capture object
cap = cv2.VideoCapture(0)

while True:
    # read frames and capture status of the current operation
    ret, frame = cap.read()
    if not ret:
        continue
    
    # if the max amount of samples is reached, exit the loop
    if count == num_samples:
        break

    # sampling area
    cv2.rectangle(frame, (x, y), (x + rect_size, y + rect_size), (255, 255, 255), 2)
    
    if start:
        bg_path = os.path.join(BG_CLASS_PATH, '{}.jpg'.format(random.randint(1, num_bg)))
        save_path = os.path.join(IMG_SAVE_CLASS_PATH, '{}.jpg'.format(count + 1))
        bg = Image.open(bg_path)
        roi = frame[x:(x + rect_size), y:(y + rect_size)]
       
        # get all pixels that are black and which will be part of the background
        black_pixels = np.where(
                (roi[:, :, 0] <= 50) &
                (roi[:, :, 1] <= 50) &
                (roi[:, :, 2] <= 50)
            ) 
        
        # sample random portion of a background with 300x300 size
        x_coord = random.randint(0, 3840 - rect_size)
        y_coord = random.randint(0, 2160 - rect_size)
        rect = (x_coord, y_coord, x_coord + rect_size, y_coord + rect_size)
        bg_array = np.asarray(bg.crop(rect))
        bg_array = cv2.cvtColor(bg_array, cv2.COLOR_BGR2RGB)

        # replace black pixels with the random sampled portion
        roi[black_pixels] = bg_array[black_pixels]
        frame[x:(x + rect_size), y:(y + rect_size)] = roi
        
        # save the file inside the folder
        cv2.imwrite(save_path, roi)
        count += 1

    # put the text of the current counter value and show the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Collecting {}".format(count),
            (5, 50), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting images", frame)

    # input check to start the sampling procedure or stopping it
    if cv2.waitKey(10) == ord('a'):
        start = not start
        
    if cv2.waitKey(10) == ord('q'):
        break
        
# release all the resources
cap.release()
cv2.destroyAllWindows()
print("\n{} image(s) saved to {}".format(count, IMG_SAVE_CLASS_PATH))