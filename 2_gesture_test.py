import numpy as np
import cv2
import imutils

import tensorflow as tf
import tflearn

from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from PIL import Image



# Global Variable: background
bg = None



''' ---------- Helper functions ---------- '''

def resizeImage(image_name):
    basewidth = 100
    img = Image.open(image_name)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
    img.save(image_name)



def run_avg(image, avgWt):
    global bg
    
    if bg is None:
        bg = image.copy().astype("float")
        return
    # Compute & accumulate weighted avg, Update background
    cv2.accumulateWeighted(image, bg, avgWt)



def segment(image, threshold = 25):
    global bg
    
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # Contours in the thresholded image
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return
    else:
        # Get max contour area => hand
        segmented = max(cnts, key = cv2.contourArea)
        return(thresholded, segmented)

    

def getPredictedClass():
    image = cv2.imread('temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(100, 100, 1)])
    return np.argmax(prediction), (np.amax(prediction) / (prediction[0][0] + prediction[0][1] + prediction[0][2]))



def showStatistics(predictedClass, confidence):
    textImage = np.zeros((300, 512, 3), np.uint8)
    className = ""

    if predictedClass == 0:
        className = "Okay"
    elif predictedClass == 1:
        className = "Stop"
    elif predictedClass == 2:
        className = "Peace"
    elif predictedClass == 3:
        className = "Punch"
    else:
        className = "Nothing"

    cv2.putText(textImage, "Predicted Class : " + className, (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(textImage, "Accuracy : " + str(confidence * 100.00) + '%', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Statistics", textImage)
    
     


''' ---------- main ---------- '''

def main():
    
    # Weight for running average
    avgWt = 0.5

    camera = cv2.VideoCapture(0)

    # ROI
    top, right, bottom, left = 10, 350, 225, 590 #10, 70, 380, 625 

    num_frames = 0
    start_recording = False

    while(True):
        # Getting the current frame
        (grabbed, frame) = camera.read()
               
        frame = imutils.resize(frame, width = 700)

        # Flipping frame to avoid mirror view(input fed from web cam)
        frame = cv2.flip(frame, 1)

        clone = frame.copy()

        (height, width) = frame.shape[:2]

        roi = frame[top:bottom, right:left]

        # convert the ROI to grayscale and apply Gaussian Blur filter
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # Threshold for Running Average model to be computed
        if num_frames < 30:
            run_avg(gray, avgWt)
        else:
            hand = segment(gray)

            if hand is not None:
                # If hand is segmented, unpack the thresh image and segment
                (thresholded, segmented) = hand


                # Display the segmented region with the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

                if start_recording:
                    cv2.imwrite('temp.png', thresholded)
                    resizeImage('temp.png')
                    predictedClass, confidence = getPredictedClass()
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thresholded", thresholded)

        # Draw segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)

        num_frames += 1

        # Display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        keypress = cv2.waitKey(1) & 0xFF
        # if user pressed 'q', then stop looping
        if keypress == ord("q"):
            break

        if keypress == ord("s"):
            start_recording = True

    print("Successfully captured")


if __name__ == '__main__':
    
    ''' Defining the Model to load the trained model file from disk '''

    tf.reset_default_graph()

    convnet = input_data(shape = [None, 100, 100, 1], name = 'input')
    
    convnet = conv_2d(convnet, 512, 2, activation = 'relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation = 'relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation = 'relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 64, 2, activation = 'relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 128, 2, activation = 'relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 256, 2, activation = 'relu')
    convnet = max_pool_2d(convnet, 2)

    convnet = conv_2d(convnet, 512, 2, activation = 'relu')
    convnet = max_pool_2d(convnet, 2)

    # Fully-Connected layers
    convnet = fully_connected(convnet, 1000, activation = 'relu')
    convnet = dropout(convnet, 0.50)

    convnet = fully_connected(convnet, 5, activation = 'softmax')

    # Loss estimate and Model-Fitting
    convnet = regression(convnet, optimizer = 'adam', learning_rate = 0.001, loss = 'categorical_crossentropy', name = 'regression')

    # CNN Model creation, [tensorboard_verbose = 0] --> to visualize graphs in tensorboard!! 
    model = tflearn.DNN(convnet, tensorboard_verbose = 0)

    model.load('./2_model_5classes_new.tfl', weights_only = True)

    # Call main(), the driver function
    main()
