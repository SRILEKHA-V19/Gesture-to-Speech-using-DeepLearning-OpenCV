import os
import re
import cv2
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


'''def switch_class_label(i, outputVec):
    switcher = {
            0: outputVec.append([1, 0, 0, 0, 0]),
            1: outputVec.append([0, 1, 0, 0, 0]),
            2: outputVec.append([0, 0, 1, 0, 0]),
            3: outputVec.append([0, 0, 0, 1, 0]),
            4: outputVec.append([0, 0, 0, 0, 1])
        }
    return switcher.get(i, "Invalid Input")'''



if __name__ == '__main__':
    
    ''' ---------- Images and their corresponding labels i.e, categories are grouped together ---------- '''

    # Load images from all 5 classes(categories), Output results in outputVec(vector)

    storeImages = []
    outputVec = []

    c1 = 0
    c2 = 0
    c3 = 0
    c4 = 0
    c5 = 0

    for dirpath, dirnames, files in os.walk("/Users/svinjamara/Documents/Preprocess_5_Resized"):
        for file in files:
            '''if bfr_extension == "iiok":
                class_name = 0
            elif bfr_extension == "stop":
                class_name = 1
            elif bfr_extension == "peace":
                class_name = 2
            elif bfr_extension == "punch":
                class_name = 3
            elif bfr_extension == "nothing":
                class_name = 4'''
            
            # Loading images 
            path_name = os.path.join(dirpath, file)

            # print(path_name)
            
            image = cv2.imread(path_name)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            storeImages.append(gray_image.reshape(100, 100, 1))
            
            # Corresponding output vectors for the loaded image
            # output = switch_class_label(class_name, outputVec)

            bfr_extension = os.path.splitext(file)[0]
            bfr_extension = re.sub("\d+", "", bfr_extension)

            # print(bfr_extension)

            if bfr_extension == "iiok__":
                outputVec.append([1, 0, 0, 0, 0])
                c1 += 1
            elif bfr_extension == "stop__":
                outputVec.append([0, 1, 0, 0, 0])
                c2 += 1
            elif bfr_extension == "peace__":
                outputVec.append([0, 0, 1, 0, 0])
                c3 += 1
            elif bfr_extension == "punch__":
                outputVec.append([0, 0, 0, 1, 0])
                c4 += 1
            else:
                outputVec.append([0, 0, 0, 0, 1])
                c5 += 1

    print("{}, {}, {}, {}, {}".format(c1, c2, c3, c4, c5))
    
    ''' ---------- Train-Test Splitting ---------- '''

    # Shuffling Training data to perform train-test split
    storeImages, outputVec = shuffle(storeImages, outputVec, random_state = 0)

    # Train-test split on the entire input data(dataset under consideration)
    train_images, test_images, train_labels, test_labels = train_test_split(storeImages, outputVec, test_size = 0.2)


    ''' ---------- CNN Model Architecture ---------- '''

    # Resetting the tensorboard graphs for every training/epoch (?) -> check the actual definition, not self-understanding
    tf.reset_default_graph()

    # Input layer
    convnet = input_data(shape = [None, 100, 100, 1], name = 'input')

    # Conv and Max-Pool layers
        
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


    ''' ---------- Fitting and Testing of model for Prediction -> accuracy, Saving model to disk ---------- '''

    train_images, train_labels = shuffle(train_images, train_labels, random_state = 0)

    model.fit(train_images, train_labels,
              n_epoch = 8,
              validation_set = (test_images, test_labels),
              show_metric = True)

    accuracy_score = model.evaluate(train_images, train_labels)

    print('Accuracy: {:.2f}%'.format(accuracy_score[0] * 100.00))

    model.save("/Users/svinjamara/Documents/gesture_NN/2_model_5classes_new.tfl")


