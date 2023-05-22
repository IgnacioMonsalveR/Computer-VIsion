# Helper functions

import os
import numpy as np
import glob # library for loading images from a directory
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2


# This function is used to load images and their labels and put them in a list 

def load_images(image_dir): 
    # Populate the empty image list 
    im_list = []
    image_types = ["day", "night"]

    # iterate through each folder 
    for im_type in image_types:
        # * is used to iterate over all files regardeless of their specific names
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            # read in the image
            im = mpimg.imread(file)

            if not im is None:
                im_list.append((im, im_type))
    
    return im_list 


# This function is a combination of several functions to standardize the images (combination of 3)

# Firstly, we resize the images
def standardize_input(image):
    # resize to a specific value 
    standard_im = cv2.resize(image, (600, 600))
    return standard_im


# Secondly, we change day ~ 1 and night ~ 0
def encode(label):
    numerical_val = 0
    if(label=="day"):
        numerical_val = 1

    return numerical_val


def standardize(image_list):
    standard_list = []

    # Iterate through image-label pairs 
    for item in image_list:
        image = item[0]
        label = item[1]

        standardize_im = standardize_input(image)
        binary_label = encode(label)

        standard_list.append((standardize_im, binary_label))
    
    return standard_list


def plot_image_channels(image): 
    # convert to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # HSV channels 
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # plot 
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
    ax1.set_title("Original Image")
    ax1.imshow(image)
    ax2.set_title('H channel')
    ax2.imshow(h, cmap='gray')
    ax3.set_title('S channel')
    ax3.imshow(s, cmap='gray')
    ax4.set_title('V channel')
    ax4.imshow(v, cmap='gray')
    plt.show()


# classification 
def estimated_label(avg):
    if avg > 0.6:
        return 1
    else:
        return 0


# find the average brightness value in the image 
def avg_brightness(rgb_image):
    # convert to hsv
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    sum_brightness = np.sum(hsv[:, :, 2])
    area = 600*600
    avg = sum_brightness/area

    label_prediction = estimated_label(avg)

    return avg, label_prediction


# determine the accuracy 
def get_misclassified_images(test_images):
    misclassified_images = []

    for image in test_images: 

        im = image[0]
        true_label = image[1]

        # get prediction 
        avg, predicted_label = avg_brightness(im)

        if(predicted_label != true_label):
            misclassified_images.append((im, predicted_label, true_label))

    return misclassified_images


# accuracy 
def get_accuracy(test_images, misclassified):
    total = len(test_images)
    num_correct = total - len(misclassified)
    accuracy = num_correct / total 
    print("Accuracy: " + str(accuracy))
    print("Number of misclassified images = " + str(len(misclassified)) + "out of" + str(total))