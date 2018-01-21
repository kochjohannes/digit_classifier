from PIL import Image as PImage
import PIL.ImageOps
import numpy as np
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


#This python script attempts to classify hand-written digits in photos. The classification is done using
#a SVM (Support Vector Machine) which was trained on the MNIST database training set and uses a polynomial
#kernel of order 3. An accuracy of ~97% was achieved on the MNIST test set.

#The key for this classifier to work is to pre-process the image of the digit to make it look as similar
#to the MNIST digits. The MNIST digits are contained in a 20x20 pixel grid and the whole image is 28x28 pixels.
#Furthermore, the digits in the MNIST database have their centre of mass in the middle of the image. They have
#also been filtered to remove background and noise, and has also been normalised.

#This script loads a pre-trained SVM model (trained using sklearn for minimum implementation), and focuses on
#the feature engineering. A high-level description of the script is as follows:
    #-Convert image to grayscale, normalise, and separate digit from background
    #-Find the digit and crop it out as a square
    #-Calculate mass centre and move it accordingly to get mass centre in middle of image
    #-Change the resolution and padd with (white) background
    #-Classify the digit

#Current limitations:
    #-Only one digit can be visible in image
    #-Static filtering of background

#Tips for higher accuracy on your own hand-written digits:
    #-Have only one digit visible
    #-Write the digit on a blank paper and avoid other dark things in the same image
    #-Use a dark pen and draw a fat digit (the higher contrast vs. background, the better)
    #-An up-close photo of the digit is a good thing, especially if the camera/image resolution to be
        #processed by this script is low.
    #-Try to avoid out-of focus, since that lowers the contrast and the sharpness of the digit's edges

#USAGE: Due to the rather large ML-model size (~50 MB), the training is done on
#your computer, requiring it to download the MNIST database. If your run the
#setup script, the database will be removed after the setup.
#From terminal, navigate to the folder with setup.sh and run "./setup.sh"
#Once the setup is completed, you can run the classifier as:
#"python3 digit_classifier.py /path/to/file.jpg"

path = sys.argv[1] #Path to image
#path = "/Users/johanneskoch/Downloads/digit_8 2.jpg"


img = PImage.open(path)
img = PIL.ImageOps.invert(img) #Invert image to make it similar to MNIST digits when greyscale
img = img.convert('LA') #convert to greyscale
#img = img.rotate(-90) #If image is rotated, might be if taken with iPhone

#img and tmp are similar to eachother, just different data types
#img is Image object, tmp is numpy array
#switching between the datatypes allows easy modification

#Normalise (to be exact: normalising and multiplying with 255), and filter background
tmp = np.asarray(img)[:,:,0] #[:,:,1] contains only values 255
tmp = tmp.reshape([1, img.height*img.width])
max_pixel = np.max(tmp); min_pixel = np.min(tmp)
tmp = tmp*(255/max_pixel)
for i in range(tmp.size):
    if(tmp[0,i] < 0.6*255): #Static, could perhaps be made dynamic by fitting a Gaussian to the histogram
                            #and do some clever reasoning
        tmp[0,i] = 0

tmp = tmp.reshape([img.height, img.width])

#Uncomment below for a visualization of the (hopefully) separated digit
#sns.heatmap(tmp)
#plt.show()

#Time to locate the digit
def find_digit_row(image, idx_range):
    two_in_row = False
    for i in idx_range:
        if(np.sum(tmp[i,:])>1000):  #this depends on the resolution and
                                    #should be made independent
            if(two_in_row): #bad variable name, I know...
                return i
            else:
                two_in_row = True

def find_digit_col(image, idx_range):
    two_in_row = False
    for i in idx_range:
        if(np.sum(tmp[:,i])>1000):  #this depends on the resolution and
                                    #should be made independent
            if(two_in_row):
                return i
            else:
                two_in_row = True


start_row = find_digit_row(tmp, range(tmp.shape[0])) - 1    #ascending i. Subtraction since two_in_row
                                                            #and want to capture the first row
end_row = find_digit_row(tmp, np.flip(range(tmp.shape[0]), axis=0)) + 1 #descending i
start_col = find_digit_col(tmp, range(tmp.shape[1])) - 1
end_col = find_digit_col(tmp, np.flip(range(tmp.shape[1]), axis=0)) + 1

#print(start_row, end_row, start_col, end_col)

#print(tmp.shape) #before cropping
tmp = tmp[start_row:end_row+1, start_col:end_col+1]
#print(tmp.shape) #after cropping

#To make the image square (which we will need in the end), we add empty background
#on both sides (i.e. 2 sides of the square)
padd_one_side = int(np.round(np.abs(tmp.shape[0]-tmp.shape[1])*0.5)) #No. of col/row to add to one side
extra_colrow = int(np.abs(2*(padd_one_side)-(tmp.shape[0]-tmp.shape[1])))   #Might need to add one more
                                                                            #since round

if(tmp.shape[0]>tmp.shape[1]):
    tmp = np.concatenate((np.zeros([tmp.shape[0], padd_one_side + extra_colrow]), tmp, np.zeros([tmp.shape[0], padd_one_side])), axis = 1)
if(tmp.shape[0]<tmp.shape[1]):
    tmp = np.concatenate((np.zeros([padd_one_side + extra_colrow, tmp.shape[1]]), tmp, np.zeros([padd_one_side, tmp.shape[1]])), axis = 0)

#print(tmp.shape) #after padding


#Uncomment below for visualisation of the digit
#sns.heatmap(tmp)
#plt.show()

img = PIL.Image.fromarray(tmp) #convert back to Image object for resizing
img = img.resize([20,20], PImage.ANTIALIAS) #convert to 20x20 pixel
tmp = np.asarray(img) #convert back to numpy array

#Time to move the centre of the digit to the centre of the image
#Start by placing the 20x20 image in the upper-left corner of the 28z28 image
tmp_28 = np.zeros([28,28])
tmp_28[0:20,0:20] = tmp
tmp = tmp_28 #tmp is now a 28x28

#Calculate the mass centre
total_mass = np.sum(np.sum(tmp))
cdf = 0
for i in range(tmp.shape[0]): #calculate centre along axis 0
    cdf += np.sum(tmp[i,:])
    if(cdf>=0.5*total_mass):
        mass_center_row = i
        break
cdf = 0
for i in range(tmp.shape[1]): #calculate centre along axis 1
    cdf += np.sum(tmp[:,i])
    if(cdf>=0.5*total_mass):
        mass_center_col = i
        break

#print(mass_center_row, mass_center_col)
diff_row = 13 - mass_center_row
diff_col = 13 - mass_center_col

tmp_28 = np.zeros([28,28])
#place the digit at the centre
tmp_28[diff_row:20+diff_row,+diff_col:20+diff_col] = tmp[0:20,0:20]

#Do one more filtering of the background to remove background level caused
#by the antialiasing when reducing image size
tmp_28 = tmp_28.reshape([1, 28*28])
for i in range(tmp_28.size):
    if(tmp_28[0,i] < 100):
        tmp_28[0,i] = 0

tmp_28 = tmp_28.reshape([28, 28])

#Uncomment below for visualisation of digit
#sns.heatmap(tmp_28)
#plt.show()


#Now it is finally time to do the classifying

X_tmp = tmp_28.reshape([1, 784])
#Load pre-trained model, can be changed easily by simply loading another pre-trained model
filename = "svc_model_tmp.sav"
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(X_tmp)

print("Predicted digit: ", result[0])
