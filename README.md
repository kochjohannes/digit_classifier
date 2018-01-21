# Digit_classifer

This small project is focused on a python script that attempts to classify hand-written digits in photos. The classification is done using a SVM (Support Vector Machine) which was trained on the MNIST database training set and uses a polynomial
kernel of order 3. An accuracy of ~97% was achieved on the MNIST test set.

The key for this classifier to work is to pre-process the image of the digit to make it look as similar
to the MNIST digits. The MNIST digits are contained in a 20x20 pixel grid and the whole image is 28x28 pixels.
Furthermore, the digits in the MNIST database have their centre of mass in the middle of the image. They have
also been filtered to remove background and noise, and has also been normalised.

The classifying python script loads a pre-trained SVM model (trained using sklearn for minimum implementation), and focuses on
the feature engineering. A high-level description of the script is as follows:
- Convert image to grayscale, normalise, and separate digit from background
- Find the digit and crop it out as a square
- Calculate mass centre and move it accordingly to get mass centre in middle of image
- Change the resolution and padd with (white) background
- Classify the digit

## Current limitations:
- Only one digit can be visible in image
- Static filtering of background

## Tips for higher accuracy on your own hand-written digits:
- Have only one digit visible
- Write the digit on a blank paper and avoid other dark things in the same image
- Use a dark pen and draw a fat digit (the higher contrast vs. background, the better)
- An up-close photo of the digit is a good thing, especially if the camera/image resolution to be processed by this script is low.
- Try to avoid out-of focus, since that lowers the contrast and the sharpness of the digit's edges

## Usage 
Due to the rather large ML-model size (~50 MB), the training is done on
your computer, requiring it to download the MNIST database. If your run the
setup script, the database will be removed after the setup.
From terminal, navigate to the folder with setup.sh and run `./setup.sh`
Once the setup is completed, you can run the classifier as:
`python3 digit_classifier.py /path/to/file.jpg`
