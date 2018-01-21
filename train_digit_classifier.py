import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pickle


#Load data from csv file
data_path_train = os.path.join('/Users/johanneskoch/Documents/mnist/mnist_train.csv')
data_path_test = os.path.join('/Users/johanneskoch/Documents/mnist/mnist_test.csv')
train = pd.read_csv(data_path_train, delimiter = ',')
test = pd.read_csv(data_path_test, delimiter = ',')

#extract labels
train_labels = train["5"]
train_data = train.drop('5', axis=1)
test_labels = test["7"]
test_data = test.drop('7', axis=1)


#Look at one digit to get a glimpse of the data
sample_index = 3
tmp = np.array(test_data)
tmp = np.array(tmp[sample_index,:].reshape([28, 28]))

sns.heatmap(tmp)
plt.show()
print("Digit label: ", test_labels[sample_index])

#Train the SVM classifier

svc = SVC(kernel='poly', shrinking=True, random_state=0)



start = time.time()
print("Starting training.")
svc.fit(train_data, train_labels)
end = time.time()
print("Stopped training. Elapsed time: ", np.round(end - start), " seconds") #300 seconds is 5 minutes

#Check the accuracy
print("Accuracy on test set: ", svc.score(test_data, test_labels))
                                                                    #Could use un-used parts of training
                                                                    #set as test set. 5000 last samples
                                                                    #from test set contain noise and is
                                                                    #harder to classify.

#Accuracy seems to be ok for this task

#Save the model as a .sav file for easier shipping to other computers
filename = 'svc_model_tmp.sav'
pickle.dump(svc, open(filename, 'wb'))
