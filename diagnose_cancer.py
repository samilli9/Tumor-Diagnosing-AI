# Cancer Diagnosing AI 


##########
# Imports
##########
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


#################
# Read in Data
#################
data = pd.read_csv('tumor_data.csv')



###################################
# Define x Features and y Output
###################################

# The input x is all of the tumor features
x = data.drop(columns=["diagnosis(1=m, 0=b)"])


# The output y is a classification of whether a cancerous tumor is malignant or beign
y = data["diagnosis(1=m, 0=b)"]



##############################################################
# Split our Data 80/20 for Training and Testing respectively
##############################################################

# 80% of our data will be for training, 20% for testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)



######################
# Build a Neural Net
######################
model = tf.keras.models.Sequential()

# Add layers

# Define our input layer, which will output 256 neurons
# Sigmoid func will help to squish neural net values between 0 & 1, those output values will be used for diagnosis
model.add(tf.keras.layers.Dense(256, input_shape = x_train.shape, activation = "sigmoid")) 

# Second layer
model.add(tf.keras.layers.Dense(256, activation = "sigmoid"))

# Output layer
model.add(tf.keras.layers.Dense(1, activation = "sigmoid")) 



######################
# Compile the Model
######################

# Adam fine tunes the weights of the algorithm to fit the data
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit(x_train, y_train, epochs = 1000)

# Using a thousand epochs is overkill however it works with this relatively small dataset
# Also when it comes to cancer better to be safe than sorry!

model.evaluate(x_test,y_test)