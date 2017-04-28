# Code to replicate Figure 3 in "On Weight Iniitalization in Deep Neural Networks"
# by Siddharth Krishna Kumar. The code uses the python packages keras, numpy and matplotlib
# The code makes no effort to optimize the runs. The point is merely
# to show that the initialization scheme proposed in the paper starts converging a lot faster
# than the Xavier initialization.
import keras
from keras.models import Sequential
from keras.initializers import VarianceScaling,glorot_normal
from keras.datasets import cifar10
from keras.layers.core import Flatten, Dense
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Conv2D, MaxPooling2D
from keras import metrics
from keras import initializers
import numpy as np
import matplotlib.pyplot as plt

# Setting some defaults regarding batch size and the optimizer; CIFAR 10 has 10 classes
batch_size = 32
num_classes = 10
epochs = 10
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()


print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def custom_initializer(factor):
	'''An initializer taking in a custom scaling factor. Based on the analysis in the paper,
	   for a neural network with sigmoid activations, the scaling factor should be approximately
	   3.6
	   
	   INPUT: Scaling factor to be used in the weight initialization
	   OUTPUT: An initializer to be fed to Keras
	'''
	return VarianceScaling(scale=factor,
                           mode='fan_in',
                           distribution='normal',
                           seed=None)

def model_for_testing(activataion_fn = 'sigmoid',initializer = 'glorot_normal'):
    ''' Neural Network with the architecture described in Figure 2 of the paper.
    
        INPUTS: Valid Keras activation function and initializer
	OUTPUT: Keras Model
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), kernel_initializer=initializer , padding='same',
    			 activation = activataion_fn, input_shape=x_train.shape[1:]))
    model.add(Conv2D(32, (3, 3), kernel_initializer=initializer , padding='same',
    			 activation = activataion_fn))
    model.add(Conv2D(32, (3, 3), kernel_initializer=initializer , padding='same',
    			 activation = activataion_fn))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(64, (3, 3), kernel_initializer=initializer , padding='same',
    			 activation = activataion_fn))
    model.add(Conv2D(64, (3, 3), kernel_initializer=initializer , padding='same',
    			 activation = activataion_fn))
    model.add(Conv2D(64, (3, 3), kernel_initializer=initializer , padding='same',
    			 activation = activataion_fn))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Conv2D(128, (3, 3), kernel_initializer=initializer , padding='same',
    			 activation = activataion_fn))
    model.add(Conv2D(128, (3, 3), kernel_initializer=initializer , padding='same',
    			 activation = activataion_fn))
    model.add(Conv2D(128, (3, 3), kernel_initializer=initializer , padding='same',
    			 activation = activataion_fn))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=[metrics.top_k_categorical_accuracy,'accuracy'])
    return model

# Running the model with the Xavier initialization
glorot_model = model_for_testing(activataion_fn = 'sigmoid',initializer = 'glorot_normal')
sigmoid_results_glorot = glorot_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)

# Running the model with the initializer provided in the paper
sid_initializer_exact = custom_initializer(3.6)
sid_model = model_for_testing(activataion_fn = 'sigmoid',initializer = sid_initializer_exact)
sigmoid_results_sid = sid_model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)


# Plotting the results
X = np.arange(1,epochs+1)
plt.figure(figsize=(10, 6), dpi=80)
plt.xlabel('Iteration Number',fontsize = 18)
plt.ylabel('Top 5 Categorical Accuracy',fontsize = 18)
plt.tick_params(labelsize=20)
plt.plot(X, sigmoid_results_sid.history['top_k_categorical_accuracy'], color="red", marker = "o", 
			markersize=10, label = "Recommended Initialization")
plt.plot(X, sigmoid_results_glorot.history['top_k_categorical_accuracy'], color="blue", marker = "o", 
			markersize=10, label = "Xavier Initialization")
plt.legend(loc='upper left')
plt.show()





