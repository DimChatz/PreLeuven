import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks	import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import Model
import keras.backend as K
from keras.models import load_model
from skimage.draw import disk
from IPython.display import clear_output
import matplotlib.pyplot as plt


#####################################################################
### Creating the dataloading function for the input and the masks ###
#####################################################################

def dataloader(filepath, subset):
	# Initiliaze return arrays - input of shape = HYPERPARAMETER
	global size
	if subset=="train":
		size = 96
	elif subset=="val":
		size = 18	
	elif subset=="test":
		size = 18
	elif subset=="hope":
		size = 132
	input_data=np.zeros((size,256,256,1),dtype=np.uint8)
	#input_data=K.zeros_like(input_data)
	output_data=np.zeros((size,256,256,1),dtype=np.bool)
	#output_data=K.zeros_like(output_data)
	# Open file and create loop
	with open(str(filepath)+str(subset)+".txt", "r") as input_file:
		# Count to pass through the file
		count=0
		for line in input_file:
			line=line.split(" ")
			data=cv2.imread(filepath+str(line[0])+".jpg",0)
			data=cv2.resize(data, (256,256), interpolation=cv2.INTER_CUBIC)
			input_data[count,:,:,0]=data
			# Case of benevolent
			if line[3]=="B":
				# BE VERY CAREFUL WITH THIS DISK FUNCTION --- IT TAKES IN THE FORMAT (Y,X) NOT (X,Y)
				rr, cc = disk(((1024-(int(line[5])))/4, (int(line[4]))/4), int(line[6])/4, shape=(256,256))
				output_data[count,rr,cc,0]=255
			# Case of malevolent
			elif line[3]=="M":
				rr, cc = disk(((1024-(int(line[5])))/4, (int(line[4]))/4), int(line[6])/4, shape=(256,256))
				output_data[count,rr,cc,0]=255
			count=count+1
	input_data=input_data.astype(np.float32)
	output_data=output_data.astype(np.float32)
	return input_data, output_data

###########################
### Definition of model ###
###########################


def unet_model(num_classes, optimizer, loss_metric, metrics, sample_width, sample_height, lr=1e-5):
	inputs = Input((sample_width, sample_height, 1))
	
	# Downsampling
	conv1 = Conv2D(16, (1, 3), activation='relu', padding='same')(inputs)
	conv1 = Conv2D(16, (3, 1), activation='relu', padding='same')(conv1)
	conv1 = Conv2D(16, (1, 3), activation='relu', padding='same')(conv1)
	conv1 = Conv2D(16, (3, 1), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	drop1 = Dropout(0.3)(pool1)

	conv2 = Conv2D(32, (1, 3), activation='relu', padding='same')(drop1)
	conv2 = Conv2D(32, (3, 1), activation='relu', padding='same')(conv2)
	conv2 = Conv2D(32, (1, 3), activation='relu', padding='same')(conv2)
	conv2 = Conv2D(32, (3, 1), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	drop2 = Dropout(0.3)(pool2)

	conv3 = Conv2D(64, (1, 3), activation='relu', padding='same')(drop2)
	conv3 = Conv2D(64, (3, 1), activation='relu', padding='same')(conv3)
	conv3 = Conv2D(64, (1, 3), activation='relu', padding='same')(conv3)
	conv3 = Conv2D(64, (3, 1), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	drop3 = Dropout(0.3)(pool3)

	conv4 = Conv2D(128, (1, 3), activation='relu', padding='same')(drop3)
	conv4 = Conv2D(128, (3, 1), activation='relu', padding='same')(conv4)
	conv4 = Conv2D(128, (1, 3), activation='relu', padding='same')(conv4)
	conv4 = Conv2D(128, (3, 1), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	drop4 = Dropout(0.3)(pool4)

	conv5 = Conv2D(256, (1, 3), activation='relu', padding='same')(drop4)
	conv5 = Conv2D(256, (3, 1), activation='relu', padding='same')(conv5)
	conv5 = Conv2D(256, (1, 3), activation='relu', padding='same')(conv5)
	conv5 = Conv2D(256, (3, 1), activation='relu', padding='same')(conv5)
	pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
	drop5 = Dropout(0.3)(pool5)

	conv6 = Conv2D(512, (1, 3), activation='relu', padding='same')(drop5)
	conv6 = Conv2D(512, (3, 1), activation='relu', padding='same')(conv6)
	conv6 = Conv2D(512, (1, 3), activation='relu', padding='same')(conv6)
	conv6 = Conv2D(512, (3, 1), activation='relu', padding='same')(conv6)
	pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
	drop6 = Dropout(0.3)(pool6)

	conv7 = Conv2D(1024, (1, 3), activation='relu', padding='same')(drop6)
	conv7 = Conv2D(1024, (3, 1), activation='relu', padding='same')(conv7)
	conv7 = Conv2D(1024, (1, 3), activation='relu', padding='same')(conv7)
	conv7 = Conv2D(1024, (3, 1), activation='relu', padding='same')(conv7)
	pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)
	drop7 = Dropout(0.3)(pool7)

	
	# Bottleneck
	conv8 = Conv2D(2048, (1, 3), activation='relu', padding='same')(drop7)
	conv8 = Conv2D(2048, (3, 1), activation='relu', padding='same')(conv8)
	conv8 = Conv2D(2048, (1, 3), activation='relu', padding='same')(conv8)
	conv8 = Conv2D(2048, (3, 1), activation='relu', padding='same')(conv8)


	# Upsampling 
	up9 = concatenate([Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(conv8), conv7], axis=3)
	conv9 = Conv2D(1024, (1, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(1024, (3, 1), activation='relu', padding='same')(conv9)
	conv9 = Conv2D(1024, (1, 3), activation='relu', padding='same')(conv9)
	conv9 = Conv2D(1024, (3, 1), activation='relu', padding='same')(conv9)

	up10 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv9), conv6], axis=3)
	conv10 = Conv2D(512, (1, 3), activation='relu', padding='same')(up10)
	conv10 = Conv2D(512, (3, 1), activation='relu', padding='same')(conv10)
	conv10 = Conv2D(512, (1, 3), activation='relu', padding='same')(conv10)
	conv10 = Conv2D(512, (3, 1), activation='relu', padding='same')(conv10)

	up11 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv10), conv5], axis=3)
	conv11 = Conv2D(256, (1, 3), activation='relu', padding='same')(up11)
	conv11 = Conv2D(256, (3, 1), activation='relu', padding='same')(conv11)
	conv11 = Conv2D(256, (1, 3), activation='relu', padding='same')(conv11)
	conv11 = Conv2D(256, (3, 1), activation='relu', padding='same')(conv11)

	up12 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv11), conv4], axis=3)
	conv12 = Conv2D(128, (1, 3), activation='relu', padding='same')(up12)
	conv12 = Conv2D(128, (3, 1), activation='relu', padding='same')(conv12)
	conv12 = Conv2D(128, (1, 3), activation='relu', padding='same')(conv12)
	conv12 = Conv2D(128, (3, 1), activation='relu', padding='same')(conv12)

	up13 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv12), conv3], axis=3)
	conv13 = Conv2D(64, (1, 3), activation='relu', padding='same')(up13)
	conv13 = Conv2D(64, (3, 1), activation='relu', padding='same')(conv13)
	conv13 = Conv2D(64, (1, 3), activation='relu', padding='same')(conv13)
	conv13 = Conv2D(64, (3, 1), activation='relu', padding='same')(conv13)

	up14 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv13), conv2], axis=3)
	conv14 = Conv2D(32, (1, 3), activation='relu', padding='same')(up14)
	conv14 = Conv2D(32, (3, 1), activation='relu', padding='same')(conv14)
	conv14 = Conv2D(32, (1, 3), activation='relu', padding='same')(conv14)
	conv14 = Conv2D(32, (3, 1), activation='relu', padding='same')(conv14)

	up15 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv14), conv1], axis=3)
	conv15 = Conv2D(16, (1, 3), activation='relu', padding='same')(up15)
	conv15 = Conv2D(16, (3, 1), activation='relu', padding='same')(conv15)
	conv15 = Conv2D(16, (1, 3), activation='relu', padding='same')(conv15)
	conv15 = Conv2D(16, (3, 1), activation='relu', padding='same')(conv15)


	# Output
	conv16 = Conv2D(num_classes, (1, 1), padding='same', activation='sigmoid')(conv15)

	model = Model(inputs=[inputs], outputs=[conv16])

	model.compile(optimizer=optimizer(lr=lr), loss=loss_metric, metrics=metrics)
	return model

######################
### Util functions ###
######################

# Normalize images
def normalize(image):
	arr = image/255
	return arr

# Dice Coefficient to work with Tensorflow
def dice_coef(y_true, y_pred):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	#print(y_true_f.dtype)
	#print(y_pred_f.dtype)
	intersection = K.sum(y_true_f * y_pred_f)
	#print("intersection is",intersection)
	dice = (2. * intersection + 0.1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.1)
	return dice

def dice_coef_loss(y_true, y_pred):
	return -dice_coef(y_true, y_pred)

# Function to display images
def display(display_list):
	plt.figure(figsize=(15, 15))
	title = ['Input Image', 'True Mask', 'Predicted Mask']
	for i in range(len(display_list)):
		plt.subplot(1, len(display_list), i+1)
		plt.title(title[i])
		plt.imshow(display_list[i],cmap='gray')
		plt.axis('off')
		#print('The unique values are ',np.unique(display_list[i]))
	plt.show()


# Predictions of final images
def show_predictions(tensor_in,tensor_out, num=1):
	for i in range(num):
		i_input=tensor_in[i,:,:,:].reshape(1,tensor_in.shape[1],tensor_in.shape[2],tensor_in.shape[3])
		pred_mask = model.predict(i_input)
		display([tensor_in[i,:,:,0], tensor_out[i,:,:,0], pred_mask[0,:,:,0]])

# Callback to display masks as model trains
class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		show_predictions(train_input,train_output)
		print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

#####################
### MAIN PROGRAM ####
#####################


# HYPERPARAMETERS
# Filepath of datasets
filepath = "/home/tzikos/Downloads/train/"

# Variable to train model or load from wright
train_model=False




# Load datasets
train_input, train_output = dataloader(filepath, "train")
test_input, test_output = dataloader(filepath, "test")
val_input, val_output = dataloader(filepath, "val")
	
# Normalize images
train_input = normalize(train_input)
test_input = normalize(test_input)
val_input = normalize(val_input)

	
# Training option
if train_model==True:

	# Load model
	model = unet_model(num_classes=1, optimizer=Adam, loss_metric='binary_crossentropy', metrics=[dice_coef], sample_width=train_input.shape[1], sample_height=train_input.shape[2],lr=1e-4)
	model.summary()

	# Define callbacks
	callbacks = [
	ModelCheckpoint(filepath="/home/tzikos/Thesis/unet.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
	ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1, verbose=1, mode='min', min_lr=1e-8)
	#EarlyStopping(monitor="val_loss", patience=5, verbose=1)
	#DisplayCallback()
	]

	history = model.fit(x=train_input, y=train_output, validation_data=(val_input,val_output), batch_size=1, epochs=50, callbacks=callbacks)

	# Show masks for test set
	model = unet_model(num_classes=1, optimizer=Adam, loss_metric='binary_crossentropy', metrics=[dice_coef], sample_width=train_input.shape[1], sample_height=train_input.shape[2],lr=1e-4)
	model = load_model("/home/tzikos/Thesis/unet.h5", custom_objects = {"dice_coef": dice_coef})

	# Test model
	show_predictions(train_input, train_output, num=train_input.shape[0])
	
# Loading from weights option
else:
	# Initialize model from weights
	model = unet_model(num_classes=1, optimizer=Adam, loss_metric='binary_crossentropy', metrics=[dice_coef], sample_width=train_input.shape[1], sample_height=train_input.shape[2],lr=1e-4)
	model = load_model('/home/tzikos/Thesis/unet.h5', custom_objects = {"dice_coef": dice_coef})

	# Test model
	show_predictions(test_input, test_output, num=test_input.shape[0])