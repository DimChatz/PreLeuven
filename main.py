import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import keras.backend as K
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
		size = 3
	input_data=np.zeros((size,512,512,1),dtype=np.uint8)
	#input_data=K.zeros_like(input_data)
	output_data=np.zeros((size,512,512,1),dtype=np.uint8)
	#output_data=K.zeros_like(output_data)
	# Open file and create loop
	with open(str(filepath)+str(subset)+".txt", "r") as input_file:
		# Count to pass through the file
		count=0
		for line in input_file:
			line=line.split(" ")
			data=cv2.imread(filepath+str(line[0])+".jpg",0)
			data=cv2.resize(data, (512,512), interpolation=cv2.INTER_CUBIC)
			input_data[count,:,:,0]=data
			# Case of benevolent
			if line[3]=="B":
				rr, cc = disk((int(line[4])/2, (1024-int(line[5]))/2), int(line[6])/2, shape=(512,512))
				output_data[count,rr,cc,0]=100
			# Case of malevolent
			elif line[3]=="M":
				rr, cc = disk((int(line[4])/2, (1024-int(line[5]))/2), int(line[6])/2, shape=(512,512))
				output_data[count,rr,cc,0]=200
			count=count+1
	input_data=input_data.astype(np.float32)
	output_data=output_data.astype(np.float32)
	print(input_data.dtype)
	return input_data, output_data

###########################
### Definition of model ###
###########################


def unet_model(optimizer, loss_metric, metrics, sample_width, sample_height, lr=1e-5):
	inputs = Input((sample_width, sample_height, 1))
	print(inputs.dtype)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	drop1 = Dropout(0.5)(pool1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	drop2 = Dropout(0.5)(pool2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(drop2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	drop3 = Dropout(0.3)(pool3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(drop3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	drop4 = Dropout(0.3)(pool4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(drop4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

	conv10 = Conv2D(1, (1, 1), activation='softmax')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])

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
	print("haha")
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	print("haha pre dice")
	intersection = K.sum(y_true_f * y_pred_f)
	dice = (2. * intersection + 0.00001) / (K.sum(y_true_f) + K.sum(y_pred_f) + 0.00001)
	#dice = K.sum(y_pred_f[y_true_f == 80/255] == 80/255) * 2.0 / (K.sum(y_pred_f[y_pred_f == 80/255] == 80/255) + K.sum(y_true_f[y_true_f == 80/255] == 80/255))
	print("haha after dice")
	#dice=dice+K.sum(y_pred_f[y_true_f == 160/255] == 160/255) * 2.0 / (K.sum(y_pred_f[y_pred_f == 160/255] == 160/255) + K.sum(y_true_f[y_true_f == 160/255] == 160/255))
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
		plt.imshow(display_list[i])
		plt.axis('off')
	plt.show()


# Predictions of final images
def show_predictions(tensor_in,tensor_out, num=1):
	for i in range(num):
		i_input=tensor_in[i,:,:,:].reshape(1,tensor_in.shape[1],tensor_in.shape[2],tensor_in.shape[3])
		i_output=tensor_out[i,:,:,:].reshape(1,tensor_out.shape[1],tensor_out.shape[2],tensor_out.shape[3])
		pred_mask = model.predict(i_input)
		display([tensor_in[i,:,:,0], tensor_out[i,:,:,0], pred_mask[i,:,:,0]])

# Callback to display masks as model trains
class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		show_predictions(train_input,train_output)
		print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

#####################
### MAIN PROGRAM ####
#####################

# Filepath of datasets
filepath = "/home/tzikos/Downloads/train/"
# Variable to train model or load from wright
train_model=True


# Training option
if train_model==True:
	# Load datasets
	train_input, train_output = dataloader(filepath, "hope")
	test_input, test_output = dataloader(filepath, "test")
	val_input, val_output = dataloader(filepath, "hope")
	train_input = normalize(train_input)
	test_input = normalize(test_input)
	val_input = normalize(val_input)
	val_output = normalize(val_output)
	train_output = normalize(train_output)
	test_output = normalize(test_output)
	

	# Load model
	model = unet_model(optimizer=Adam, loss_metric=dice_coef_loss, metrics=[dice_coef], sample_width=train_input.shape[1], sample_height=train_input.shape[2],lr=1e-5)
	history = model.fit(x=train_input, y=train_output, validation_data=(val_input,val_output), batch_size=1, epochs=2, callbacks=[DisplayCallback()])

	# Save weights
	model_filepath = '/home/tzikos/Desktop/thesis_DENSE-IN-UNET/unet_weights.h5'
	model.save(model_filepath)

	# Check results
	results = model.evaluate(test_input, test_output, batch_size=1)


# Loading from weights option
else:
	# Load dataset
	test_input, test_output = dataloader(filepath, "test")
	test_input = normalize(test_input)
	test_output = normalize(test_output)

	# Initialize model from weights
	model = keras.models.load_model('/home/tzikos/Desktop/thesis_DENSE-IN-UNET/unet_weights.h5')

	# Test model
	results = model.evaluate(test_input, test_output, batch_size=1)


