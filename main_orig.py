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
from keras.utils import to_categorical
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder


#####################################################################
### Creating the dataloading function for the input and the masks ###
#####################################################################

def dataloader(filepath, subset):
	# Initiliaze return arrays
	global size
	if subset=="train":
		size = 96
	elif subset=="val":
		size = 18
	elif subset=="test":
		size = 18
	elif subset=="all":
		size = 314
	input_data=np.zeros((size*4,128,128,1),dtype=np.uint8)
	output_data=np.zeros((size*4,1,128,128),dtype=np.uint8)
	# Open file to read and create loop
	with open(str(filepath)+str(subset)+".txt", "r") as input_file:
		# Count to pass through the file
		count=0
		for line in input_file:
			line=line.split(" ")
			data=cv2.imread(filepath+str(line[0])+".jpg",0)
			data=cv2.resize(data, (128,128), interpolation=cv2.INTER_CUBIC)
			input_data[count,:,:,0]=data
			# Flipped
			input_data[size+count,:,:,0]=np.flip(input_data[count,:,:,0], 1)
			# Zoom
			cropped_in_img = input_data[count,25:102,13:115,0]
			input_data[size*2+count,:,:,0]  = cv2.resize(cropped_in_img, (128, 128), interpolation=cv2.INTER_CUBIC)
			# Zoom - flip
			input_data[size*3+count,:,:,0]=np.flip(input_data[size*2+count,:,:,0], 1)
			# Case of benevolent
			if line[3]=="B":
				# BE VERY CAREFUL WITH THIS DISK FUNCTION --- IT TAKES IN THE FORMAT (Y,X) NOT (X,Y)
				rr, cc = disk(((1024-(int(line[5])))/8, (int(line[4]))/8), int(line[6])/8, shape=(128,128))
				output_data[count,0,rr,cc]=1
				# Flipped
				output_data[size+count,0,:,:]=np.flip(output_data[count,0,:,:], 1)
				# Zoom
				cropped_img = output_data[count,0,25:102,13:115]
				output_data[size*2+count,0,:,:]  = cv2.resize(cropped_img, (128, 128), interpolation=cv2.INTER_CUBIC)
				#print(np.unique(output_data[size*2+count,0,:,:]))
				# Zoom - flip
				output_data[size*3+count,0,:,:]=np.flip(output_data[size*2+count,0,:,:], 1)
			# Case of malevolent
			elif line[3]=="M":
				rr, cc = disk(((1024-(int(line[5])))/8, (int(line[4]))/8), int(line[6])/8, shape=(128,128))
				output_data[count,0,rr,cc]=1
				# Flipped
				output_data[size+count,0,:,:]=np.flip(output_data[count,0,:,:], 1)
				# Zoom
				cropped_img = output_data[count,0,25:102,13:115]
				output_data[size*2+count,0,:,:]  = cv2.resize(cropped_img, (128, 128), interpolation=cv2.INTER_CUBIC)
				#print(np.unique(output_data[size*2+count,0,:,:]))
				# Zoom - flip
				output_data[size*3+count,0,:,:]=np.flip(output_data[size*2+count,0,:,:], 1)
			count=count+1
	input_data=input_data.astype(np.float32)
	# Reshape output_data in order to be used in the weights function
	output_data = output_data.reshape(output_data.shape[0],output_data.shape[2]*output_data.shape[3],output_data.shape[1])
	output_data[output_data>2] = 2
	output_data=output_data.astype(np.float32)
	print(np.unique(output_data))
	return input_data, output_data

###########################
### Definition of model ###
###########################

# Definition of block functions
def encoder_block(inputs,filter_size, kernel_size):
	conv = Conv2D(filter_size, (1,kernel_size), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv = Conv2D(filter_size, (kernel_size,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
	conv = BatchNormalization(axis=-1)(conv)
	conv = Conv2D(filter_size, (1,kernel_size), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
	conv = Conv2D(filter_size, (kernel_size,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
	conv = BatchNormalization(axis=-1)(conv)
	pool = MaxPooling2D(pool_size=(2, 2))(conv)
	drop = Dropout(0.3)(pool)
	return conv,drop

def bottleneck(inputs,filter_size, kernel_size):
	conv = Conv2D(filter_size, (1,kernel_size), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv = Conv2D(filter_size, (kernel_size,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
	conv = BatchNormalization(axis=-1)(conv)
	conv = Conv2D(filter_size, (1,kernel_size), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
	conv = Conv2D(filter_size, (kernel_size,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
	conv = BatchNormalization(axis=-1)(conv)
	return conv

def decoder_block(input1, input2,filter_size, kernel_size):
	up = concatenate([Conv2DTranspose(filter_size, (kernel_size,kernel_size), strides=(2, 2), padding='same')(input1), input2], axis=3)
	conv = Conv2D(filter_size, (1,kernel_size), activation='relu', padding='same', kernel_initializer='he_normal')(up)
	conv = Conv2D(filter_size, (kernel_size,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
	conv = BatchNormalization(axis=-1)(conv)
	conv = Conv2D(filter_size, (1,kernel_size), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
	conv = Conv2D(filter_size, (kernel_size,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
	conv = BatchNormalization(axis=-1)(conv)
	return conv

def inception_block(inputs, filter_size):
	conv1 = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1 = BatchNormalization(axis=-1)(conv1)

	conv2 = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv2 = Conv2D(filter_size, (1,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	conv2 = Conv2D(filter_size, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	conv2 = BatchNormalization(axis=-1)(conv2)
	
	conv3 = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv3 = Conv2D(filter_size, (1,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	conv3 = Conv2D(filter_size, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	conv3 = BatchNormalization(axis=-1)(conv3)
	conv3 = Conv2D(filter_size, (1,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	conv3 = Conv2D(filter_size, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	conv3 = BatchNormalization(axis=-1)(conv3)
	
	conv_all = concatenate([conv1,conv2,conv3],axis=3)
	conv_all = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_all)

	last = concatenate([inputs,conv_all],axis=3)
	return last

def dense_inception_block(inputs, filter_size):
	conv1 = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv1 = BatchNormalization(axis=-1)(conv1)
	
	conv2 = MaxPooling2D(pool_size=(2, 2))(inputs)
	conv2 = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
	conv2 = BatchNormalization(axis=-1)(conv2)

	conv3 = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv3 = Conv2D(filter_size, (1,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	conv3 = BatchNormalization(axis=-1)(conv3)
	conv3 = Conv2D(filter_size, (3,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
	conv3 = BatchNormalization(axis=-1)(conv3)
	
	conv_all = concatenate([conv1,conv2,conv3],axis=3)
	conv_all = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_all)

	last = concatenate([inputs,conv_all],axis=3)
	return last

def dense_sum(inputs, filter_size):
	forward1 = inception_block(inputs,64)
	down1 = 

def downsample_block(input, filter_size):
	conv1 = MaxPooling2D(pool_size=(2, 2))(inputs)

	conv2 = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv2 = Conv2D(filter_size, (3,3), strides=(2, 2), padding='same')(conv2)

	conv3 = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv3 = Conv2D(filter_size, (1,3), padding='same')(conv3)
	conv3 = Conv2D(filter_size, (3,1), padding='same')(conv3)
	conv3 = Conv2D(filter_size, (3,3), strides=(2, 2), padding='same')(conv3)

	conv_all = concatenate([conv1,conv2,conv3],axis=3)
	conv_all = Conv2D(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_all)
	return conv_all

def upsample_block(inputs, filter_size):
	conv1 = UpSampling2D(size=(2, 2))(inputs)

	conv2 = Conv2Transpose(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv2 = Conv2DTranspose(filter_size, (3,3), strides=(2, 2), padding='same')(conv2)

	conv3 = Conv2DTranspose(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
	conv3 = Conv2DTranspose(filter_size, (1,3), padding='same')(conv3)
	conv3 = Conv2DTranspose(filter_size, (3,1), padding='same')(conv3)
	conv3 = Conv2DTranspose(filter_size, (3,3), strides=(2, 2), padding='same')(conv3)

	conv_all = concatenate([conv1,conv2,conv3],axis=3)
	conv_all = Conv2DTranspose(filter_size, (1,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv_all)
	return conv_all

# Alternative
def DIUnet(num_classes, optimizer, loss_metric, metrics, sample_width, sample_height, lr=1e-5):

# Main model
def unet_model(num_classes, optimizer, loss_metric, metrics, sample_width, sample_height, lr=1e-5):
	inputs = Input((sample_width, sample_height, 1))
	
	# Downsampling
	conv1, drop1 = encoder_block(inputs,16,3)
	conv2, drop2 = encoder_block(drop1,32,3)
	conv3, drop3 = encoder_block(drop2,64,3)
	conv4, drop4 = encoder_block(drop3,128,3)
	conv5, drop5 = encoder_block(drop4,256,3)
	
	# Bottleneck
	conv6 = bottleneck(drop5,512,3)

	# Upsampling
	conv7 = decoder_block(conv6,conv5,256,2)
	conv8 = decoder_block(conv7,conv4,128,2)
	conv9 = decoder_block(conv8,conv3,64,2)
	conv10 = decoder_block(conv9,conv2,32,2)
	conv11 = decoder_block(conv10,conv1,16,2)


	# Output
	conv12 = Conv2D(num_classes, 1, padding='same', activation='sigmoid')(conv11)
	conv12 = Reshape((sample_width*sample_height,num_classes))(conv12)

	model = Model(inputs=[inputs], outputs=[conv12])

	model.compile(optimizer=optimizer(lr=lr), loss=loss_metric, metrics=metrics, sample_weight_mode="temporal")
	return model

######################
### Util functions ###
######################

# Normalize images
def normalize(image):
	arr = image/255
	return arr

# Dice Coefficient to work with Tensorflow
def dice_coef(y_true, y_pred, smooth=0.001):
  intersection = K.sum(y_true * y_pred)
  union = K.sum(y_true) + K.sum(y_pred)
  dice = (2. * intersection + smooth)/(union + smooth)
  return dice


def dice_coef_loss(y_true, y_pred,smooth=0.001):
	return -dice_coef(y_true, y_pred, smooth)

# Function to display images
def display(display_list):
	plt.figure(figsize=(15, 15))
	title = ['Input Image', 'True Mask', 'Predicted Mask']
	for i in range(len(display_list)):
		plt.subplot(1, len(display_list), i+1)
		plt.title(title[i])
		plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
		plt.axis('off')
	plt.show()


# Predictions of images
def show_predictions(tensor_in,tensor_out, num=5, size=314):
	for i in range(num):
		i_output = tensor_out.reshape((tensor_out.shape[0],int(np.sqrt(tensor_out.shape[1])),int(np.sqrt(tensor_out.shape[1])),tensor_out.shape[2]))
		i_input=tensor_in[i,:,:,:].reshape(1,tensor_in.shape[1],tensor_in.shape[2],tensor_in.shape[3])
		#print('The unique values for the ground truth in displaycallback are ',np.unique(i_output[i]))
		pred_mask = model.predict(i_input)
		pred_mask_test = create_mask(pred_mask)
		pred_mask_test = pred_mask_test.reshape((int(np.sqrt(pred_mask_test.shape[1])),int(np.sqrt(pred_mask_test.shape[1],)),pred_mask_test.shape[0]))
		display([tensor_in[i,:,:,:], i_output[i,:,:,:], pred_mask_test[:,:,:]])

# Callback to display masks as model trains at epoch end
class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		show_predictions(train_input,train_output)


def create_mask(pred_mask):
	pred_mask = np.argmax(pred_mask, axis = -1)
	return pred_mask

def iou(y_true, y_pred):
	smooth=0.001
	intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
	union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
	iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
	return iou

def iou_loss(y_true, y_preds):
	return -iou(y_pred, y_true)

# Create weights
def weight_calc(num_images, size_image, y_true, filepath, subset):
	num_classes = np.unique(y_true)
	# Necessary steps -> ask stack why
	labelencoder = LabelEncoder()
	train_masks_mod = y_true.reshape(-1,1)
	train_masks_mod = labelencoder.fit_transform(train_masks_mod)
	# Calculate weights
	class_weights=class_weight.compute_class_weight('balanced',	np.unique(train_masks_mod), train_masks_mod)
	# Pass weights to dictionary so as to be called  
	Class_Weights = {i : class_weights[i] for i in range(2)}
	# Create array to store weights
	# Open file and create loop to parse weights
	sample_weights = np.full((num_images, int(np.sqrt(size_image)), int(np.sqrt(size_image))), Class_Weights[0], dtype=np.uint8)
	with open(str(filepath)+str(subset)+".txt", "r") as input_file:
		# Count to pass through the file
		count=0
		for line in input_file:
			line=line.split(" ")
			# Case of benevolent
			if line[3]=="B":
				# BE VERY CAREFUL WITH THIS DISK FUNCTION --- IT TAKES IN THE FORMAT (Y,X) NOT (X,Y)
				rr, cc = disk(((1024-(int(line[5])))/8, (int(line[4]))/8), int(line[6])/8, shape=(128,128))
				sample_weights[count,rr,cc]= Class_Weights[1]
			# Case of malevolent
			elif line[3]=="M":
				rr, cc = disk(((1024-(int(line[5])))/8, (int(line[4]))/8), int(line[6])/8, shape=(128,128))
				sample_weights[count,rr,cc]=Class_Weights[1]
			count=count+1
	sample_weights = sample_weights.reshape(sample_weights.shape[0],sample_weights.shape[1]*sample_weights.shape[2])
	return sample_weights

#####################
### MAIN PROGRAM ####
#####################


# HYPERPARAMETERS
# Filepath of datasets
filepath = "/home/tzikos/Downloads/train/"


# Variable to train model or load from wright
train_model=True


# Load datasets
train_input, train_output = dataloader(filepath, "all")
test_input, test_output = dataloader(filepath, "test")
val_input, val_output = dataloader(filepath, "val")


# Normalize images
train_input = normalize(train_input)
test_input = normalize(test_input)
val_input = normalize(val_input)

# Create weights
weights = weight_calc(train_output.shape[0], train_output.shape[1], train_output, filepath, "all")

# Training option
if train_model==True:

	# Load model
	model = unet_model(num_classes=2, optimizer=Adam, loss_metric='sparse_categorical_crossentropy', metrics=['accuracy'], sample_width=train_input.shape[1], sample_height=train_input.shape[2], lr=1e-3)
	model.summary()

	# Define callbacks
	callbacks = [
	ModelCheckpoint(filepath="/home/tzikos/Thesis/unet.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
	ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1, verbose=1, mode='min', min_lr=1e-8),
	EarlyStopping(monitor="val_loss", patience=5, verbose=1),
	DisplayCallback()
	]

	show_predictions(train_input,train_output)
	history = model.fit(x=train_input, y=train_output, validation_data=(val_input,val_output), sample_weight=weights, batch_size=4, epochs=50, callbacks=callbacks)

# Loading from weights option
else:
	# Initialize model from weights
	model = unet_model(num_classes=3, optimizer=Adam, loss_metric='sparse_categorical_crossentropy', metrics=['accuracy'], sample_width=train_input.shape[1], sample_height=train_input.shape[2], lr=1e-3)
	model = load_model('/home/tzikos/Thesis/unet.h5')
	model = load_model('/home/tzikos/Thesis/unet.h5', custom_objects = {"dice_coef": dice_coef,"dice_coef_loss": dice_coef_loss})

	# Test model
	show_predictions(test_input, test_output, num=test_input.shape[0])