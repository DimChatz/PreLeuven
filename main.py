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
from keras_unet_collection import models
import os
from tensorboard.plugins.hparams import api as hp


#####################################################################
### Creating the dataloading function for the input and the masks ###
#####################################################################

def dataloader_mias(filepath, subset):
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
	output_data=np.zeros((size*4,128,128,1),dtype=np.uint8)
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
			input_data[size+count,:,:,0] = np.flip(input_data[count,:,:,0], 1)
			# Zoom
			cropped_in_img = input_data[count,25:102,13:115,0]
			input_data[size*2+count,:,:,0]  = cv2.resize(cropped_in_img, (128, 128), interpolation=cv2.INTER_CUBIC)
			# Zoom - flip
			input_data[size*3+count,:,:,0]=np.flip(input_data[size*2+count,:,:,0], 1)
			# Case of benevolent
			if line[3]=="B":
				# BE VERY CAREFUL WITH THIS DISK FUNCTION --- IT TAKES IN THE FORMAT (Y,X) NOT (X,Y)
				rr, cc = disk(((1024-(int(line[5])))/8, (int(line[4]))/8), int(line[6])/8, shape=(128,128))
				output_data[count,rr,cc,0]=1
				# Flipped
				output_data[size+count,:,:,0]=np.flip(output_data[count,:,:,0], 1)
				# Zoom
				cropped_img = output_data[count,25:102,13:115,0]
				output_data[size*2+count,:,:,0]  = cv2.resize(cropped_img, (128, 128), interpolation=cv2.INTER_CUBIC)
				#print(np.unique(output_data[size*2+count,0,:,:]))
				# Zoom - flip
				output_data[size*3+count,:,:,0]=np.flip(output_data[size*2+count,:,:,0], 1)
			# Case of malevolent
			elif line[3]=="M":
				rr, cc = disk(((1024-(int(line[5])))/8, (int(line[4]))/8), int(line[6])/8, shape=(128,128))
				output_data[count,rr,cc,0]=1
				# Flipped
				output_data[size+count,:,:,0]=np.flip(output_data[count,:,:,0], 1)
				# Zoom
				cropped_img = output_data[count,25:102,13:115,0]
				output_data[size*2+count,:,:,0]  = cv2.resize(cropped_img, (128, 128), interpolation=cv2.INTER_CUBIC)
				#print(np.unique(output_data[size*2+count,0,:,:]))
				# Zoom - flip
				output_data[size*3+count,:,:,0]=np.flip(output_data[size*2+count,:,:,0], 1)
			count=count+1
	input_data=input_data.astype(np.float32)
	# Reshape output_data in order to be used in the weights function
	output_data[output_data>1] = 1
	output_data=output_data.astype(np.float32)
	print(np.unique(output_data))
	return input_data, output_data

def dataloader_ddsm(filepath,size):
	count = 0
	#array for reading into
	input_data=np.zeros((size,512,512,1),dtype=np.uint8)
	# Read
	for file in os.listdir(filepath):
		# Find paths
		test_path = os.path.join(filepath, file)
		data=cv2.imread(test_path,0)
		input_data[count,:,:,0] = data
		count = count + 1
	return input_data

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


# Main model
def unet_model(num_classes, optimizer, loss_metric, metrics, sample_width, sample_height, lr=1e-5):
    inputs = Input((sample_width, sample_height, 1))
    
    # Downsampling
    conv1, drop1 = encoder_block(inputs,64,3)
    conv2, drop2 = encoder_block(drop1,128,3)
    conv3, drop3 = encoder_block(drop2,256,3)
    conv4, drop4 = encoder_block(drop3,512,3)
    
    # Bottleneck
    conv5 = bottleneck(drop4,1024,3)

    # Upsampling
    conv6 = decoder_block(conv5,conv4,512,2)
    conv7 = decoder_block(conv6,conv3,256,2)
    conv8 = decoder_block(conv7,conv2,128,2)
    conv9 = decoder_block(conv8,conv1,64,2)

    # Output
    conv10 = Conv2D(num_classes, 1, padding='same', activation='softmax')(conv9)

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
def show_predictions(tensor_in,tensor_out, num=6, size=314):
	for i in range(num):
		i_input=tensor_in[i,:,:,:].reshape(1,tensor_in.shape[1],tensor_in.shape[2],tensor_in.shape[3])
		#print('The unique values for the ground truth in displaycallback are ',np.unique(i_output[i]))
		pred_mask = unet.predict(i_input)
		pred_mask_test = create_mask(pred_mask)
		pred_mask_test = pred_mask_test.reshape((pred_mask_test.shape[1],pred_mask_test.shape[2],1))
		display([tensor_in[i,:,:,:], tensor_out[i,:,:,:], pred_mask_test[:,:,:]])

# Callback to display masks as model trains at epoch end
class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		show_predictions(train_input,train_masks)


def create_mask(pred_mask):
	pred_mask = np.argmax(pred_mask, axis = -1)
	return pred_mask

def iou(y_true, y_pred):
	intersection = K.sum(y_true * y_pred)
	union =  K.sum(y_true) +  K.sum(y_pred) - intersection
	x = (intersection + 1e-15) / (union + 1e-15)
	return x

def iou_loss(y_true, y_pred):
	return -iou(y_pred, y_true)

#####################
### MAIN PROGRAM ####
#####################

# HYPERPARAMETERS
# Filepath of datasets
filepath_train_img = "/home/bitu/Downloads/train/images"
filepath_train_mask = "/home/bitu/Downloads/train/masks"
filepath_test_img = "/home/bitu/Downloads/test/images"
filepath_test_mask = "/home/bitu/Downloads/test/masks"

train_mode = False


# Get data
train_input = dataloader_ddsm(filepath_train_img,1231)
train_masks = dataloader_ddsm(filepath_train_mask,1231)
test_input = dataloader_ddsm(filepath_test_img,361)
test_masks = dataloader_ddsm(filepath_test_mask,361)

# Normalize images
train_input = normalize(train_input)
test_input = normalize(test_input)
test_masks = normalize(test_masks)
train_masks = normalize(train_masks)

if train_mode == True:
    # Load model
    unet = unet_model(num_classes = 2, optimizer = Adam, loss_metric = iou_loss, metrics = dice_coef, sample_width = test_input.shape[1], sample_height = test_input.shape[2], lr=1e-3)
    unet.compile(optimizer=Adam(lr=1e-3), loss = iou_loss,  metrics=[dice_coef])
    unet.summary()

    # Define callbacks
    callbacks = [
    ModelCheckpoint(filepath="C:/Users/xatzo/Downloads/unet_new.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
    ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1, verbose=1, mode='min', min_lr=1e-8),
    EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    DisplayCallback()
    ]

    show_predictions(test_input,test_masks)
    history = unet.fit(x=test_input, y=test_masks, 
        validation_data=(test_input,test_masks), 
    batch_size=1, epochs=50, callbacks=callbacks)
    

    # Test model
    show_predictions(test_input, test_masks, num=test_input.shape[0])

else:
    # Initialize model from weights
    unet = unet_model(num_classes=2, optimizer=Adam, loss_metric=iou_loss,
                       metrics=[dice_coef], sample_width=train_input.shape[1], sample_height=train_input.shape[2],lr=1e-3)
    unet = load_model('C:/Users/xatzo/Downloads/unet64_iou4_drop3.h5', custom_objects={"dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss, "iou": iou, "iou_loss": iou_loss})
    unet.summary()
    
    # Test model
    show_predictions(train_input, train_masks, num=5)
    show_predictions(test_input, test_masks, num=5)