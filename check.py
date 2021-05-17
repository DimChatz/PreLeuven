import numpy as np
import cv2
import tensorflow as tf
import time
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks	import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import Model
import keras.backend as K
from keras.models import load_model
from skimage.draw import disk
from IPython.display import clear_output
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from keras_unet_collection import models

###########################
### Definition of model ###
###########################

######################
### Util functions ###
######################

# Normalize images
def normalize(input_image, input_mask):
	input_image = tf.cast(input_image, tf.float32) / 255.0
	input_mask -= 1
	return input_image, input_mask


# Function to display images
def display(display_list):
	plt.figure(figsize=(15, 15))
	title = ['Input Image', 'True Mask', 'Predicted Mask']
	for i in range(len(display_list)):
		plt.subplot(1, len(display_list), i+1)
		plt.title(title[i])
		print('The images for display are of shape',display_list[i].shape)
		plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
		plt.axis('off')
	plt.show()


# Predictions of final images
def show_predictions(dataset=None, num=3):
	if dataset:
		for image, mask in dataset.take(num):
			pred_mask = unet.predict(image)
			print(pred_mask.shape)
			pred_mask_test=create_mask(pred_mask)
			display([image[0], mask[0], pred_mask_test])
	else:
		pred_mask = unet.predict(sample_image[tf.newaxis, ...])
		print('The masks when out of the NN are of shape',pred_mask.shape, ' and when have unique values of ', np.unique(pred_mask[:,:,:,0])," ",np.unique(pred_mask[:,:,:,1]) ," ",np.unique(pred_mask[:,:,:,2]))
		display([sample_image, sample_mask, create_mask(pred_mask)])

class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		show_predictions()


# Load image
def load_image_train(datapoint):
	input_image = tf.image.resize(datapoint['image'], (128, 128))
	input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
	input_image = tf.image.rgb_to_grayscale(input_image)
	print('When loaded the images are of shape',input_image.shape)
	print('When loaded the masks are of shape',input_mask.shape)
	if tf.random.uniform(()) > 0.5:
		input_image = tf.image.flip_left_right(input_image)
		input_mask = tf.image.flip_left_right(input_mask)

	input_image, input_mask = normalize(input_image, input_mask)

	return input_image, input_mask

def load_image_test(datapoint):
	input_image = tf.image.resize(datapoint['image'], (128, 128))
	input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
	input_image = tf.image.rgb_to_grayscale(input_image)
	input_image, input_mask = normalize(input_image, input_mask)

	return input_image, input_mask


def create_mask(pred_mask):
	pred_mask = tf.argmax(pred_mask, axis=-1)
	pred_mask = pred_mask[..., tf.newaxis]
	print('The masked image created is of shape',pred_mask.shape)
	return pred_mask[0]

def iou(y_pred, y_true):
	num_classes = y_pred.shape[-1]
	y_pred = np.array([ np.argmax(y_pred, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)
	axes = (1,2) # W,H axes of each image
	intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
	union = np.sum(np.logical_or(y_pred, y_true), axis=axes)
	smooth = .001
	iou = (intersection + smooth) / (union + smooth)
	iou = np.mean(iou)
	return iou

def dice_coeff(y_pred, y_true):
	num_classes = y_pred.shape[-1]
	y_pred = np.array([ np.argmax(y_pred, axis=-1)==i for i in range(num_classes) ]).transpose(1,2,3,0)
	axes = (1,2) # W,H axes of each image
	intersection = np.sum(np.logical_and(y_pred, y_true), axis=axes)
	mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
	smooth = .001
	dice = 2 * (intersection + smooth)/(mask_sum + smooth)
	dice = np.mean(dice)
	return dice

def dice_coeff_loss(y_pred, y_true):
	return -dice_coeff(y_pred, y_true)

def iou_loss(y_pred, y_true):
	return -iou(y_pred, y_true)


#####################
### MAIN PROGRAM ####
#####################


# HYPERPARAMETERS
# Filepath of datasets
filepath = "/home/tzikos/Downloads/train/"

# Variable to train model or load from wright
train_model=True

# Download different dataset
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


#TRAIN_LENGTH = info.splits['train'].num_examples
TRAIN_LENGTH = 96
BATCH_SIZE = 16
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
print(STEPS_PER_EPOCH)
train = dataset['train'].map(load_image_train)
small_train = train.take(96)
test = dataset['test'].map(load_image_test)
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
small_train_dataset= small_train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
test_dataset = test.batch(BATCH_SIZE)


# Load model
unet = models.unet_2d((128, 128, 1), [32, 64, 128, 256], n_labels=3,
                      stack_num_down=2, stack_num_up=2,
                      activation='ReLU', output_activation='Softmax', 
                      batch_norm=True, pool=True, unpool=False, name='unet')
unet.compile(optimizer=Adam(lr=1e-3), loss='sparse_categorical_crossentropy')
unet.summary()

for image, mask in train.take(1):
	sample_image, sample_mask = image, mask
display([sample_image, sample_mask])


OUTPUT_CHANNELS = 3
show_predictions()


# Training option
if train_model==True:
	

	# Define callbacks
	callbacks = [
	ModelCheckpoint(filepath="/home/tzikos/Thesis/unet.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
	ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1, verbose=1, mode='min', min_lr=1e-8),
	#EarlyStopping(monitor="val_loss", patience=5, verbose=1)
	DisplayCallback()
	]

	EPOCHS = 20
	VAL_SUBSPLITS = 5
	VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

	model_history = unet.fit(small_train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          callbacks=callbacks)
	show_predictions(test_dataset, 3)
 
# Loading from weights option
else:
	# Initialize model from weights
	model = unet_model(num_classes=1, optimizer=Adam, loss_metric='binary_crossentropy', metrics=[dice_coef], sample_width=train_input.shape[1], sample_height=train_input.shape[2],lr=1e-4)
	model = load_model('/home/tzikos/Thesis/unet.h5', custom_objects = {"dice_coef": dice_coef})

	# Test model
	show_predictions(test_input, test_output, num=test_input.shape[0])