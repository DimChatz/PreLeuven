# Importing Libraries

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
import os


# Load data from CBIS-DDSM
# For ram reasons at some point I wanted the specified count of images
def dataloader_ddsm(filepath,size):
    count = 0
    # Array for reading into
    input_data=np.zeros((size,512,512,1),dtype=np.uint8)
    # Read
    for file in os.listdir(filepath):
        # Find paths
        if (count>5):
            test_path = os.path.join(filepath, file)
            data=cv2.imread(test_path,0)
            input_data[count,:,:,0] = data
        count = count + 1
        if (count>=11):
            break;
    return input_data


# Load data from MIAS
def dataloader_mias(filepath, subset):
    # Initiliaze return arrays
    global size
    if subset=="train":
        size = 312
    elif subset=="val":
        size = 18
    elif subset=="test":
        size = 18
    input_data=np.zeros((size*4,128,128,1),dtype=np.uint8)
    output_data=np.zeros((size*4,128,128,1),dtype=np.uint8)
    # Open file to read and create loop
    with open(str(filepath)+str(subset)+".txt", "r") as input_file:
        # Count to pass through the file
        count=0
        for line in input_file:
            line=line.split(" ")
            data=cv2.imread(filepath+str(line[0])+".pgm",0)
            data=cv2.resize(data, (128,128), interpolation=cv2.INTER_CUBIC)
            input_data[count,:,:,0]=data
            # Flipped
            input_data[size+count,:,:,0] = np.flip(input_data[count,:,:,0], 1)
            # Zoom
            cropped_in_img = input_data[count,14:113,14:113,0]
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
                cropped_img = output_data[count,14:113,14:113,0]
                output_data[size*2+count,:,:,0]  = cv2.resize(cropped_img, (128, 128), interpolation=cv2.INTER_CUBIC)
                # Zoom - flip
                output_data[size*3+count,:,:,0]=np.flip(output_data[size*2+count,:,:,0], 1)
            # Case of malevolent
            elif line[3]=="M":
                rr, cc = disk(((1024-(int(line[5])))/8, (int(line[4]))/8), int(line[6])/8, shape=(128,128))
                output_data[count,rr,cc,0]=1
                # Flipped
                output_data[size+count,:,:,0]=np.flip(output_data[count,:,:,0], 1)
                # Zoom
                cropped_img = output_data[count,14:113,14:113,0]
                output_data[size*2+count,:,:,0]  = cv2.resize(cropped_img, (128, 128), interpolation=cv2.INTER_CUBIC)
                # Zoom - flip
                output_data[size*3+count,:,:,0]=np.flip(output_data[size*2+count,:,:,0], 1)
            count=count+1
    input_data=input_data.astype(np.float32)
    input_data = np.asarray(input_data)
    # Reshape output_data in order to be used in the weights function - NO LONGER SUPPORTED
    output_data[output_data>1] = 1
    output_data=output_data.astype(np.float32)
    output_data = np.asarray(output_data)
    return input_data, output_data




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

# with a minus it can become loss (or with 1-dice)
def dice_coef_loss(y_true, y_pred,smooth=0.001):
    return -dice_coef(y_true, y_pred, smooth)

# Function to display images
def display(display_list):
    # image display size
    plt.figure(figsize=(15, 15))
    # image title
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

# Predictions of images
# tensor_in is input images
# tensor_out is masked input
# num is how many images to see when called
def show_predictions(tensor_in,tensor_out, num=1):
    for i in range(num):
        print("The input tensor has shape ", tensor_in[:,:,:,:].shape)
        pred_mask = unet.predict(tensor_in[:,:,:,:])
        pred_mask_test = create_mask(pred_mask)
        print("The mask initially has shape", pred_mask_test.shape)
        pred_mask_test = pred_mask_test[i,:,]
        print("The mask keeping a slice has shape", pred_mask_test.shape)
        print(np.sqrt(pred_mask_test.shape[0]))
        pred_mask_test = np.reshape(pred_mask_test, (int(np.sqrt(pred_mask_test.shape[0])),int(np.sqrt(pred_mask_test.shape[0])),1))
        print( "The reformed mask has shape ",pred_mask_test.shape)
        pred_mask_test = np.max(pred_mask_test) - pred_mask_test
        #pred_mask_test = pred_mask_test.reshape((pred_mask_test.shape[1],pred_mask_test.shape[2],1))
        display([tensor_in[i,:,:,:], tensor_out[i,:,:,:], pred_mask_test[:,:,:]])

# Create mask
def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis = -1)
    return pred_mask


# Callback to display masks as model trains at epoch end
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(test_input,test_masks)

# Intesection over union metric		
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union =  K.sum(y_true) +  K.sum(y_pred) - intersection
    x = (intersection + 1e-15) / (union + 1e-15)
    return x

# IoU as loss with a minus
def iou_loss(y_true, y_pred):
    return -iou(y_pred, y_true)


# HYPERPARAMETERS

#Filepath of datasets DDSM
filepath_train_img = "D:/Windows/torrents/manifest-ZkhPvrLo5216730872708713142/Processed/train/images/"
filepath_train_mask = "D:/Windows/torrents/manifest-ZkhPvrLo5216730872708713142/Processed/train/masks/"
filepath_test_img = "D:/Windows/torrents/manifest-ZkhPvrLo5216730872708713142/Processed/test/images/"
filepath_test_mask = "D:/Windows/torrents/manifest-ZkhPvrLo5216730872708713142/Processed/test/masks/"
# train mode to get the programm to train or to check and test
train_mode = False

# Get data DDSM
train_input = dataloader_ddsm(filepath_train_img,83)
train_masks = dataloader_ddsm(filepath_train_mask,83)
test_input = dataloader_ddsm(filepath_test_img,93)
test_masks = dataloader_ddsm(filepath_test_mask,93)

# Normalize images
train_input = normalize(train_input)
test_input = normalize(test_input)
test_masks = normalize(test_masks)
train_masks = normalize(train_masks)

print("The train input has dimensions ",train_input.shape)

if train_mode == True:
    # Load model
    unet = unet_model(num_classes = 2, optimizer = Adam, loss_metric = iou_loss, metrics = dice_coef, sample_width = test_input.shape[1], sample_height = test_input.shape[2], lr=1e-3)
    unet.compile(optimizer=Adam(lr=1e-3), loss = iou_loss,  metrics=[dice_coef])
    unet.summary()

 # Define callbacks
    callbacks = [
	# Callback to save the best model
    ModelCheckpoint(filepath="C:/Users/xatzo/Downloads/unet_new.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
	# Callback to reduce learning rate incase training plateaus
    ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1, verbose=1, mode='min', min_lr=1e-8),
	# To stop the training early if further training does nothing
    EarlyStopping(monitor="val_loss", patience=5, verbose=1),
	# Callback to check qualitative resutls at end of training
    DisplayCallback()
    ]

	# Display some data before initializing training
    show_predictions(test_input,test_masks)
    # Train
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

