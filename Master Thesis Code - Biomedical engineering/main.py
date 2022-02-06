# Importing libraries
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


# Dataloader for CBIS-DDSM dataset
def dataloader_ddsm(filepath,size):
    # tracker because my current ram has restrictions
    count = 0
    #array for reading into
    input_data=np.zeros((size,512,512,1),dtype=np.uint8)
    # Read
    for file in os.listdir(filepath):
        # Find paths
        if (count>=360) & (count<361):
            test_path = os.path.join(filepath, file)
            data=cv2.imread(test_path,0)
            input_data[count-360,:,:,0] = data
        count=count+1
    return input_data


# Dataloader for MIAS dataset
def dataloader_mias(filepath, subset):
    # Initiliaze return arrays
    global size
    # Pick for cases of train, val and test
    if subset=="train":
        size = 312
    elif subset=="val":
        size = 18
    elif subset=="test":
        size = 18
    # Input and output arrays
    input_data=np.zeros((size*4,128,128,1),dtype=np.uint8)
    output_data=np.zeros((size*4,128,128,1),dtype=np.uint8)
    # Open file to read and create loop
    with open(str(filepath)+str(subset)+".txt", "r") as input_file:
        # Count to pass through the file
        count=0
        for line in input_file:
            line=line.split(" ")
            data=cv2.imread(filepath+str(line[0])+".pgm",0)
            # Resize
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
    # Reshape output_data in order to be used in the weights function
    output_data[output_data>1] = 1
    output_data=output_data.astype(np.float32)
    output_data = np.asarray(output_data)
    return input_data, output_data


#######################################
### FUNCTIONS FOR NEURAL UNET model ###
#######################################
# Definition of block functions
def encoder_block(inputs,filter_size, kernel_size):
    # Convs in (1xn) and (nx1) to speed the process)
    conv = Conv2D(filter_size, (1,kernel_size), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(filter_size, (kernel_size,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    # Batchnorm for the vanishing gradient and regularization 
    conv = BatchNormalization(axis=-1)(conv)
    conv = Conv2D(filter_size, (1,kernel_size), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = Conv2D(filter_size, (kernel_size,1), activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    conv = BatchNormalization(axis=-1)(conv)
    # Dimnesionality reduction
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    # Dropout for prevention of overfitting
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


##################
### MAIN MODEL ###
##################
def unet_model(num_classes, optimizer, loss_metric, metrics, sample_width, sample_height, lr=1e-5):
    inputs = Input((sample_width, sample_height, 1))
    
    # Downsampling
    conv1, drop1 = encoder_block(inputs,32,3)
    conv2, drop2 = encoder_block(drop1,64,3)
    conv3, drop3 = encoder_block(drop2,128,3)
    conv4, drop4 = encoder_block(drop3,256,3)
    
    # Bottleneck
    conv5 = bottleneck(drop4,512,3)

    # Upsampling
    conv6 = decoder_block(conv5,conv4,256,2)
    conv7 = decoder_block(conv6,conv3,128,2)
    conv8 = decoder_block(conv7,conv2,64,2)
    conv9 = decoder_block(conv8,conv1,32,2)

    # Output
    conv10 = Conv2D(num_classes, 1, padding='same', activation='softmax')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=optimizer(lr=lr), loss=loss_metric, metrics=metrics)
    return model


#########################
### UTILITY FUNCTIONS ###
#########################
# Normalize images
def normalize(image):
    arr = image/255
    return arr

# Dice Coefficient metric to work with Tensorflow
def dice_coef(y_true, y_pred, smooth=0.001):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred)
    dice = (2. * intersection + smooth)/(union + smooth)
    return dice

# Dice made to loss function
def dice_coef_loss(y_true, y_pred,smooth=0.001):
    return -dice_coef(y_true, y_pred, smooth)

# Intersection over Union metric to work with Tensorflow
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union =  K.sum(y_true) +  K.sum(y_pred) - intersection
    x = (intersection + 1e-15) / (union + 1e-15)
    return x

# IoU as loss
def iou_loss(y_true, y_pred):
    return -iou(y_pred, y_true)

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
def show_predictions(tensor_in,tensor_out, num=1):
        print("The input tensor has shape ", tensor_in[:,:,:,:].shape)
        pred_mask = unet.predict(tensor_in[:,:,:,:])
        pred_mask_test = create_mask(pred_mask)
        print("The mask initially has shape", pred_mask_test.shape)
        for i in range(num):
            pred_mask_test_now = pred_mask_test[i,:,]
            print("The mask keeping a slice has shape", pred_mask_test_now.shape)
            pred_mask_test_now = np.reshape(pred_mask_test_now, (int(np.sqrt(pred_mask_test_now.shape[0])),int(np.sqrt(pred_mask_test_now.shape[0])),1))
            print( "The reformed mask has shape ",pred_mask_test_now.shape)
            pred_mask_test_now = np.max(pred_mask_test_now) - pred_mask_test_now
            display([tensor_in[i,:,:,:], tensor_out[i,:,:,:], pred_mask_test_now[:,:,:]])

# Callback to display masks as model trains at epoch end
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(test_input,test_masks)

# Create mask of the image
def create_mask(pred_mask):
    pred_mask = np.argmax(pred_mask, axis = -1)
    return pred_mask


############
### MAIN ###
############

#Filepath of datasets DDSM
filepath_train_img = "D:/Windows/torrents/manifest-ZkhPvrLo5216730872708713142/Processed/train/images/"
filepath_train_mask = "D:/Windows/torrents/manifest-ZkhPvrLo5216730872708713142/Processed/train/masks/"
filepath_test_img = "D:/Windows/torrents/manifest-ZkhPvrLo5216730872708713142/Processed/test/images/"
filepath_test_mask = "D:/Windows/torrents/manifest-ZkhPvrLo5216730872708713142/Processed/test/masks/"

#Picking mode train or testing
train_mode = False

# Get data DDSM
train_input = dataloader_ddsm(filepath_train_img,10)
train_masks = dataloader_ddsm(filepath_train_mask,10)
test_input = dataloader_ddsm(filepath_test_img,1)
test_masks = dataloader_ddsm(filepath_test_mask,1)

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
    # Save best model
    ModelCheckpoint(filepath="C:/Users/xatzo/Downloads/unet_new.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
    # Reduce learning rate so the training can be unstuck
    ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1, verbose=1, mode='min', min_lr=1e-8),
    # Early stopping for saving time in training
    EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    # My custom callback to display resulting masks at end of epochs
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
                       metrics=[dice_coef], sample_width=test_input.shape[1], sample_height=test_input.shape[2],lr=1e-3)
    unet = load_model('C:/Users/xatzo/Downloads/unet64_iou4_drop3.h5', custom_objects={"dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss, "iou": iou, "iou_loss": iou_loss})
    unet.summary()
    
    # Test model
    show_predictions(test_input, test_masks,num=1)

