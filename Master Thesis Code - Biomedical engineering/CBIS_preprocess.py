#########################################################################
### Script for transforming the CBIS-DDSM dataset into legible format ###
#########################################################################

# The following analysis was done based on a 3 part preprocessing process I found at:
# https://towardsdatascience.com/can-you-find-the-breast-tumours-part-1-of-3-1473ba685036
# https://towardsdatascience.com/can-you-find-the-breast-tumours-part-2-of-3-1d43840707fc
# https://towardsdatascience.com/can-you-find-the-breast-tumours-part-3-of-3-388324241035


# Importing libraries
import numpy as np
import cv2
import os
import pydicom as dicom
import fnmatch

# Cropping borders
# l=left, r=right, u=upepr, d=down
def cropBorders(img, l=0.02, r=0.02, u=0.04, d=0.04):
    nrows, ncols = img.shape
    # Get the start and end rows and columns
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))
    cropped_img = img[u_crop:d_crop, l_crop:r_crop]
    return cropped_img

# MinMax Normalize
def minMaxNormalise(img):
    norm_img = (img - img.min()) / (img.max() - img.min())
    return norm_img

# Binarise Image to maxval
# thresh=threshold, maxval=value of binarization
def Binarise(img, thresh, maxval):
    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval
    return binarised_img

# Expand binarized mask to capture artefacts
# ksize is the size of the window to be applied
# operation can be "open" or "closed"
# erosion and then dilation is performed on the image in the case of opening,
# and the reverse in the case of closing. 
# When erosion is employed, the kernel is used on each pixel in order to set it to the minimum of its kernel pixel neighbors 
# while dilation sets it to the maximum.
# In the case of the opening small bright spots are removed, while in the case of closing, dark pixels become bright
def editMask(mask, ksize=(23, 23), operation="open"):
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)
    # Choose operation
    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Then dilate
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)
    return edited_mask

# Sorts contours based on area
def sortContoursByArea(contours, reverse=True):
    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)
    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]
    return sorted_contours, bounding_boxes

# Find largest contours
# top_x is the max it saves
# reverse if the way to sort the contours
def xLargestBlobs(mask, top_x=None, reverse=True):
    # Find all contours from binarised image (masked image).
    # Note: parts of the image that you want to get should be white.
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    n_contours = len(contours)
    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:
        if n_contours < top_x or top_x == None:
            top_x = n_contours
        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = sortContoursByArea(contours=contours, reverse=reverse)
        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_x]
        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)
        # Draw contours in X_largest_contours.
        X_large_blobs = cv2.drawContours(
            image=to_draw_on,  # Draw the contours on `to_draw_on`.
            contours=X_largest_contours,  # List of contours to draw.
            contourIdx=-1,  # Draw all contours in `contours`.
            color=1,  # Draw the contours in white.
            thickness=-1,  # Thickness of the contour lines.
        )
    return n_contours, X_large_blobs

# Apply mask to get the right image
def applyMask(img, mask):
    masked_img = img.copy()
    # where the mask=0 the returned image must also be 0
    masked_img[mask == 0] = 0
    return masked_img

# Contrast enhancement - CLAHE method - Contrast Limited Adaptive Histogram Equalization 
# the clip is the threshold above which the histogram is capped (see also thesis)
# tile is the window the method "sees" to process at each step
# For more info on clip: https://stackoverflow.com/questions/64576472/what-does-clip-limit-mean-exactly-in-opencv-clahe
def clahe(img, clip=2.0, tile=(8, 8)):
    # Normalize image
    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")
    # Apply clahe
    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)
    return clahe_img

# Combine above into Image preprocessing
def MammoPreprocess(img):
    # Step 1: Initial crop.
    cropped_img = cropBorders(img=img)
    crop_resized_img = cv2.resize(cropped_img, (512,512), cv2.INTER_CUBIC)
    # Step 2: Min-max normalise.
    norm_img = minMaxNormalise(img=crop_resized_img)
    # Step 3: Remove artefacts.
    binarised_img = Binarise(img=norm_img, thresh=1e-4, maxval=255)
    edited_mask = editMask(mask=binarised_img)
    _, xlargest_mask = xLargestBlobs(mask=edited_mask, top_x=1)
    masked_img = applyMask(img=norm_img, mask=xlargest_mask)
    # Step 4: CLAHE enhancement.
    clahe_img = clahe(img=masked_img)
    # Step 5: Min-max normalise.
    img_pre = minMaxNormalise(img=clahe_img)
    img_pre = img_pre*255
    return img_pre

# Combine into Mask preprocessing
def MaskPreprocess(mask):
    # Step 1: Initial crop.
    cropped_mask = cropBorders(img=mask)
    # Step 2: Resize
    mask = cv2.resize(cropped_mask, (512,512), cv2.INTER_CUBIC)
    # Step 3: Find contours
    _, largest_mask = xLargestBlobs(mask=mask,top_x=5)
    # Step 4: Threshold it
    _, mask = cv2.threshold(mask, 200,255, cv2.THRESH_BINARY )
    return mask


# Function of rewriting the images
def dataloader(filepath,subset):
    # Tracker
    count = 0
    # Initiliaze prev, these are for certain cases where two consecutive photos have different masks 
    # because they correspond to different tumors in the same image.
    # That is why you need to always remember the last photo you read
    prev_mask = 0
    prev_path = 0
    # Path to downloaded images - path to dataset
    down_path = '/home/bitu/Downloads/manifest-ZkhPvrLo5216730872708713142/CBIS-DDSM/'
    # Where to create the file with new masks and images
    home_path = '/home/bitu/Downloads/'+ subset
    # Main code
    with open(str(filepath), "r") as input_file:
        for line in input_file:
            # To ignore fist line of csv
            if count==0:
                count=count+1
                continue
            # To ignore " at each other line in csv file (open it and you will get it)
            if line[0] == '"':
                continue
            # Read file and place values
            line = line.split(',')
            patient_id = line[0]
            img_view = line[3]
            cancer_type = line[9]
            filepath_img = line[11]
            filepath_mask = line[13]
            # Process paths
            filepath_img = filepath_img.replace('"','')
            filepath_img = filepath_img.split('/')
            filepath_mask = filepath_mask.replace('"', '')
            filepath_mask = filepath_mask.split('/')
            img_path = down_path + str(filepath_img[0])
            msk_path = down_path + str(filepath_mask[0])
            # Read images
            for path, subdirs, files in os.walk(img_path):
                for name in files:
                    # Find paths
                    test_path =  os.path.join(path, name)
                    # Read Images
                    img = dicom.dcmread(test_path)
                    # Transform to numpy
                    pixel_array_numpy_img = img.pixel_array
                    img = pixel_array_numpy_img
                    # Preprocess images
                    img = MammoPreprocess(img)
                    # Save them for use in training and testing
                    image_path = home_path + 'images/' + filepath_img[0] + '.jpg'
                    cv2.imwrite(image_path, img)
            # Read masks
            for path, subdirs, files in os.walk(msk_path):
                for name in files:
                    # Find paths
                    test_path =  os.path.join(path, name)
                    # Read Images
                    mask = dicom.dcmread(test_path)
                     # Transform to numpy
                    pixel_array_numpy_mask = mask.pixel_array
                    if len(np.unique(pixel_array_numpy_mask))==2:
                        # Read images and NOT CROPPED
                        mask = pixel_array_numpy_mask
                        mask_path = home_path + 'masks/' + filepath_img[0] + '.jpg'
                        # Preprocess mask
                        mask = MaskPreprocess(mask)
                        # Check for same ID
                        if prev_path == mask_path:
                            print(mask_path)
                            # Sum current with previous mask in case of multiple tumors
                            summed_mask = np.add(mask, prev_mask)
                            # Threshold it
                            _, mask = cv2.threshold(src=summed_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
                        # Write path
                        cv2.imwrite(mask_path, mask)
                        # Prepare for next iteration - new prevs
                        prev_path = mask_path
                        prev_mask = mask
            count = count+1
    return 0


############
### MAIN ###
############

filepath_1 = '/home/bitu/Downloads/mass_case_description_train_set.csv'
filepath_2 = '/home/bitu/Downloads/mass_case_description_test_set.csv'

result = dataloader(filepath_1, 'train/')
result = dataloader(filepath_2, 'test/')
