import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import re
#in this method we use "Sklearn" to estimate weight vector for logistic Regression
patch_size = 16  # each patch is 16*16 pixels
n = 100  #Number of images that we want to use for train. It can vary from 1 to 100.

#Four functions below are helper functions

# Loading images by adress of file
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

#converting r,g,b for each pixel from intercal(0,1) to integer 0 to 255
def img_float_to_uint8(img):
    print(type(img))
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

#Cropping image with size w and h
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches



# Loding images for training, it includes satellite images and groundtruth images.
root_dir = "Datasets/training/"
image_dir = root_dir + "images/"
files = os.listdir(image_dir)
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]
gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " images")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]



#This part had written for rotating images 45 degrees, since it does not
#improve the result, we highlighten it.
'''
imgs_rot=[]
gt_imgs_rot=[]
teta=45
w = 113*np.sin(np.radians(teta))*np.cos(np.radians(teta))
w = 64
for i in range(n):
   #teta=45
   #w = 135*np.sin(np.radians(teta))*np.cos(np.radians(teta))
   imgs_rot.append(Image.open("training/images/" + "satImage_%.3d" % (i+1) + ".png"))
   gt_imgs_rot.append(Image.open("training/groundtruth/" + "satImage_%.3d" % (i+1) + ".png"))
   imgs_rot[i] = imgs_rot[i].rotate(teta)
   imgs_rot[i] = np.array(imgs_rot[i].crop((w,w,400-w,400-w)))
   #imgs_rot[i] = np.array(imgs_rot[i].resize((400,400))) 
   gt_imgs_rot[i] = gt_imgs_rot[i].rotate(teta)
   gt_imgs_rot[i] = np.array(gt_imgs_rot[i].crop((w,w,400-w,400-w)))
   #gt_imgs_rot[i] = np.array(gt_imgs_rot[i].resize((400,400)))
'''


img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

#This part as well as above useful for rotation.
'''
img_patches_rot = [img_crop(imgs_rot[i], patch_size, patch_size) for i in range(n)]
gt_patches_rot = [img_crop(gt_imgs_rot[i], patch_size, patch_size) for i in range(n)]

# Linearize list of patches
img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
img_patches_rot = np.asarray([img_patches_rot[i][j] for i in range(len(img_patches_rot)) for j in range(len(img_patches_rot[i]))])
gt_patches_rot =  np.asarray([gt_patches_rot[i][j] for i in range(len(gt_patches_rot)) for j in range(len(gt_patches_rot[i]))])

img_patches_all = np.concatenate((img_patches,img_patches_rot))
gt_patches_all = np.concatenate((gt_patches,gt_patches_rot))
'''

# Linearize list of patches
img_patches_all = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
gt_patches_all = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

#This part is "ORIGINAL" and deeply calculated by the team.
#We tried to extract feautures by our mathematical knowledge.
#The core part of this model is the function below.
#In the function "tons_of_features" we generate more than 140 features
#to understand the road more accurately.
#Most of features depend on relation between eigenvalue of each patch(each patch has three matrix for red,green,blue)

#We explained some of the features in the report, and why we picked them.
def tons_of_features(img):
    trrg = np.trace((img[:, :, 0] + img[:, :, 1]) / 2)
    trrb = np.trace((img[:, :, 0] + img[:, :, 2]) / 2)
    trgb = np.trace((img[:, :, 1] + img[:, :, 2]) / 2)
    trrgb = np.trace((img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3)

    detrg = np.linalg.det((img[:, :, 0] + img[:, :, 1]) / 2)
    detrb = np.linalg.det((img[:, :, 0] + img[:, :, 2]) / 2)
    detgb = np.linalg.det((img[:, :, 1] + img[:, :, 2]) / 2)
    detrgb = np.linalg.det((img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3)
    detrgb2 = np.linalg.det(((img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3) ** 2)
    detrgb3 = np.linalg.det(((img[:, :, 0] + img[:, :, 1] + img[:, :, 2]) / 3) ** 3)

    det_r_r = np.linalg.det(img[:, :, 0] ** 2)
    det_g_g = np.linalg.det(img[:, :, 1] ** 2)
    det_b_b = np.linalg.det(img[:, :, 2] ** 2)
    det_r_r_r = np.linalg.det(img[:, :, 0] ** 3)
    det_g_g_g = np.linalg.det(img[:, :, 1] ** 3)
    det_g_g_g1 = np.linalg.det(img[:, :, 1] ** 3)
    det_g_g_g_g = np.linalg.det(img[:, :, 1] ** 4)
    det_b_b_b = np.linalg.det(img[:, :, 2] ** 3)

    det_r = np.linalg.det(img[:, :, 0])
    det_g = np.linalg.det(img[:, :, 1])
    det_b = np.linalg.det(img[:, :, 2])
    det_r_g = np.linalg.det(img[:, :, 0] @ img[:, :, 1])
    det_r_b = np.linalg.det(img[:, :, 0] @ img[:, :, 2])
    det_g_b = np.linalg.det(img[:, :, 1] @ img[:, :, 2])
    det_r_g_b = np.linalg.det(img[:, :, 0] @ img[:, :, 1] @ img[:, :, 2])

    tr_r = np.trace(img[:, :, 0])
    tr_g = np.trace(img[:, :, 1])
    tr_b = np.trace(img[:, :, 2])
    tr_r_g = np.trace(img[:, :, 0] @ img[:, :, 1])
    tr_r_b = np.trace(img[:, :, 0] @ img[:, :, 2])
    tr_g_b = np.trace(img[:, :, 1] @ img[:, :, 2])
    tr_r_g_b = np.trace(img[:, :, 0] @ img[:, :, 1] @ img[:, :, 2])

    r = np.mean(img[:, :, 0])
    g = np.mean(img[:, :, 1])
    b = np.mean(img[:, :, 2])
    var_r = np.var(img[:, :, 0])
    var_g = np.var(img[:, :, 1])
    var_b = np.var(img[:, :, 2])
    rgb = np.mean(img)
    var_rgb = np.var(img)

    rg = (np.mean(img[:, :, 0]) + np.mean(img[:, :, 1])) / 2
    rb = (np.mean(img[:, :, 0]) + np.mean(img[:, :, 2])) / 2
    gb = (np.mean(img[:, :, 1]) + np.mean(img[:, :, 2])) / 2
    var_rg = np.var(((img[:, :, 0]) + (img[:, :, 1])) / 2)
    var_rb = np.var(((img[:, :, 0]) + (img[:, :, 2])) / 2)
    var_gb = np.var(((img[:, :, 1]) + (img[:, :, 2])) / 2)

    r_g = np.mean(img[:, :, 0] @ img[:, :, 1])
    g_r = np.mean(img[:, :, 1] @ img[:, :, 0])
    r_b = np.mean(img[:, :, 0] @ img[:, :, 2])
    b_r = np.mean(img[:, :, 2] @ img[:, :, 0])
    g_b = np.mean(img[:, :, 1] @ img[:, :, 2])
    b_g = np.mean(img[:, :, 2] @ img[:, :, 1])
    vv_r_g = np.var(img[:, :, 0] @ img[:, :, 1])
    vv_g_r = np.var(img[:, :, 1] @ img[:, :, 0])
    vv_r_b = np.var(img[:, :, 0] @ img[:, :, 2])
    vv_b_r = np.var(img[:, :, 2] @ img[:, :, 0])
    vv_g_b = np.var(img[:, :, 1] @ img[:, :, 2])
    vv_b_g = np.var(img[:, :, 2] @ img[:, :, 1])
    r_r = np.mean(img[:, :, 0] @ img[:, :, 0])
    g_g = np.mean(img[:, :, 1] @ img[:, :, 1])
    b_b = np.mean(img[:, :, 2] @ img[:, :, 2])
    var_r_r = np.var(img[:, :, 0] @ img[:, :, 0])
    var_g_g = np.var(img[:, :, 1] @ img[:, :, 1])
    var_b_b = np.var(img[:, :, 2] @ img[:, :, 2])
    #
    r_g1 = np.mean(img[:, :, 0] * img[:, :, 1])
    g_r1 = np.mean(img[:, :, 1] * img[:, :, 0])
    r_b1 = np.mean(img[:, :, 0] * img[:, :, 2])
    b_r1 = np.mean(img[:, :, 2] * img[:, :, 0])
    g_b1 = np.mean(img[:, :, 1] * img[:, :, 2])
    b_g1 = np.mean(img[:, :, 2] * img[:, :, 1])
    vv_r_g1 = np.var(img[:, :, 0] * img[:, :, 1])
    vv_g_r1 = np.var(img[:, :, 1] * img[:, :, 0])
    vv_r_b1 = np.var(img[:, :, 0] * img[:, :, 2])
    vv_b_r1 = np.var(img[:, :, 2] * img[:, :, 0])
    vv_g_b1 = np.var(img[:, :, 1] * img[:, :, 2])
    vv_b_g1 = np.var(img[:, :, 2] * img[:, :, 1])
    r_r1 = np.mean(img[:, :, 0] * img[:, :, 0])
    g_g1 = np.mean(img[:, :, 1] * img[:, :, 1])
    b_b1 = np.mean(img[:, :, 2] * img[:, :, 2])
    var_r_r1 = np.var(img[:, :, 0] * img[:, :, 0])
    var_g_g1 = np.var(img[:, :, 1] * img[:, :, 1])
    var_b_b1 = np.var(img[:, :, 2] * img[:, :, 2])
    #

    r_g_b = np.mean(img[:, :, 0] @ img[:, :, 1] @ img[:, :, 2])
    r_b_g = np.mean(img[:, :, 0] @ img[:, :, 2] @ img[:, :, 1])
    g_r_b = np.mean(img[:, :, 1] @ img[:, :, 0] @ img[:, :, 2])
    g_b_r = np.mean(img[:, :, 1] @ img[:, :, 2] @ img[:, :, 0])
    b_r_g = np.mean(img[:, :, 2] @ img[:, :, 0] @ img[:, :, 1])
    b_g_r = np.mean(img[:, :, 2] @ img[:, :, 1] @ img[:, :, 0])
    vv_r_g_b = np.var(img[:, :, 0] @ img[:, :, 1] @ img[:, :, 2])
    vv_r_b_g = np.var(img[:, :, 0] @ img[:, :, 2] @ img[:, :, 1])
    vv_g_r_b = np.var(img[:, :, 1] @ img[:, :, 0] @ img[:, :, 2])
    vv_g_b_r = np.var(img[:, :, 1] @ img[:, :, 2] @ img[:, :, 0])
    vv_b_r_g = np.var(img[:, :, 2] @ img[:, :, 0] @ img[:, :, 1])
    vv_b_g_r = np.var(img[:, :, 2] @ img[:, :, 1] @ img[:, :, 0])

    #
    r_g_b1 = np.mean(img[:, :, 0] * img[:, :, 1] * img[:, :, 2])
    r_b_g1 = np.mean(img[:, :, 0] * img[:, :, 2] * img[:, :, 1])
    g_r_b1 = np.mean(img[:, :, 1] * img[:, :, 0] * img[:, :, 2])
    g_b_r1 = np.mean(img[:, :, 1] * img[:, :, 2] * img[:, :, 0])
    b_r_g1 = np.mean(img[:, :, 2] * img[:, :, 0] * img[:, :, 1])
    b_g_r1 = np.mean(img[:, :, 2] * img[:, :, 1] * img[:, :, 0])
    vv_r_g_b1 = np.var(img[:, :, 0] * img[:, :, 1] * img[:, :, 2])
    vv_r_b_g1 = np.var(img[:, :, 0] * img[:, :, 2] * img[:, :, 1])
    vv_g_r_b1 = np.var(img[:, :, 1] * img[:, :, 0] * img[:, :, 2])
    vv_g_b_r1 = np.var(img[:, :, 1] * img[:, :, 2] * img[:, :, 0])
    vv_b_r_g1 = np.var(img[:, :, 2] * img[:, :, 0] * img[:, :, 1])
    vv_b_g_r1 = np.var(img[:, :, 2] * img[:, :, 1] * img[:, :, 0])
    #

    rg_rb = np.mean((img[:, :, 0] + img[:, :, 1]) @ (img[:, :, 0] + img[:, :, 2])) / 4
    rb_rg = np.mean((img[:, :, 0] + img[:, :, 2]) @ (img[:, :, 0] + img[:, :, 1])) / 4
    rg_gb = np.mean((img[:, :, 0] + img[:, :, 1]) @ (img[:, :, 1] + img[:, :, 2])) / 4
    gb_rg = np.mean((img[:, :, 1] + img[:, :, 2]) @ (img[:, :, 0] + img[:, :, 1])) / 4
    rb_gb = np.mean((img[:, :, 0] + img[:, :, 2]) @ (img[:, :, 1] + img[:, :, 2])) / 4
    gb_rb = np.mean((img[:, :, 1] + img[:, :, 2]) @ (img[:, :, 0] + img[:, :, 2])) / 4
    vv_rg_rb = np.var((img[:, :, 0] + img[:, :, 1]) @ (img[:, :, 0] + img[:, :, 2])) / 4
    vv_rb_rg = np.var((img[:, :, 0] + img[:, :, 2]) @ (img[:, :, 0] + img[:, :, 1])) / 4
    vv_rg_gb = np.var((img[:, :, 0] + img[:, :, 1]) @ (img[:, :, 1] + img[:, :, 2])) / 4
    vv_gb_rg = np.var((img[:, :, 1] + img[:, :, 2]) @ (img[:, :, 0] + img[:, :, 1])) / 4
    vv_rb_gb = np.var((img[:, :, 0] + img[:, :, 2]) @ (img[:, :, 1] + img[:, :, 2])) / 4
    vv_gb_rb = np.var((img[:, :, 1] + img[:, :, 2]) @ (img[:, :, 0] + img[:, :, 2])) / 4

    #
    rg_rb1 = np.mean((img[:, :, 0] + img[:, :, 1]) * (img[:, :, 0] + img[:, :, 2])) / 4
    rb_rg1 = np.mean((img[:, :, 0] + img[:, :, 2]) * (img[:, :, 0] + img[:, :, 1])) / 4
    rg_gb1 = np.mean((img[:, :, 0] + img[:, :, 1]) * (img[:, :, 1] + img[:, :, 2])) / 4
    gb_rg1 = np.mean((img[:, :, 1] + img[:, :, 2]) * (img[:, :, 0] + img[:, :, 1])) / 4
    rb_gb1 = np.mean((img[:, :, 0] + img[:, :, 2]) * (img[:, :, 1] + img[:, :, 2])) / 4
    gb_rb1 = np.mean((img[:, :, 1] + img[:, :, 2]) * (img[:, :, 0] + img[:, :, 2])) / 4
    vv_rg_rb1 = np.var((img[:, :, 0] + img[:, :, 1]) * (img[:, :, 0] + img[:, :, 2])) / 4
    vv_rb_rg1 = np.var((img[:, :, 0] + img[:, :, 2]) * (img[:, :, 0] + img[:, :, 1])) / 4
    vv_rg_gb1 = np.var((img[:, :, 0] + img[:, :, 1]) * (img[:, :, 1] + img[:, :, 2])) / 4
    vv_gb_rg1 = np.var((img[:, :, 1] + img[:, :, 2]) * (img[:, :, 0] + img[:, :, 1])) / 4
    vv_rb_gb1 = np.var((img[:, :, 0] + img[:, :, 2]) * (img[:, :, 1] + img[:, :, 2])) / 4
    vv_gb_rb1 = np.var((img[:, :, 1] + img[:, :, 2]) * (img[:, :, 0] + img[:, :, 2])) / 4
    #

    rg_rg = np.mean((img[:, :, 0] + img[:, :, 1]) @ (img[:, :, 0] + img[:, :, 1])) / 4
    rb_rb = np.mean((img[:, :, 0] + img[:, :, 2]) @ (img[:, :, 0] + img[:, :, 2])) / 4
    gb_gb = np.mean((img[:, :, 1] + img[:, :, 2]) @ (img[:, :, 1] + img[:, :, 2])) / 4
    vv_rg_rg = np.var((img[:, :, 0] + img[:, :, 1]) @ (img[:, :, 0] + img[:, :, 1])) / 4
    vv_rb_rb = np.var((img[:, :, 0] + img[:, :, 2]) @ (img[:, :, 0] + img[:, :, 2])) / 4
    vv_gb_gb = np.var((img[:, :, 1] + img[:, :, 2]) @ (img[:, :, 1] + img[:, :, 2])) / 4

    #
    rg_rg1 = np.mean((img[:, :, 0] + img[:, :, 1]) * (img[:, :, 0] + img[:, :, 1])) / 4
    rb_rb1 = np.mean((img[:, :, 0] + img[:, :, 2]) * (img[:, :, 0] + img[:, :, 2])) / 4
    gb_gb1 = np.mean((img[:, :, 1] + img[:, :, 2]) * (img[:, :, 1] + img[:, :, 2])) / 4
    vv_rg_rg1 = np.var((img[:, :, 0] + img[:, :, 1]) * (img[:, :, 0] + img[:, :, 1])) / 4
    vv_rb_rb1 = np.var((img[:, :, 0] + img[:, :, 2]) * (img[:, :, 0] + img[:, :, 2])) / 4
    vv_gb_gb1 = np.var((img[:, :, 1] + img[:, :, 2]) * (img[:, :, 1] + img[:, :, 2])) / 4

    #
    return det_g_g_g_g, det_r_r, det_g_g_g, det_b_b, det_r_r_r, det_g_g, det_b_b_b, detrgb2, detrgb3, detrg, detrb, detgb, detrgb, det_r, det_g, det_b, det_r_g, det_r_b, det_g_b, det_r_g_b, trrg, trrb, trgb, trrgb, tr_r, tr_g, tr_b, tr_r_g, tr_r_b, tr_g_b, tr_r_g_b, r, g, b, r_g, g_r, r_b, b_r, g_b, b_g, r_g_b, r_b_g, g_r_b, g_b_r, b_r_g, b_g_r, vv_r_g, vv_g_r, vv_r_b, vv_b_r, vv_g_b, vv_b_g, vv_r_g_b, vv_r_b_g, vv_g_r_b, vv_g_b_r, vv_b_r_g, vv_b_g_r, rgb, var_r, var_g, var_b, var_rgb, rg, rb, gb, var_rg, var_rb, var_gb, r_r, g_g, b_b, var_r_r, var_g_g, var_b_b, rg_rb, rb_rg, rg_gb, gb_rg, rb_gb, gb_rb, vv_rg_rb, vv_rb_rg, vv_rg_gb, vv_gb_rg, vv_rb_gb, vv_gb_rb, rg_rg, rb_rb, gb_gb, vv_rg_rg, vv_rb_rb, vv_gb_gb, r_g1, g_r1, r_b1, b_r1, g_b1, b_g1, r_g_b1, r_b_g1, g_r_b1, g_b_r1, b_r_g1, b_g_r1, vv_r_g1, vv_g_r1, vv_r_b1, vv_b_r1, vv_g_b1, vv_b_g1, vv_r_g_b1, vv_r_b_g1, vv_g_r_b1, vv_g_b_r1, vv_b_r_g1, vv_b_g_r1, r_r1, g_g1, b_b1, var_r_r1, var_g_g1, var_b_b1, rg_rb1, rb_rg1, rg_gb1, gb_rg1, rb_gb1, gb_rb1, vv_rg_rb1, vv_rb_rg1, vv_rg_gb1, vv_gb_rg1, vv_rb_gb1, vv_gb_rb1, rg_rg1, rb_rb1, gb_gb1, vv_rg_rg1, vv_rb_rb1, vv_gb_gb1


# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([tons_of_features(img_patches[i]) for i in range(len(img_patches))])
    return X


# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            im[j:j + w, i:i + h] = labels[idx]
            idx = idx + 1
    return im

#making an overlay
def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * 255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img



# Compute features for each image patch
foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch

#identify a patch whther belongs to road or background by putting the treshhold
def value_to_class(v):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1
    else:
        return 0

#generating Matrix X and vector Y for train data
X = np.asarray([tons_of_features(img_patches_all[i]) for i in range(len(img_patches_all))])
Y = np.asarray([value_to_class(np.mean(gt_patches_all[i])) for i in range(len(gt_patches_all))])

#We import Sklearn
from sklearn import linear_model

# we create an instance of the classifier and fit the data
logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
#Approach for the crossvalidation we test different degree of polynomial to fit
#the best degree was one.
degree = 1
logreg.fit(X, Y)
#finding the weight vector
Z = logreg.predict(X)


w = 608
h = 608

#load test data images
test_dir = "Datasets/test_set_images/test_"
s=[]
for i in range(1,51):
    s.append(test_dir + np.str(i) + "/" + "test_" + np.str(i) + ".png")
imgs_test = [load_image(s[i]) for i in range(50)]

#this part is working on test data to make a prediction
XX = []
Zii = []
for j in range(50):
    Xi_test = extract_img_features(s[j])
    XX.append(np.vstack([Xi_test ** i for i in range(1, degree + 1)]))
    Zii.append(logreg.predict(XX[j]))
    print("number of test image:" + str(j))
print(np.shape(Zii))
W = np.hstack(Zii)

#for this part we made prediction, so the result is for putting it in csv file.
def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, 608, patch_size):
        for i in range(0, 608, patch_size):
            label = W[int(((j * 38 + i) / 16) + (38 * 38) * (img_number - 1))]
            yield ("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, "w") as f:
        f.write("id,prediction\n")
        for fn in image_filenames[0:]:
            f.writelines("{}\n".format(s) for s in mask_to_submission_strings(fn))


#write a file for submission on test data
submission_filename = "Submission_logistic.csv"
image_filenames = s
masks_to_submission(submission_filename, *image_filenames)

#this part is use for calcualting f1-score. We decided to do it more visually on notebook file which is added.
#Also we consider training images in notebook file. in this file the final result is for test images.
'''
from sklearn.metrics import f1_score
def compute_f1_score(img_prediction, y_true):
        y_true = 1 - y_true
        y_true = y_true.flatten()
        img_prediction = img_prediction.flatten()
        score = f1_score(y_true, img_prediction, average='macro')
        return score
'''