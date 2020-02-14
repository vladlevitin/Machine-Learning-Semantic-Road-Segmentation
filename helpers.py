import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image
import helpers as hp
from sklearn.metrics import f1_score


#rotate image by a degree
def rotate_img(infilename, degree):
    img  = Image.open(infilename)
    rotated = img.rotate(degree)
    rotated.save(infilename[:-4]+str(degree)+".png")

#rotate the image by 45 degrees and crop the image
def rotate_crop_img(infilename, degree):
    if (degree==45):
      img  = Image.open(infilename)
      rotated = img.rotate(degree)
      cropped=rotated.crop((70,70,330,330)) #crop the image so the center
      cropped.save(infilename[:-4]+"+"+str(degree)+".png")    

        
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data

def img_float_to_uint8(img): #initial image is float values
    rimg = img - np.min(img) #min of each color
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8) # converts the pixel colors to scale 0 to 255
    return rimg

# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8) #initializes a new gt image, the last param is of length 3
        gt_img8 = img_float_to_uint8(gt_img)  #converts float values to rgb scale        
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h] #covert it into patches
            else:
                im_patch = im[j:j+w, i:i+h, :] #covert into patches and keep the extra data
            list_patches.append(im_patch)
    return list_patches

# Extract RGB features consisting of average of RGB colors as well as variance
def extract_features_rgb(img):
    feat_m = np.mean(img, axis=(0,1))
    feat_v = np.var(img, axis=(0,1))
    feat = np.append(feat_m, feat_v)
    return feat

# Extract 2-dimensional features consisting of average gray color as well as variance
def extract_features_2d(img):
    feat_m = np.mean(img)
    feat = np.append(feat_m, feat_v)
    return feat


#Extracts 22 features based on the RBG colors
def extract_features_22(img):
    r=np.mean(img[:,:,0])
    g=np.mean(img[:,:,1])
    b=np.mean(img[:,:,2])
    r_var=np.var(img[:,:,0])
    g_var=np.var(img[:,:,1])
    b_var=np.var(img[:,:,2])
    rg_s=np.mean(img[:,:,0]+img[:,:,1])/2
    rb_s=np.mean(img[:,:,0]+img[:,:,2])/2
    gb_s=np.mean(img[:,:,1]+img[:,:,2])/2
    rgb_s=np.mean((img[:,:,0]+img[:,:,1]+img[:,:,2]))/3
    rg_var_s = np.var(((img[:,:,0])+(img[:,:,1]))/2)
    rb_var_s = np.var(((img[:,:,0])+(img[:,:,2]))/2)
    gb_var_s = np.var(((img[:,:,1])+(img[:,:,2]))/2)
    rgb_var_s = np.var(img)
    rg=np.mean(img[:,:,0]*img[:,:,1])
    rb=np.mean(img[:,:,0]*img[:,:,2])
    gb=np.mean(img[:,:,1]*img[:,:,2])
    rgb=np.mean(img[:,:,0]*img[:,:,1]*img[:,:,2])
    rg_var=np.var(img[:,:,0]*img[:,:,1])
    rb_var=np.var(img[:,:,0]*img[:,:,2])
    gb_var=np.var(img[:,:,1]*img[:,:,2])
    rgb_var=np.var(img[:,:,0]*img[:,:,1]*img[:,:,2])
    return r,g,b,r_var,g_var,b_var,rg_s,rb_s,gb_s,rgb_s,rg_var_s,rb_var_s,gb_var_s,rgb_var_s,rg,rb,gb,rgb,rg_var,rb_var,gb_var,rgb_var

#Extract 124 features on the RBG colors
def extract_features_124(img):
    det_r = np.linalg.det(img[:,:,0])
    det_g = np.linalg.det(img[:,:,1])
    det_b = np.linalg.det(img[:,:,2])
    det_r_g = np.linalg.det(img[:,:,0]@img[:,:,1])
    det_r_b = np.linalg.det(img[:,:,0]@img[:,:,2])
    det_g_b = np.linalg.det(img[:,:,1]@img[:,:,2])
    det_r_g_b = np.linalg.det(img[:,:,0]@img[:,:,1]@img[:,:,2])
    tr_r = np.trace(img[:,:,0])
    tr_g = np.trace(img[:,:,1])
    tr_b = np.trace(img[:,:,2])
    tr_r_g = np.trace(img[:,:,0]@img[:,:,1])
    tr_r_b = np.trace(img[:,:,0]@img[:,:,2])
    tr_g_b = np.trace(img[:,:,1]@img[:,:,2])
    tr_r_g_b = np.trace(img[:,:,0]@img[:,:,1]@img[:,:,2])
    
    r = np.mean(img[:,:,0])
    g = np.mean(img[:,:,1])
    b = np.mean(img[:,:,2])
    var_r = np.var(img[:,:,0])
    var_g = np.var(img[:,:,1])
    var_b = np.var(img[:,:,2])
    rgb = np.mean(img)
    var_rgb = np.var(img)

    rg = (np.mean(img[:,:,0])+np.mean(img[:,:,1]))/2
    rb = (np.mean(img[:,:,0])+np.mean(img[:,:,2]))/2
    gb = (np.mean(img[:,:,1])+np.mean(img[:,:,2]))/2
    var_rg = np.var(((img[:,:,0])+(img[:,:,1]))/2)
    var_rb = np.var(((img[:,:,0])+(img[:,:,2]))/2)
    var_gb = np.var(((img[:,:,1])+(img[:,:,2]))/2)
    
    r_g = np.mean(img[:,:,0]@img[:,:,1])
    g_r = np.mean(img[:,:,1]@img[:,:,0])
    r_b = np.mean(img[:,:,0]@img[:,:,2])
    b_r = np.mean(img[:,:,2]@img[:,:,0])
    g_b = np.mean(img[:,:,1]@img[:,:,2])
    b_g = np.mean(img[:,:,2]@img[:,:,1])
    vv_r_g = np.var(img[:,:,0]@img[:,:,1])
    vv_g_r = np.var(img[:,:,1]@img[:,:,0])
    vv_r_b = np.var(img[:,:,0]@img[:,:,2])
    vv_b_r = np.var(img[:,:,2]@img[:,:,0])
    vv_g_b = np.var(img[:,:,1]@img[:,:,2])
    vv_b_g = np.var(img[:,:,2]@img[:,:,1])
    r_r = np.mean(img[:,:,0]@img[:,:,0])
    g_g = np.mean(img[:,:,1]@img[:,:,1])
    b_b = np.mean(img[:,:,2]@img[:,:,2])
    var_r_r = np.var(img[:,:,0]@img[:,:,0])
    var_g_g = np.var(img[:,:,1]@img[:,:,1])
    var_b_b = np.var(img[:,:,2]@img[:,:,2])
    #
    r_g1 = np.mean(img[:,:,0]*img[:,:,1])
    g_r1 = np.mean(img[:,:,1]*img[:,:,0])
    r_b1 = np.mean(img[:,:,0]*img[:,:,2])
    b_r1 = np.mean(img[:,:,2]*img[:,:,0])
    g_b1 = np.mean(img[:,:,1]*img[:,:,2])
    b_g1 = np.mean(img[:,:,2]*img[:,:,1])
    vv_r_g1 = np.var(img[:,:,0]*img[:,:,1])
    vv_g_r1 = np.var(img[:,:,1]*img[:,:,0])
    vv_r_b1 = np.var(img[:,:,0]*img[:,:,2])
    vv_b_r1 = np.var(img[:,:,2]*img[:,:,0])
    vv_g_b1 = np.var(img[:,:,1]*img[:,:,2])
    vv_b_g1 = np.var(img[:,:,2]*img[:,:,1])
    r_r1 = np.mean(img[:,:,0]*img[:,:,0])
    g_g1 = np.mean(img[:,:,1]*img[:,:,1])
    b_b1 = np.mean(img[:,:,2]*img[:,:,2])
    var_r_r1 = np.var(img[:,:,0]*img[:,:,0])
    var_g_g1 = np.var(img[:,:,1]*img[:,:,1])
    var_b_b1 = np.var(img[:,:,2]*img[:,:,2])    
    #
    
    r_g_b = np.mean(img[:,:,0]@img[:,:,1]@img[:,:,2])
    r_b_g = np.mean(img[:,:,0]@img[:,:,2]@img[:,:,1])
    g_r_b = np.mean(img[:,:,1]@img[:,:,0]@img[:,:,2])
    g_b_r = np.mean(img[:,:,1]@img[:,:,2]@img[:,:,0])
    b_r_g = np.mean(img[:,:,2]@img[:,:,0]@img[:,:,1])
    b_g_r = np.mean(img[:,:,2]@img[:,:,1]@img[:,:,0])
    vv_r_g_b = np.var(img[:,:,0]@img[:,:,1]@img[:,:,2])
    vv_r_b_g = np.var(img[:,:,0]@img[:,:,2]@img[:,:,1])
    vv_g_r_b = np.var(img[:,:,1]@img[:,:,0]@img[:,:,2])
    vv_g_b_r = np.var(img[:,:,1]@img[:,:,2]@img[:,:,0])
    vv_b_r_g = np.var(img[:,:,2]@img[:,:,0]@img[:,:,1])
    vv_b_g_r = np.var(img[:,:,2]@img[:,:,1]@img[:,:,0])
    
    #
    r_g_b1 = np.mean(img[:,:,0]*img[:,:,1]*img[:,:,2])
    r_b_g1 = np.mean(img[:,:,0]*img[:,:,2]*img[:,:,1])
    g_r_b1 = np.mean(img[:,:,1]*img[:,:,0]*img[:,:,2])
    g_b_r1 = np.mean(img[:,:,1]*img[:,:,2]*img[:,:,0])
    b_r_g1 = np.mean(img[:,:,2]*img[:,:,0]*img[:,:,1])
    b_g_r1 = np.mean(img[:,:,2]*img[:,:,1]*img[:,:,0])
    vv_r_g_b1 = np.var(img[:,:,0]*img[:,:,1]*img[:,:,2])
    vv_r_b_g1 = np.var(img[:,:,0]*img[:,:,2]*img[:,:,1])
    vv_g_r_b1 = np.var(img[:,:,1]*img[:,:,0]*img[:,:,2])
    vv_g_b_r1 = np.var(img[:,:,1]*img[:,:,2]*img[:,:,0])
    vv_b_r_g1 = np.var(img[:,:,2]*img[:,:,0]*img[:,:,1])
    vv_b_g_r1 = np.var(img[:,:,2]*img[:,:,1]*img[:,:,0])    
    #
   
    
    rg_rb = np.mean((img[:,:,0]+img[:,:,1])@(img[:,:,0]+img[:,:,2]))/4
    rb_rg = np.mean((img[:,:,0]+img[:,:,2])@(img[:,:,0]+img[:,:,1]))/4
    rg_gb = np.mean((img[:,:,0]+img[:,:,1])@(img[:,:,1]+img[:,:,2]))/4
    gb_rg = np.mean((img[:,:,1]+img[:,:,2])@(img[:,:,0]+img[:,:,1]))/4
    rb_gb = np.mean((img[:,:,0]+img[:,:,2])@(img[:,:,1]+img[:,:,2]))/4
    gb_rb = np.mean((img[:,:,1]+img[:,:,2])@(img[:,:,0]+img[:,:,2]))/4
    vv_rg_rb = np.var((img[:,:,0]+img[:,:,1])@(img[:,:,0]+img[:,:,2]))/4
    vv_rb_rg = np.var((img[:,:,0]+img[:,:,2])@(img[:,:,0]+img[:,:,1]))/4
    vv_rg_gb = np.var((img[:,:,0]+img[:,:,1])@(img[:,:,1]+img[:,:,2]))/4
    vv_gb_rg = np.var((img[:,:,1]+img[:,:,2])@(img[:,:,0]+img[:,:,1]))/4
    vv_rb_gb = np.var((img[:,:,0]+img[:,:,2])@(img[:,:,1]+img[:,:,2]))/4
    vv_gb_rb = np.var((img[:,:,1]+img[:,:,2])@(img[:,:,0]+img[:,:,2]))/4

#
    rg_rb1 = np.mean((img[:,:,0]+img[:,:,1])*(img[:,:,0]+img[:,:,2]))/4
    rb_rg1 = np.mean((img[:,:,0]+img[:,:,2])*(img[:,:,0]+img[:,:,1]))/4
    rg_gb1 = np.mean((img[:,:,0]+img[:,:,1])*(img[:,:,1]+img[:,:,2]))/4
    gb_rg1 = np.mean((img[:,:,1]+img[:,:,2])*(img[:,:,0]+img[:,:,1]))/4
    rb_gb1 = np.mean((img[:,:,0]+img[:,:,2])*(img[:,:,1]+img[:,:,2]))/4
    gb_rb1 = np.mean((img[:,:,1]+img[:,:,2])*(img[:,:,0]+img[:,:,2]))/4
    vv_rg_rb1 = np.var((img[:,:,0]+img[:,:,1])*(img[:,:,0]+img[:,:,2]))/4
    vv_rb_rg1 = np.var((img[:,:,0]+img[:,:,2])*(img[:,:,0]+img[:,:,1]))/4
    vv_rg_gb1 = np.var((img[:,:,0]+img[:,:,1])*(img[:,:,1]+img[:,:,2]))/4
    vv_gb_rg1 = np.var((img[:,:,1]+img[:,:,2])*(img[:,:,0]+img[:,:,1]))/4
    vv_rb_gb1 = np.var((img[:,:,0]+img[:,:,2])*(img[:,:,1]+img[:,:,2]))/4
    vv_gb_rb1 = np.var((img[:,:,1]+img[:,:,2])*(img[:,:,0]+img[:,:,2]))/4
#
    
    rg_rg = np.mean((img[:,:,0]+img[:,:,1])@(img[:,:,0]+img[:,:,1]))/4
    rb_rb = np.mean((img[:,:,0]+img[:,:,2])@(img[:,:,0]+img[:,:,2]))/4
    gb_gb = np.mean((img[:,:,1]+img[:,:,2])@(img[:,:,1]+img[:,:,2]))/4
    vv_rg_rg = np.var((img[:,:,0]+img[:,:,1])@(img[:,:,0]+img[:,:,1]))/4
    vv_rb_rb = np.var((img[:,:,0]+img[:,:,2])@(img[:,:,0]+img[:,:,2]))/4
    vv_gb_gb = np.var((img[:,:,1]+img[:,:,2])@(img[:,:,1]+img[:,:,2]))/4

#
    rg_rg1 = np.mean((img[:,:,0]+img[:,:,1])*(img[:,:,0]+img[:,:,1]))/4
    rb_rb1 = np.mean((img[:,:,0]+img[:,:,2])*(img[:,:,0]+img[:,:,2]))/4
    gb_gb1 = np.mean((img[:,:,1]+img[:,:,2])*(img[:,:,1]+img[:,:,2]))/4
    vv_rg_rg1 = np.var((img[:,:,0]+img[:,:,1])*(img[:,:,0]+img[:,:,1]))/4
    vv_rb_rb1 = np.var((img[:,:,0]+img[:,:,2])*(img[:,:,0]+img[:,:,2]))/4
    vv_gb_gb1 = np.var((img[:,:,1]+img[:,:,2])*(img[:,:,1]+img[:,:,2]))/4    
    
#    

    return det_r,det_g,det_b,det_r_g,det_r_b,det_g_b,det_r_g_b, tr_r,tr_g,tr_b,tr_r_g,tr_r_b,tr_g_b,tr_r_g_b, r,g,b,r_g,g_r,r_b,b_r,g_b,b_g,r_g_b,r_b_g,g_r_b,g_b_r,b_r_g,b_g_r, vv_r_g,vv_g_r,vv_r_b,vv_b_r,vv_g_b,vv_b_g,vv_r_g_b,vv_r_b_g,vv_g_r_b,vv_g_b_r,vv_b_r_g,vv_b_g_r,  rgb , var_r,var_g,var_b,var_rgb,rg,rb,gb,var_rg,var_rb,var_gb,r_r,g_g,b_b,var_r_r,var_g_g,var_b_b,rg_rb,rb_rg,rg_gb,gb_rg,rb_gb,gb_rb,vv_rg_rb,vv_rb_rg,vv_rg_gb,vv_gb_rg,vv_rb_gb,vv_gb_rb,rg_rg,rb_rb,gb_gb,vv_rg_rg,vv_rb_rb,vv_gb_gb,    r_g1,g_r1,r_b1,b_r1,g_b1,b_g1,r_g_b1,r_b_g1,g_r_b1,g_b_r1,b_r_g1,b_g_r1, vv_r_g1,vv_g_r1,vv_r_b1,vv_b_r1,vv_g_b1,vv_b_g1,vv_r_g_b1,vv_r_b_g1,vv_g_r_b1,vv_g_b_r1,vv_b_r_g1,vv_b_g_r1,r_r1,g_g1,b_b1,var_r_r1,var_g_g1,var_b_b1,rg_rb1,rb_rg1,rg_gb1,gb_rg1,rb_gb1,gb_rb1,vv_rg_rb1,vv_rb_rg1,vv_rg_gb1,vv_gb_rg1,vv_rb_gb1,vv_gb_rb1,rg_rg1,rb_rb1,gb_gb1,vv_rg_rg1,vv_rb_rb1,vv_gb_gb1


# Extract features for a given image
def extract_img_features(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([extract_features_2d(img_patches[i]) for i in range(len(img_patches))])
    return X

def extract_img_features_rgb(filename):
    img = load_image(filename)
    img_patches = img_crop(img, patch_size, patch_size)
    X = np.asarray([ extract_features_rgb(img_patches[i]) for i in range(len(img_patches))])
    return X

def extract_img_gt(filename,patch_size):
    gt=load_image(filename)
    gt_patches=img_crop(gt, patch_size, patch_size)
    Y= np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    

#covert the value other to 0 or 1 when comparing it to the threshold
def value_to_class(v,foreground_threshold):
    df = np.sum(v)
    if df > foreground_threshold:
        return 1 #road
    else:
        return 0 #background
    

#Assign labels to patches to create a prediction image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im

#Overlay the prediction on the image
def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


# compute the f1 score a set of predictions
def compute_f1_score(img_prediction, y_true):
        img_prediction = img_prediction.flatten()
        score = f1_score(y_true, img_prediction, average='binary')
        return score
