# Creates the prediction for a single image and outputs the strings that should go into the submission file
from tensorflow.compat.v1.keras.models import load_model
import numpy as np 


# percentage of pixels > 1 required to assign a foreground label to a patch
foreground_threshold = 0.25


def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0

def predict_img(img):
    img1 = img[:400,:400]
    img2 = img[:400,-400:]
    img3 = img[-400:,:400]
    img4 = img[-400:,-400:]
    imgs = np.array([img1,img2,img3,img4])
    labels = model.predict(imgs).round()
    img_label = np.empty((608,608,1))
    img_label[-400:,-400:] = labels[3]
    img_label[-400:,:400] = labels[2]
    img_label[:400,-400:] = labels[1]
    img_label[:400,:400] = labels[0]
    return img_label

def mask_to_submission_strings(image_filename, i):
    img_number = i+1
    print("Predicting image {}".format(image_filename))
    im = mpimg.imread(image_filename)
    im_label = predict_img(im)
  
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im_label[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))

# Converts images into a submission file
def masks_to_submission(submission_filename, *image_filenames):
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for i in range(len(image_filenames)):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image_filenames[i],i))
            
            