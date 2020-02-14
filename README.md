## Machine Learning Project
Road Segmentation on satellite image

Team: Vahid Shahverdi , Vladislav Levitin , Seyedehfatemeh Arfaeezarandi

CrowdAI team name: dsakjvdssavd 

### Python Libraries
-Matplotlib
-Pillow
-Sci-kit Learn
-Tensorflow
-Keras


### Structure of Code
The code is structed in 7 different components.

##### "basic_convolutional_model_baseline.py"
This model was given to us as a baseline and it use two-layer convolutional neural network in order to make the predictions.

##### "Special_Logistic_regression.py"
our first try is based on this model, we generate more than 140 features and we used logistic regression to creat submission.

##### "combine.py"
We combine two previous method to build a stronger result.
##### "ML_Notebook.ipynb"
Althouhg this file is a little bit messy, but we upload it since some of our cross validation are here
and we focus on training image by visualizing the result. To check how these things work.

##### "unet.py"

Our unet model crashed many times, so by the help of last year group, their names are "Maxime SchoemansPedro de Tavora SantosYannick Paul Klose"
we managed to run this model.(We cite them on the report a scientific article)
This model improves a little bit but most of the credit went back to them.

##### "run.py"

This is the file that we use to make the final submission file based on the u-net model.

##### "helpers.py"
We build this file for ML_Notebook.ipynb file.

##### "mask_to_submission"
Includes functions that are used to convert the predictions from the logistic regression into submission fil.
#### "unet_mask_to_submission"
Includes functions that are used to convert the predictions from the unet into submission file for the unet.

##########################################
#How to run the code
unet model have lots of parameters, so training on a PC is not possible. We used Kaggel deep learning platform to train them.
Download the model from  https://drive.google.com/drive/folders/1zrJjwZiFmMRkUpFuIiuTwk1vKtxaRo3z?usp=sharing . After that you can run the run.py file.

