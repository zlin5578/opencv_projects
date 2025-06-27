## Model Implementation
This modified U-Net model is intended to perform road segmentation. 
The model architecture consists of an encoder-decoder structure with skip connections to retain spatial information during upsampling. 
The model is trained to optimize a binary cross-entropy/DICE hybrid loss and accuracy metrics, with additional evaluation using the Intersection over Union (IoU) metric.
Two models/image sets are used. The 'detailed' model is trained on the custom dataset, which is resized to 512x512 and normalized. The 'simple' model is trained on the provided dataset, which is resized to 512x512 and normalized.
The custom dataset is far more detailed while the provided dataset is more homogeneous, performing relatively well at a smaller size (256x256) while training & running inference at a fraction of the time.
However, a consistent training size is necessary so 512x512 was used to accommodate the more detailed model, despite its far longer computation time.

## Usage / Loading a Model (load_model.py)
To load one of the models, enter the name of the model under the following line in the load_model.py file:

model = tf.keras.models.load_model("Best_Detailed_Model.keras", custom_objects={"bce_dice_loss": bce_dice_loss})

Then run the file.
It requires the presence of the unet.py file to run.
By default, the name of the 'detailed' model is entered.
The model will run inference on ALL files in the target dataset.
Once a batch is complete, press 'Enter' in the terminal to generate the next batch or 'q' to terminate the process prematurely.

## Training (unet.py)
The 'detailed' model is trained on a customized road image set for 10 epochs with a batch size of 16.
The 'simple' model is trained on the provided dataset for 5 epochs with a batch size of 16.
The resolution, number of epochs, batch size, and learning rate are all easily modifiable at the start of the unet.py file.
To train the model on a dataset, place your images in /dataset/images and your masks in /dataset/masks.
Alternatively, you can modify image_path & mask_path in unet.py.
Only use .png files for your images and masks.
Once the data is in place, run unet.py and wait for the model training to complete.
After training, the best model will save and the training loss/accuracy graphs will display.
Once the graphs are closed, inference will run on the testing portion of the dataset, using the best model.
Inference will work in batches, displaying a set of test images with their accompanying masks until all of the test images are cycled through.


## Dependencies
* Scikit-learn
* Tensorflow
* OpenCV
* Numpy
* MatPlotLib

## Testing
Aside from clear differences in the natures of the datasets, there also differences in labeling that contributed to varying performances when models were ran on the alternate dataset.

