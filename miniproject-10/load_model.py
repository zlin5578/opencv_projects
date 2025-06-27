import numpy as np 
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from unet import load_and_prepare_data, bce_dice_loss, calculate_iou

# Paths to image and mask folders
image_path = "./dataset/images/*.png"
mask_path = "./dataset/masks/*.png"
# image_path = "./sample_dataset/images/*.png"
# mask_path = "./sample_dataset/masks/*.png"

# Load data
image_paths = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
mask_paths = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])
images, masks = load_and_prepare_data(image_paths, mask_paths)

# Function to display ALL original images, masks, and predicted masks with IoU scores
def display_predictions(model, images, masks, samples_per_batch=3):
    num_samples = len(images)
    num_batches = (num_samples + samples_per_batch - 1) // samples_per_batch

    for batch_idx in range(num_batches):
        start = batch_idx * samples_per_batch
        end = min(start + samples_per_batch, num_samples)

        fig, axes = plt.subplots(end - start, 3, figsize=(12, 4 * (end - start)))
        fig.suptitle(f'Batch {batch_idx + 1} of {num_batches}', fontsize=16)

        if end - start == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(start, end):
            img = images[i]
            mask = masks[i].squeeze()
            pred_mask = model.predict(np.expand_dims(img, axis=0))[0].squeeze()
            iou_score = calculate_iou(mask > 0.5, pred_mask > 0.5)

            row = i - start
            axes[row, 0].imshow(img)
            axes[row, 0].set_title('Original Image')
            axes[row, 0].axis('off')

            axes[row, 1].imshow(mask)
            axes[row, 1].set_title('Original Mask')
            axes[row, 1].axis('off')

            axes[row, 2].imshow(pred_mask)
            axes[row, 2].set_title(f'Predicted Mask\nIoU: {iou_score:.2f}')
            axes[row, 2].axis('off')

        plt.tight_layout()
        plt.show()

        user_input = input("Press 'Enter' to continue or 'q' to quit: ").strip().lower()
        if user_input == 'q':
            break

# Display predictions
model = tf.keras.models.load_model("Best_Detailed_Model.keras", custom_objects={"bce_dice_loss": bce_dice_loss})
# model = tf.keras.models.load_model("Best_Simple_Model.keras", custom_objects={"bce_dice_loss": bce_dice_loss})
display_predictions(model, images, masks)
