import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import random 
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split

RESOLUTION = 512
NUM_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 1e-3

# Paths to image and mask folders
image_path = "./dataset/images/*.png"
mask_path = "./dataset/masks/*.png"
# image_path = "./sample_dataset/images/*.png"
# mask_path = "./sample_dataset/masks/*.png"

def display_images(image_folder, num_images=9):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('jpg', 'png', 'jpeg'))]
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for i, file_name in enumerate(image_files[:num_images]):
        image = cv2.imread(os.path.join(image_folder, file_name))
        axes[i // 3, i % 3].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i // 3, i % 3].set_title(file_name)
        axes[i // 3, i % 3].axis('off')
    plt.tight_layout()
    plt.show()

def resize_image(image, size=(RESOLUTION, RESOLUTION)):
    return cv2.resize(image, size)

def resize_mask(mask, size=(RESOLUTION, RESOLUTION)):
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return np.expand_dims(cv2.resize(mask_gray, size, interpolation=cv2.INTER_NEAREST), axis=2)

def load_and_prepare_data(image_paths, mask_paths, target_size=(RESOLUTION, RESOLUTION)):
    image_list, mask_list = [], []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image = plt.imread(image_path).astype(np.float32)
        mask = plt.imread(mask_path).astype(np.float32)
        image_list.append(resize_image(image, target_size))
        mask_list.append(resize_mask(mask, target_size))
    return np.array(image_list), np.array(mask_list)

def create_unet_model(input_shape=(RESOLUTION, RESOLUTION, 3)):
    inputs = tf.keras.layers.Input(input_shape)

    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        return tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)

    enc1 = conv_block(inputs, 64)
    enc2 = tf.keras.layers.MaxPooling2D((2, 2))(enc1)
    enc3 = conv_block(enc2, 128)
    enc4 = tf.keras.layers.MaxPooling2D((2, 2))(enc3)
    enc5 = conv_block(enc4, 256)

    def upconv_block(x, skip, filters):
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Concatenate()([x, skip])
        return tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)

    dec1 = upconv_block(enc5, enc3, 128)
    dec2 = upconv_block(dec1, enc1, 64)
    outputs = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation='sigmoid')(dec2)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(y_true, y_pred, smooth=1e-6):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (denominator + smooth)
    dice_loss = 1.0 - dice_coeff
    return bce + dice_loss

def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

def display_predictions(model, X_test, y_test, batch_size=4):
    num_samples = len(X_test)
    total_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_samples)
        current_batch_size = end - start

        fig, axes = plt.subplots(current_batch_size, 3, figsize=(12, 4 * current_batch_size))

        if current_batch_size == 1:
            axes = np.expand_dims(axes, axis=0)

        for i in range(current_batch_size):
            idx = start + i
            original_img = X_test[idx]
            original_mask = y_test[idx]

            predicted_mask = model.predict(np.expand_dims(original_img, axis=0)).reshape(RESOLUTION, RESOLUTION)
            iou_score = calculate_iou(original_mask.squeeze(), predicted_mask > 0.5)

            axes[i, 0].imshow(original_img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(original_mask.squeeze())
            axes[i, 1].set_title('Original Mask')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(predicted_mask)
            axes[i, 2].set_title(f'Predicted Mask\nIoU: {iou_score:.2f}')
            axes[i, 2].axis('off')

        plt.tight_layout()
        print(f"Showing batch {batch_idx + 1}/{total_batches}. Close window to continue...")
        plt.show(block=True)

def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    
    # Load and prepare data
    image_paths = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
    mask_paths = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])
    images, masks = load_and_prepare_data(image_paths, mask_paths)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=23)

    # Create and compile model
    model = create_unet_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss=bce_dice_loss, metrics=['accuracy'])

    # Setup checkpoint callback
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_loss", mode="min", verbose=1)

    # Train the model
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_data=(X_test, y_test),
                        callbacks=[checkpoint])

    # Plot training history
    plot_training_history(history)

    # Load the best model saved and display predictions
    model = tf.keras.models.load_model("best_model.keras", custom_objects={"bce_dice_loss": bce_dice_loss})
    display_predictions(model, X_test, y_test)
