import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths and parameters
test_data_dir = r"C:\Users\joysh\task4\archive (4)\test\test"
img_width, img_height = 128, 128
batch_size = 32

# Load the saved model
model = tf.keras.models.load_model('hand_gesture_m.h5')

# Data generator for testing (only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generator for test data
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Important: set shuffle=False for test evaluation
)

# Evaluate the model on the test data
evaluation = model.evaluate(test_generator)

print("Test Accuracy:", evaluation[1])
