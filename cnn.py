from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import os

folder_path = "data"  # Replace with the actual folder path

image_paths = []
labels = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    new_folder_path = os.path.join(folder_path, filename)
    for new_filename in os.listdir(new_folder_path):
        if new_filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(new_folder_path, new_filename)
            image_paths.append(image_path)

            # Extract the label from the filename
            label = filename.split("_")[0]
            labels.append(label)


# Split the data into training and remaining data (validation + testing)
train_image_paths, remaining_image_paths, train_labels, remaining_labels = \
    train_test_split(image_paths, labels, train_size=0.7)

# Split the remaining data into validation and testing
val_image_paths, test_image_paths, val_labels, test_labels = \
    train_test_split(remaining_image_paths, remaining_labels, test_size=0.5)

# Define the target size for resizing
target_size = (224, 224)  # Adjust as per your needs

# Resize the training images
train_images_resized = []
for image_path in train_image_paths:
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    train_images_resized.append(resized_image)

# Convert the resized training images to a NumPy array
train_images_resized = np.array(train_images_resized)

# Normalize pixel values to the range [0, 1]
train_images_resized = train_images_resized / 255.0


# Resize the validation images
val_images_resized = []
for image_path in val_image_paths:
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    val_images_resized.append(resized_image)

# Convert the resized validation images to a NumPy array
val_images_resized = np.array(val_images_resized)

# Normalize pixel values to the range [0, 1]
val_images_resized = val_images_resized / 255.0


# Resize the testing images
test_images_resized = []
for image_path in test_image_paths:
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, target_size)
    test_images_resized.append(resized_image)

# Convert the resized testing images to a NumPy array
test_images_resized = np.array(test_images_resized)

# Normalize pixel values to the range [0, 1]
test_images_resized = test_images_resized / 255.0

# Convert labels to numeric form
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Apply one-hot encoding
onehot_encoder = OneHotEncoder(sparse=False)
train_labels_onehot = onehot_encoder.fit_transform(train_labels_encoded.reshape(-1, 1))
val_labels_onehot = onehot_encoder.transform(val_labels_encoded.reshape(-1, 1))
test_labels_onehot = onehot_encoder.transform(test_labels_encoded.reshape(-1, 1))

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(9, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images_resized, train_labels_onehot, validation_data=(val_images_resized, val_labels_onehot), epochs=10, batch_size=32)

test_loss, test_accuracy = model.evaluate(test_images_resized, test_labels_onehot, verbose=2)
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)

predictions = model.predict(test_images_resized)
