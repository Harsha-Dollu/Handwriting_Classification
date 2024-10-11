import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def preprocess_images(images, size=(128, 128)):
    preprocessed_images = []
    for img in images:
        img = cv2.resize(img, size)
        img = img / 255.0  # Normalize pixel values to [0, 1]
        preprocessed_images.append(img)
    return np.array(preprocessed_images)

augmented_good_folder = 'C:\\Users\\harsh\\OneDrive\\Desktop\\python\\python_project1\\augmentedGoodHandwriting'
augmented_bad_folder = 'C:\\Users\\harsh\\OneDrive\\Desktop\\python\\python_project1\\augmentedBadHandwriting'

good_images = load_images_from_folder(augmented_good_folder)
bad_images = load_images_from_folder(augmented_bad_folder)

good_images = preprocess_images(good_images)
bad_images = preprocess_images(bad_images)

X = np.concatenate((good_images, bad_images), axis=0)
y = np.array([1] * len(good_images) + [0] * len(bad_images))

X = X[..., np.newaxis]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

test_good_folder = 'C:\\Users\\harsh\\OneDrive\\Desktop\\python\\python_project1\\Handwritings(test)\\Good'
test_bad_folder = 'C:\\Users\\harsh\\OneDrive\\Desktop\\python\\python_project1\\Handwritings(test)\\Bad'

good_test_images = preprocess_images(load_images_from_folder(test_good_folder))
bad_test_images = preprocess_images(load_images_from_folder(test_bad_folder))

X_test = np.concatenate((good_test_images, bad_test_images), axis=0)
y_test = np.array([1] * len(good_test_images) + [0] * len(bad_test_images))

X_test = X_test[..., np.newaxis]

def create_improved_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Dropout to prevent overfitting
        
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

input_shape = (128, 128, 1)
model = create_improved_model(input_shape)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('improved_handwriting_model.keras', save_best_only=True)

history = model.fit(X_train, y_train, 
                    epochs=20, 
                    validation_data=(X_val, y_val),  # Use validation data here
                    callbacks=[early_stopping, model_checkpoint], 
                    verbose=1)

training_accuracy = history.history['accuracy'][-1]
validation_accuracy = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {training_accuracy:.4f}")
print(f"Final Validation Accuracy: {validation_accuracy:.4f}")

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
