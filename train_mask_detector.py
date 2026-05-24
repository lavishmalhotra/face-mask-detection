import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Dataset path
dataset_path = "dataset"

data = []
labels = []

print("[INFO] Loading dataset...")

# Load images
for category in ["with_mask", "without_mask"]:
    path = os.path.join(dataset_path, category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)

        if image is None:
            continue

        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)

        data.append(image)
        labels.append(category)

# Convert to numpy
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Split data
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, random_state=42)

# Load MobileNetV2
baseModel = MobileNetV2(weights="imagenet",
                        include_top=False,
                        input_shape=(224, 224, 3))

# Build head model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile
model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=1e-4),
              metrics=["accuracy"])

print("[INFO] Training model...")

# Train
model.fit(
    trainX, trainY,
    batch_size=32,
    epochs=1,  
    validation_data=(testX, testY)
)

# Save model
model.save("mask_detector.keras") 
print("[INFO] Model saved!") 