import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk

# Define dataset paths
TRAIN_IMAGE_DIR = "dataset/images/train/"
VAL_IMAGE_DIR = "dataset/images/val/"
TRAIN_LABEL_DIR = "dataset/labels/train/"
VAL_LABEL_DIR = "dataset/labels/val/"

# Read class labels from classes.txt
with open("classes.txt", "r") as f:
    class_labels = [line.strip() for line in f.readlines()]
num_classes = len(class_labels)

# Preprocessing function for images
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    return img_array

# Function to load dataset
def load_dataset(image_dir, label_dir):
    images = []
    labels = []

    for image_file in os.listdir(image_dir):
        if image_file.endswith(".jpeg") or image_file.endswith(".jpg") or image_file.endswith(".png"):
            # Load image
            image_path = os.path.join(image_dir, image_file)
            image_array = preprocess_image(image_path)

            # Load label
            label_file = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    label_data = f.read().strip()
                    # Assuming label format: `class_index x_min y_min x_max y_max`
                    class_index = int(label_data.split()[0])  # Extract class index
                    labels.append(class_index)
                    images.append(image_array)

    return np.array(images), np.array(labels)

# Load train and validation datasets
print("Loading training dataset...")
X_train, y_train = load_dataset(TRAIN_IMAGE_DIR, TRAIN_LABEL_DIR)
print(f"Loaded {len(X_train)} training images and labels.")

print("Loading validation dataset...")
X_val, y_val = load_dataset(VAL_IMAGE_DIR, VAL_LABEL_DIR)
print(f"Loaded {len(X_val)} validation images and labels.")

# Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Train the model
model = build_model()
print("Training model...")
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)
model.save("crop_weed_classifier.h5")
print("Model trained and saved.")

# Tkinter GUI
class WeedCropClassifierApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model

        self.root.title("Crop vs Weed Classifier")
        self.root.geometry("600x400")

        # Widgets
        self.label = tk.Label(self.root, text="Upload an image to classify (crop/weed):")
        self.label.pack(pady=20)

        self.upload_button = tk.Button(self.root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.canvas = tk.Canvas(self.root, width=300, height=300)
        self.canvas.pack()

        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            # Display image
            img = Image.open(file_path)
            img = img.resize((300, 300))
            self.image = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)

            # Classify the image
            result = self.classify_image(file_path)
            self.result_label.config(text=f"Prediction: {result}")

    def classify_image(self, file_path):
        img_array = preprocess_image(file_path)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        predictions = self.model.predict(img_array)
        predicted_class = np.argmax(predictions)
        return class_labels[predicted_class]

# Main script
if __name__ == "__main__":
    # Load or train the model
    model_path = "crop_weed_classifier.h5"
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("Training a new model...")
        model = build_model()
        model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val, y_val)
        )
        model.save(model_path)

    # Start the GUI
    root = tk.Tk()
    app = WeedCropClassifierApp(root, model)
    root.mainloop()
