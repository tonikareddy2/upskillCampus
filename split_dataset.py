import os
import shutil
from sklearn.model_selection import train_test_split

# Define your paths
image_dir = 'dataset/images'
label_dir = 'dataset/labels'

# Ensure directories exist
assert os.path.exists(image_dir), f"Image directory '{image_dir}' does not exist!"
assert os.path.exists(label_dir), f"Label directory '{label_dir}' does not exist!"

# Get a list of all image filenames (with '.jpeg' extension for images)
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpeg')]
label_files = [f.replace('.jpeg', '.txt') for f in image_files]  # Match labels with images

# Check if all labels exist
missing_labels = [img for img, lbl in zip(image_files, label_files) if not os.path.exists(os.path.join(label_dir, lbl))]
if missing_labels:
    print("Warning: Some labels are missing for these images:", missing_labels)

# Split the dataset into training and validation sets (80% train, 20% val)
train_images, val_images = train_test_split(image_files, test_size=0.2, random_state=42)

# Create directories for train/val datasets
os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)
os.makedirs('dataset/labels/train', exist_ok=True)
os.makedirs('dataset/labels/val', exist_ok=True)

# Move files to the appropriate directories
for img in train_images:
    shutil.move(os.path.join(image_dir, img), os.path.join('dataset/images/train', img))
    label_file = img.replace('.jpeg', '.txt')
    shutil.move(os.path.join(label_dir, label_file), os.path.join('dataset/labels/train', label_file))

for img in val_images:
    shutil.move(os.path.join(image_dir, img), os.path.join('dataset/images/val', img))
    label_file = img.replace('.jpeg', '.txt')
    shutil.move(os.path.join(label_dir, label_file), os.path.join('dataset/labels/val', label_file))

# Create the dataset.yaml file
dataset_yaml_content = f"""
train: ./dataset/images/train
val: ./dataset/images/val

nc: 2  # Number of classes
names: ['crop', 'weed']  # Class names
"""

with open('dataset.yaml', 'w') as file:
    file.write(dataset_yaml_content)

print("Dataset has been split into train/val and dataset.yaml file has been created.")
