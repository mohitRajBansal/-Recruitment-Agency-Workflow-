from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import cv2
from skimage import io, filters
import numpy as np
import os
import shutil

# Load VGG16 model
image_processor = VGG16(weights="imagenet")
model = Model(inputs=image_processor.input, outputs=image_processor.get_layer('block5_pool').output)

# Function to extract features from an image
def extract_features(image_path):
    try:
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        features = model.predict(image)
        return features.flatten()
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return np.zeros((512,))

# Function to calculate sharpness
def calculate_sharpness(image_path):
    try:
        image = io.imread(image_path, as_gray=True)
        edges = filters.sobel(image)
        return edges.var()
    except Exception as e:
        print(f"Error calculating sharpness for {image_path}: {e}")
        return 0

# Function to calculate colorfulness
def calculate_colorfulness(image_path):
    try:
        image = cv2.imread(image_path)
        (B, G, R) = cv2.split(image.astype("float"))
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        std_root = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
        mean_root = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
        return std_root + (0.3 * mean_root)
    except Exception as e:
        print(f"Error calculating colorfulness for {image_path}: {e}")
        return 0

# Function to calculate brightness
def calculate_brightness(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return np.mean(image)
    except Exception as e:
        print(f"Error calculating brightness for {image_path}: {e}")
        return 0


# Function to calculate the resolution of an image
def calculate_resolution(image_path):
    try:
        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            return height * width
        else:
            print(f"Error: Unable to load image at {image_path}")
            return 0
    except Exception as e:
        print(f"Error calculating resolution for {image_path}: {e}")
        return 0



# Function to estimate noise
def estimate_noise(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        high_pass_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered_image = cv2.filter2D(image, -1, high_pass_filter)
        return np.std(filtered_image)
    except Exception as e:
        print(f"Error estimating noise for {image_path}: {e}")
        return 0

# Define the function to calculate the quality score of an image
def calculate_image_quality(features, sharpness, colorfulness, brightness, noise, resolution):
    sharpness_norm = sharpness / (np.max(sharpness_scores) + 1e-7)
    colorfulness_norm = colorfulness / (np.max(colorfulness_scores) + 1e-7)
    brightness_norm = brightness / (np.max(brightness_scores) + 1e-7)
    noise_norm = noise / (np.max(noise_scores) + 1e-7)
    resolution_norm = resolution / (np.max(resolution_scores) + 1e-7)

    image_quality_score = (
        0.4 * sharpness_norm +
        0.2 * colorfulness_norm +
        0.15 * resolution_norm +
        0.15 * brightness_norm -
        0.1 * noise_norm
    )
    return image_quality_score

# Process images in folder
folder_path = 'N'
discard_folder_path = os.path.join(os.path.dirname(folder_path), 'discarded_images')

if not os.path.exists(discard_folder_path):
    os.makedirs(discard_folder_path)

sharpness_scores, colorfulness_scores, brightness_scores, noise_scores, resolution_scores = [], [], [], [], []
features_list, filenames = [], []

# Process each image
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder_path, filename)

        # Extract features and calculate metrics
        features = extract_features(image_path)
        sharpness = calculate_sharpness(image_path)
        colorfulness = calculate_colorfulness(image_path)
        brightness = calculate_brightness(image_path)
        noise = estimate_noise(image_path)
        resolution = calculate_resolution(image_path)

        # Store scores
        features_list.append(features)
        sharpness_scores.append(sharpness)
        colorfulness_scores.append(colorfulness)
        brightness_scores.append(brightness)
        noise_scores.append(noise)
        resolution_scores.append(resolution)
        filenames.append(filename)

# Normalize features
scaler = StandardScaler()
features_array = np.array(features_list)
scaled_features = scaler.fit_transform(features_array)

# Calculate quality scores
scores = []
for features, sharpness, colorfulness, brightness, noise, resolution, filename in zip(
        scaled_features, sharpness_scores, colorfulness_scores, brightness_scores, noise_scores, resolution_scores, filenames):
    
    quality_score = calculate_image_quality(features, sharpness, colorfulness, brightness, noise, resolution)
    scores.append(quality_score)

# Sort and discard low-quality images
threshold = 0.5
sorted_images = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)
for filename, score in sorted_images:
    if score < threshold:
        src_path = os.path.join(folder_path, filename)
        dest_path = os.path.join(discard_folder_path, filename)
        shutil.move(src_path, dest_path)

for filename, score in sorted_images:
    print(f"{filename}: {score:.4f}")
