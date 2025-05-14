import cv2
from skimage import io, filters
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
import os
from fer import FER
import shutil











image_processor = VGG16(weights="imagenet")
model = Model(inputs=image_processor.input, outputs=image_processor.get_layer('block5_pool').output)

# Define a function to extract features from an image
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

# Define a function to detect if an image is black and white
def is_black_and_white(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False, 0
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(image)
        
        # Check if the color channels are similar
        color_deviation = np.mean(np.abs(r - g)) + np.mean(np.abs(g - b))
        if color_deviation < 5:  # threshold
            brightness_contrast = gray_image.std()
            return True, brightness_contrast
        return False, 0
    except Exception as e:
        print(f"Error detecting B&W for {image_path}: {e}")
        return False, 0

# Define functions for sharpness, contrast, colorfulness, resolution, expression detection, brightness, and noise
def calculate_sharpness(image_path):
    try:
        image = io.imread(image_path, as_gray=True)
        edges = filters.sobel(image)
        return edges.var()
    except Exception as e:
        print(f"Error calculating sharpness for {image_path}: {e}")
        return 0

def calculate_contrast(image_path):
    try:
        image = io.imread(image_path, as_gray=True)
        return image.std()
    except Exception as e:
        print(f"Error calculating contrast for {image_path}: {e}")
        return 0

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

def calculate_resolution(image_path):
    try:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        return height * width
    except Exception as e:
        print(f"Error calculating resolution for {image_path}: {e}")
        return 0

def detect_facial_expressions(image_path):
    try:
        image = cv2.imread(image_path)
        detector = FER(mtcnn=True)
        result = detector.detect_emotions(image)
        if result:
            emotions = result[0]['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            return emotions[dominant_emotion]
        return 0
    except Exception as e:
        print(f"Error detecting facial expressions for {image_path}: {e}")
        return 0

def calculate_brightness(image_path):
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return np.mean(image)
    except Exception as e:
        print(f"Error calculating brightness for {image_path}: {e}")
        return 0

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
def calculate_image_quality(features, sharpness, contrast, colorfulness, resolution, expression_score, brightness, noise, is_bw):
    sharpness_norm = (sharpness - np.min(sharpness_scores)) / (np.max(sharpness_scores) - np.min(sharpness_scores) + 1e-7)
    contrast_norm = (contrast - np.min(contrast_scores)) / (np.max(contrast_scores) - np.min(contrast_scores) + 1e-7)
    resolution_norm = (resolution - np.min(resolution_scores)) / (np.max(resolution_scores) - np.min(resolution_scores) + 1e-7)
    brightness_norm = (brightness - np.min(brightness_scores)) / (np.max(brightness_scores) - np.min(brightness_scores) + 1e-7)
    noise_norm = (noise - np.min(noise_scores)) / (np.max(noise_scores) - np.min(noise_scores) + 1e-7)

    colorfulness_norm = 0 if is_bw else (colorfulness - np.min(colorfulness_scores)) / (np.max(colorfulness_scores) - np.min(colorfulness_scores) + 1e-7)
    sharpness_penalty = 0.1 * sharpness_norm if sharpness < 0.08 else sharpness_norm

    image_quality_score = (
        0.1 * np.mean(features) +
        0.2 * np.std(features) +
        0.6 * sharpness_penalty +
        0.2 * contrast_norm +
        0.05 * colorfulness_norm +
        0.05 * resolution_norm +
        0.1 * brightness_norm +
        -0.1 * noise_norm +
        0.15 * expression_score +
        (0.1 if is_bw else 0) 
    )
    
    return image_quality_score

# Define folder paths
folder_path = 'N'
discard_folder_path = os.path.join(os.path.dirname(folder_path), 'discardss')

if not os.path.exists(discard_folder_path):
    os.makedirs(discard_folder_path)

# Initialize lists to store scores
features_list, sharpness_scores, contrast_scores, colorfulness_scores = [], [], [], []
resolution_scores, expression_scores, brightness_scores, noise_scores, filenames = [], [], [], [], []

# Process each image
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder_path, filename)
        
        # Detect if the image is black-and-white
        is_bw, brightness_contrast = is_black_and_white(image_path)
        
        # Extract features and scores
        features = extract_features(image_path)
        sharpness = calculate_sharpness(image_path)
        contrast = brightness_contrast if is_bw else calculate_contrast(image_path)
        colorfulness = 0 if is_bw else calculate_colorfulness(image_path)
        resolution = calculate_resolution(image_path)
        expression_score = detect_facial_expressions(image_path)
        brightness = calculate_brightness(image_path)
        noise = estimate_noise(image_path)
        
        # Store scores
        features_list.append(features)
        sharpness_scores.append(sharpness)
        contrast_scores.append(contrast)
        colorfulness_scores.append(colorfulness)
        resolution_scores.append(resolution)
        expression_scores.append(expression_score)
        brightness_scores.append(brightness)
        noise_scores.append(noise)
        filenames.append(filename)

# Normalize and scale features
features_array = np.array(features_list)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_array)

# Calculate image quality scores
scores = []
for features, sharpness, contrast, colorfulness, resolution, expression_score, brightness, noise, filename in zip(
        scaled_features, sharpness_scores, contrast_scores, colorfulness_scores, resolution_scores, expression_scores, brightness_scores, noise_scores, filenames):
    
    # Calculate quality score
    is_bw, _ = is_black_and_white(os.path.join(folder_path, filename))
    image_quality_score = calculate_image_quality(features, sharpness, contrast, colorfulness, resolution, expression_score, brightness, noise, is_bw)
    scores.append(image_quality_score)

# Sort and discard images with low quality scores
threshold = 0.278
sorted_images = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)
for filename, score in sorted_images:
    if score < threshold:
        src_path = os.path.join(folder_path, filename)
        dest_path = os.path.join(discard_folder_path, filename)
        shutil.move(src_path, dest_path)


for filename, score in sorted_images:
    print(f"{filename}: {score:.4f}")


#end of the code
