"""
Data preparation module.
Handles data loading, balancing, feature extraction, and splitting.
"""

import os
import glob
import random
import numpy as np
from PIL import Image
import cv2
import logging

logger = logging.getLogger(__name__)

def download_dataset():
    """Download the dataset using kagglehub."""
    try:
        import kagglehub
        path = kagglehub.dataset_download("alifatahi/multi-class-neurological-disorder-mcnd-dataset")
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise

def load_image_paths(base_path):
    """Load paths for MS and Normal images."""
    ms_path = os.path.join(base_path, "MS")
    normal_path = os.path.join(base_path, "Normal")

    ms_images = glob.glob(os.path.join(ms_path, "*.png"))
    normal_images = glob.glob(os.path.join(normal_path, "*.jpg"))

    logger.info(f"Loaded {len(ms_images)} MS images and {len(normal_images)} Normal images")
    return ms_images, normal_images

def balance_dataset(ms_images, normal_images, random_state=42):
    """Balance the dataset by sampling Normal images to match MS count."""
    random.seed(random_state)
    ms_selected = ms_images
    normal_selected = random.sample(normal_images, len(ms_selected))

    all_images = ms_selected + normal_selected
    labels = [1] * len(ms_selected) + [0] * len(normal_selected)

    combined = list(zip(all_images, labels))
    random.shuffle(combined)
    all_images, labels = zip(*combined)

    logger.info(f"Balanced dataset: {len(all_images)} images ({sum(labels)} MS, {len(labels)-sum(labels)} Normal)")
    return list(all_images), list(labels)

def extract_6_features_simple(img_path, thresholds):
    """
    Extract 6 features from an MRI image.

    Args:
        img_path (str): Path to the image
        thresholds (dict): Thresholds for white and black pixels

    Returns:
        np.array: Array of 6 features [luminosity, contrast, %white, %black, %gray, edges]
    """
    try:
        # Load image
        img_pil = Image.open(img_path)
        img_gray = img_pil.convert('L')
        img_array = np.array(img_gray, dtype=np.float32)

        if img_array.size == 0:
            raise ValueError("Empty image")

        # Features
        luminosity = np.mean(img_array)
        contrast = np.std(img_array)

        seuil_blanc = thresholds['blanc']
        seuil_noir = thresholds['noir']

        pourcentage_blanc = (np.sum(img_array > seuil_blanc) / img_array.size) * 100
        pourcentage_noir = (np.sum(img_array < seuil_noir) / img_array.size) * 100
        pourcentage_gris = 100 - pourcentage_blanc - pourcentage_noir

        # Edges
        try:
            grad_x = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            bordures = np.mean(magnitude)
        except:
            bordures = np.var(img_array)

        features = np.array([luminosity, contrast, pourcentage_blanc, pourcentage_noir, pourcentage_gris, bordures])

        if np.isnan(features).any():
            features = np.nan_to_num(features, nan=0.0)

        return features

    except Exception as e:
        logger.warning(f"Error processing {os.path.basename(img_path)}: {e}")
        return np.array([128.0, 50.0, 20.0, 20.0, 60.0, 100.0])

def extract_features_batch(image_paths, thresholds):
    """Extract features for a batch of images."""
    features = []
    for path in image_paths:
        feat = extract_6_features_simple(path, thresholds)
        features.append(feat)
    return np.array(features)

def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split
    stratify_param = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)
