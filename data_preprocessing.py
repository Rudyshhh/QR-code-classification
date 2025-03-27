import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from skimage import feature, color
from sklearn.model_selection import train_test_split

class QRCodeDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.first_prints_dir = os.path.join(data_dir, 'First Print')
        self.second_prints_dir = os.path.join(data_dir, 'Second Print')
        
    def load_images(self):
        """
        Load images from first and second print directories
        Returns:
        - images: List of loaded images
        - labels: Corresponding binary labels (0: first print, 1: second print)
        """
        images = []
        labels = []
        
        # Load first print images
        for img_name in os.listdir(self.first_prints_dir):
            img_path = os.path.join(self.first_prints_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150,150))
            images.append(img)
            labels.append(0)
        
        # Load second print images
        for img_name in os.listdir(self.second_prints_dir):
            img_path = os.path.join(self.second_prints_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150,150))
            images.append(img)
            labels.append(1)
        
        return np.array(images), np.array(labels)
    
    def analyze_dataset(self):
        """
        Perform basic dataset analysis
        """
        images, labels = self.load_images()
        
        # Dataset composition
        unique, counts = np.unique(labels, return_counts=True)
        print("Dataset Composition:")
        print(f"First Prints (0): {counts[0]}")
        print(f"Second Prints (1): {counts[1]}")
        
        # Image statistics
        print("\nImage Statistics:")
        print(f"Total Images: {len(images)}")
        print(f"Image Shape: {images[0].shape}")
        
        # Visualize image differences
        plt.figure(figsize=(15, 5))
        
        # First print average
        plt.subplot(131)
        first_prints = images[labels == 0]
        plt.title("First Print Average")
        plt.imshow(np.mean(first_prints, axis=0), cmap='gray')
        
        # Second print average
        plt.subplot(132)
        second_prints = images[labels == 1]
        plt.title("Second Print Average")
        plt.imshow(np.mean(second_prints, axis=0), cmap='gray')
        
        # Difference visualization
        plt.subplot(133)
        plt.title("First vs Second Print Difference")
        plt.imshow(np.mean(first_prints, axis=0) - np.mean(second_prints, axis=0), cmap='coolwarm')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis.png')
        plt.close()
    
    def feature_extraction(self):
        """
        Extract advanced features from QR code images
        """
        images, labels = self.load_images()
        features = []
        
        for img in images:
            # Local Binary Patterns (LBP)
            lbp = feature.local_binary_pattern(img, P=8, R=1, method='uniform')
            
            # Histogram of Oriented Gradients (HOG)
            hog_features = feature.hog(img, 
                                       orientations=9, 
                                       pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2), 
                                       transform_sqrt=True)
            
            # Combine features
            combined_features = np.concatenate([
                np.histogram(lbp.ravel(), bins=np.arange(10))[0],
                hog_features
            ])
            
            features.append(combined_features)
        
        return np.array(features), labels

# Example usage
if __name__ == "__main__":
    processor = QRCodeDataProcessor('data')
    processor.analyze_dataset()
    
    X_features, y_labels = processor.feature_extraction()
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.2, random_state=42
    )