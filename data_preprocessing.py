import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from skimage import feature, color, filters, measure
from scipy import stats
from sklearn.model_selection import train_test_split

class QRCodeDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.first_prints_dir = os.path.join(data_dir, 'First Print')
        self.second_prints_dir = os.path.join(data_dir, 'Second Print')
        
    def load_images(self):
        images = []
        labels = []
        
        for img_name in os.listdir(self.first_prints_dir):
            img_path = os.path.join(self.first_prints_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150,150))
            images.append(img)
            labels.append(0)
        
        for img_name in os.listdir(self.second_prints_dir):
            img_path = os.path.join(self.second_prints_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (150,150))
            images.append(img)
            labels.append(1)
        
        return np.array(images), np.array(labels)
    
    def analyze_dataset(self):
        images, labels = self.load_images()
        
        unique, counts = np.unique(labels, return_counts=True)
        print("Dataset Composition:")
        print(f"First Prints (0): {counts[0]}")
        print(f"Second Prints (1): {counts[1]}")
        
        print("\nImage Statistics:")
        print(f"Total Images: {len(images)}")
        print(f"Image Shape: {images[0].shape}")
        
        plt.figure(figsize=(20, 5))
        
        plt.subplot(141)
        first_prints = images[labels == 0]
        plt.title("First Print Average")
        plt.imshow(np.mean(first_prints, axis=0), cmap='gray')
        
        plt.subplot(142)
        second_prints = images[labels == 1]
        plt.title("Second Print Average")
        plt.imshow(np.mean(second_prints, axis=0), cmap='gray')
        
        plt.subplot(143)
        plt.title("First vs Second Print Difference")
        plt.imshow(np.mean(first_prints, axis=0) - np.mean(second_prints, axis=0), cmap='coolwarm')
        
        plt.subplot(144)
        noise_diff = np.std(first_prints, axis=0) - np.std(second_prints, axis=0)
        plt.title("Noise Difference")
        plt.imshow(noise_diff, cmap='coolwarm')
        
        plt.tight_layout()
        plt.savefig('dataset_analysis_enhanced.png')
        plt.close()
    
    def feature_extraction(self):
        images, labels = self.load_images()
        features = []
        
        for img in images:
            lbp = feature.local_binary_pattern(img, P=8, R=1, method='uniform')
            lbp_hist = np.histogram(lbp.ravel(), bins=np.arange(10))[0]
            
            hog_features = feature.hog(img, 
                                       orientations=9, 
                                       pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2), 
                                       transform_sqrt=True)
            
            statistical_features = [
                np.mean(img),       
                np.std(img),        
                stats.skew(img.ravel()),  
                stats.kurtosis(img.ravel())  
            ]
            
            edges = feature.canny(img)
            edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])
            edge_features = [edge_density]
            
            f_transform = np.fft.fft2(img)
            f_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shifted))
            freq_features = [
                np.mean(magnitude_spectrum),
                np.std(magnitude_spectrum)
            ]
            
            contrast = filters.difference_of_gaussians(img, 1, 5)
            entropy = measure.shannon_entropy(img)
            
            combined_features = np.concatenate([
                lbp_hist,
                hog_features,
                statistical_features,
                edge_features,
                freq_features,
                [contrast.mean(), contrast.std()],
                [entropy]
            ])
            
            features.append(combined_features)
        
        return np.array(features), labels
    
    def visualize_feature_differences(self, features, labels):
        plt.figure(figsize=(20, 10))
        
        selected_features = [
            (0, "LBP Histogram"),
            (-4, "Contrast Mean"),
            (-2, "Entropy"),
            (10, "Statistical Mean"),
            (11, "Statistical Std Dev")
        ]
        
        for i, (feature_idx, feature_name) in enumerate(selected_features, 1):
            plt.subplot(2, 3, i)
            first_print_features = features[labels == 0][:, feature_idx]
            second_print_features = features[labels == 1][:, feature_idx]
            
            sns.histplot(
                x=first_print_features, 
                label='First Print', 
                kde=True, 
                color='blue', 
                alpha=0.5
            )
            sns.histplot(
                x=second_print_features, 
                label='Second Print', 
                kde=True, 
                color='red', 
                alpha=0.5
            )
            
            plt.title(f'Distribution of {feature_name}')
            plt.xlabel('Feature Value')
            plt.ylabel('Frequency')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        plt.close()

if __name__ == "__main__":
    processor = QRCodeDataProcessor('data')
    processor.analyze_dataset()
    X_features, y_labels = processor.feature_extraction()
    processor.visualize_feature_differences(X_features, y_labels)
