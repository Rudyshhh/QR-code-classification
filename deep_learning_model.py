import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, 
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class QRCodeDeepLearningClassifier:
    def __init__(self, input_shape=(128, 128, 1)):
        self.input_shape = input_shape
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Construct a Convolutional Neural Network for QR code classification
        """
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Flatten and Dense Layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            
            # Output Layer
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def prepare_data(self, X, y, test_size=0.2):
        """
        Preprocess data for deep learning model
        """
        # Reshape and normalize images
        X = X.reshape(-1, *self.input_shape)
        X = X / 255.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, X_test, y_train, y_test, epochs=50):
        """
        Train the deep learning model with data augmentation
        """
        # Data Augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Training
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            validation_data=(X_test, y_test),
            epochs=epochs,
            steps_per_epoch=len(X_train) // 32
        )
        
        # Visualization of Training History
        plt.figure(figsize=(12, 4))
        
        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        results = self.model.evaluate(X_test, y_test)
        print("Test Results:")
        print(f"Loss: {results[0]}")
        print(f"Accuracy: {results[1]}")
        print(f"Precision: {results[2]}")
        print(f"Recall: {results[3]}")
        
        return results

# Main Execution Script
def main():
    from data_preprocessing import QRCodeDataProcessor
    
    # Load and preprocess data
    processor = QRCodeDataProcessor('./data')
    X_images, y_labels = processor.load_images()
    
    # Initialize and prepare deep learning model
    classifier = QRCodeDeepLearningClassifier(input_shape=(X_images[0].shape[0], X_images[0].shape[1], 1))
    
    X_train, X_test, y_train, y_test = classifier.prepare_data(X_images, y_labels)
    
    # Train the model
    history = classifier.train(X_train, X_test, y_train, y_test)
    
    # Evaluate performance

    classifier.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()