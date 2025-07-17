"""
Edge AI Recyclable Items Classifier
Author: Student
Date: July 2025

This module implements a lightweight CNN model for classifying recyclable items
that can be deployed on edge devices using TensorFlow Lite.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from pathlib import Path

class RecyclableClassifier:
    """
    A lightweight CNN classifier for recyclable items detection.
    Designed for edge deployment with TensorFlow Lite.
    """
    
    def __init__(self):
        self.model = None
        self.tflite_model = None
        self.class_names = ['plastic', 'paper', 'glass', 'metal', 'organic']
        self.input_shape = (224, 224, 3)
        
    def create_lightweight_model(self, num_classes=5):
        """
        Create a lightweight CNN model optimized for edge devices.
        
        Args:
            num_classes (int): Number of classification classes
            
        Returns:
            tf.keras.Model: Compiled lightweight model
        """
        model = keras.Sequential([
            # First Conv Block
            keras.layers.Conv2D(32, 3, activation='relu', input_shape=self.input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2),
            
            # Second Conv Block
            keras.layers.Conv2D(64, 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D(2),
            
            # Third Conv Block
            keras.layers.Conv2D(128, 3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.GlobalAveragePooling2D(),
            
            # Classification Head
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def generate_synthetic_data(self, num_samples=1000):
        """
        Generate synthetic data for demonstration purposes.
        In real implementation, you would load actual recyclable item images.
        
        Args:
            num_samples (int): Number of synthetic samples to generate
            
        Returns:
            tuple: (X, y) training data
        """
        np.random.seed(42)
        
        # Generate synthetic image data
        X = np.random.rand(num_samples, 224, 224, 3).astype(np.float32)
        
        # Generate synthetic labels
        y = np.random.randint(0, len(self.class_names), num_samples)
        y_categorical = keras.utils.to_categorical(y, len(self.class_names))
        
        return X, y_categorical
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=20):
        """
        Train the recyclable classifier model.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs (int): Number of training epochs
            
        Returns:
            keras.callbacks.History: Training history
        """
        self.model = self.create_lightweight_model()
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_data=(X_val, y_val),
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def convert_to_tflite(self, save_path='recyclable_model.tflite'):
        """
        Convert the trained model to TensorFlow Lite format for edge deployment.
        
        Args:
            save_path (str): Path to save the TFLite model
            
        Returns:
            bytes: TFLite model bytes
        """
        if self.model is None:
            raise ValueError("Model must be trained before conversion")
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Optimization settings for edge devices
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Convert model
        self.tflite_model = converter.convert()
        
        # Save to file
        with open(save_path, 'wb') as f:
            f.write(self.tflite_model)
        
        print(f"TFLite model saved to {save_path}")
        print(f"Model size: {len(self.tflite_model) / 1024:.2f} KB")
        
        return self.tflite_model
    
    def test_tflite_model(self, test_data, model_path='recyclable_model.tflite'):
        """
        Test the TFLite model performance.
        
        Args:
            test_data (np.array): Test images
            model_path (str): Path to TFLite model
            
        Returns:
            np.array: Predictions
        """
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        # Get input and output tensors
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        predictions = []
        
        for image in test_data:
            # Prepare input
            input_data = np.expand_dims(image, axis=0).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            interpreter.invoke()
            
            # Get output
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output_data[0])
        
        return np.array(predictions)
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance with detailed metrics.
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Classification report
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'classification_report': report
        }
    
    def plot_training_history(self, history):
        """
        Plot training history for analysis.
        
        Args:
            history: Keras training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """
    Main function to demonstrate the Edge AI recyclable classifier.
    """
    print("🌱 Edge AI Recyclable Items Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = RecyclableClassifier()
    
    # Generate synthetic data (replace with real data in production)
    print("📊 Generating synthetic dataset...")
    X, y = classifier.generate_synthetic_data(num_samples=1000)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("\n🚀 Training the model...")
    history = classifier.train_model(X_train, y_train, X_val, y_val, epochs=15)
    
    # Evaluate model
    print("\n📈 Evaluating model performance...")
    metrics = classifier.evaluate_model(X_test, y_test)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Loss: {metrics['loss']:.4f}")
    
    # Convert to TFLite
    print("\n📱 Converting to TensorFlow Lite...")
    classifier.convert_to_tflite('recyclable_model.tflite')
    
    # Test TFLite model
    print("\n🔬 Testing TFLite model...")
    tflite_predictions = classifier.test_tflite_model(X_test[:5])
    original_predictions = classifier.model.predict(X_test[:5])
    
    print("Comparison of original vs TFLite predictions:")
    for i in range(5):
        orig_class = np.argmax(original_predictions[i])
        tflite_class = np.argmax(tflite_predictions[i])
        print(f"Sample {i+1}: Original={classifier.class_names[orig_class]}, "
              f"TFLite={classifier.class_names[tflite_class]}")
    
    # Plot training history
    classifier.plot_training_history(history)
    
    print("\n✅ Edge AI prototype completed successfully!")

if __name__ == "__main__":
    main()
