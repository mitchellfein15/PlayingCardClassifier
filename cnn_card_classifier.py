import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

class CNNCardClassifier:
    def __init__(self, data_dir='card-data'):
        """Initialize the CNN card classifier"""
        self.data_dir = data_dir
        self.model = None
        self.label_encoder = None
        self.img_size = (224, 224)  # Standard size for MobileNetV2
        self.classes = None
        
    def load_and_preprocess_data(self):
        """Load training data and extract card images"""
        print("Loading training data...")
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory '{self.data_dir}' not found")
        
        train_csv_path = os.path.join(self.data_dir, 'train_labels.csv')
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Training labels file '{train_csv_path}' not found")
        
        train_df = pd.read_csv(train_csv_path)
        
        # Check if required columns exist
        required_columns = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        missing_columns = [col for col in required_columns if col not in train_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Get unique classes
        self.classes = sorted(train_df['class'].unique())
        print(f"Found {len(self.classes)} unique card classes")
        
        # Check class distribution
        class_counts = train_df['class'].value_counts()
        print(f"\nClass distribution:")
        print(f"Classes with 1 sample: {len(class_counts[class_counts == 1])}")
        print(f"Classes with 2+ samples: {len(class_counts[class_counts >= 2])}")
        
        # Filter out classes with insufficient samples for training
        min_samples_per_class = 2
        valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
        print(f"\nUsing {len(valid_classes)} classes with at least {min_samples_per_class} samples")
        
        if len(valid_classes) == 0:
            raise ValueError("No classes have sufficient samples for training. Need at least 2 samples per class.")
        
        # Filter dataframe to only include valid classes
        train_df_filtered = train_df[train_df['class'].isin(valid_classes)].copy()
        print(f"Filtered training samples: {len(train_df_filtered)}")
        
        # Update classes to only include valid ones
        self.classes = sorted(valid_classes)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        train_df_filtered['label_encoded'] = self.label_encoder.fit_transform(train_df_filtered['class'])
        
        # Prepare data
        images = []
        labels = []
        skipped_files = []
        
        print("Extracting card images from training data...")
        for idx, row in train_df_filtered.iterrows():
            try:
                # Load image
                img_path = os.path.join(self.data_dir, 'train', 'train', row['filename'])
                if not os.path.exists(img_path):
                    skipped_files.append(f"{row['filename']}: File not found")
                    continue
                    
                img = cv2.imread(img_path)
                if img is None:
                    skipped_files.append(f"{row['filename']}: Could not read image")
                    continue
                
                # Extract card region using bounding box
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                
                # Ensure bounding box coordinates are valid
                if x1 >= x2 or y1 >= y2:
                    skipped_files.append(f"{row['filename']}: Invalid bounding box coordinates")
                    continue
                
                # Ensure coordinates are within image bounds
                img_height, img_width = img.shape[:2]
                if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                    skipped_files.append(f"{row['filename']}: Bounding box outside image bounds")
                    continue
                    
                card_img = img[y1:y2, x1:x2]
                
                if card_img.size == 0:
                    skipped_files.append(f"{row['filename']}: Empty card image")
                    continue
                
                # Resize to standard size
                card_img = cv2.resize(card_img, self.img_size)
                
                # Convert BGR to RGB (OpenCV uses BGR, Keras expects RGB)
                card_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
                
                # Normalize pixel values
                card_img = card_img.astype(np.float32) / 255.0
                
                images.append(card_img)
                labels.append(row['label_encoded'])
                
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(train_df_filtered)} images")
                    
            except Exception as e:
                error_msg = f"{row['filename']}: {str(e)}"
                skipped_files.append(error_msg)
                print(f"Error processing {row['filename']}: {e}")
                continue
        
        print(f"Successfully processed {len(images)} images")
        
        if len(skipped_files) > 0:
            print(f"\nSkipped {len(skipped_files)} files due to errors:")
            for i, msg in enumerate(skipped_files[:10]):  # Show first 10 errors
                print(f"  {i+1}. {msg}")
            if len(skipped_files) > 10:
                print(f"  ... and {len(skipped_files) - 10} more")
        
        if len(images) == 0:
            raise ValueError("No valid images could be processed. Check your data paths and bounding box coordinates.")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Check if we can use stratified split
        unique_labels, label_counts = np.unique(y, return_counts=True)
        min_samples = np.min(label_counts)
        
        if min_samples >= 2:
            print("Using stratified train-test split...")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            print("Warning: Some classes have insufficient samples. Using random split...")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def build_model(self, num_classes):
        """Build a simple CNN model using transfer learning"""
        print("Building CNN model...")
        
        # Use MobileNetV2 as base (pre-trained on ImageNet)
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Build the full model
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model summary:")
        model.summary()
        
        return model
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=20, batch_size=32):
        """Train the CNN model"""
        print(f"\nTraining model for {epochs} epochs...")
        
        # Data augmentation for training
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # Don't flip cards horizontally
            fill_mode='nearest'
        )
        
        # Train the model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_val, y_val):
        """Evaluate the trained model"""
        print("\nEvaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_val, y_pred_classes)
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        class_names = [self.label_encoder.inverse_transform([i])[0] for i in range(len(self.classes))]
        print(classification_report(y_val, y_pred_classes, target_names=class_names))
        
        return accuracy, y_pred_classes
    
    def save_model(self, filepath='cnn_card_model.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='cnn_card_model.h5'):
        """Load a trained model"""
        if os.path.exists(filepath):
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return True
        else:
            print(f"Model file {filepath} not found")
            return False
    
    def predict_single_image(self, image):
        """Predict the card in a single image"""
        if self.model is None:
            return "Model not loaded"
        
        # Preprocess the image
        if len(image.shape) == 3:
            # If image has 3 channels, convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, self.img_size)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = self.model.predict(image)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Get class name
        class_name = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return class_name, confidence
    
    def start_camera(self):
        """Start real-time camera prediction"""
        if self.model is None:
            print("Please load or train a model first!")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Camera started! Press 'q' to quit, 's' to save image")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Make prediction
            prediction, confidence = self.predict_single_image(frame)
            
            # Draw prediction on frame
            cv2.putText(frame, f"Card: {prediction}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Card Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_card_{prediction.replace(' ', '_')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved as {filename}")
        
        cap.release()
        cv2.destroyAllWindows()

    def diagnose_data_issues(self):
        """Diagnose potential issues with the training data"""
        print("Diagnosing training data...")
        
        try:
            train_df = pd.read_csv(os.path.join(self.data_dir, 'train_labels.csv'))
            
            print(f"Total samples: {len(train_df)}")
            print(f"Total classes: {len(train_df['class'].unique())}")
            
            # Check for missing values
            missing_values = train_df.isnull().sum()
            if missing_values.sum() > 0:
                print(f"\nMissing values found:")
                print(missing_values[missing_values > 0])
            
            # Check class distribution
            class_counts = train_df['class'].value_counts()
            print(f"\nClass distribution:")
            print(f"Classes with 1 sample: {len(class_counts[class_counts == 1])}")
            print(f"Classes with 2-5 samples: {len(class_counts[(class_counts >= 2) & (class_counts <= 5)])}")
            print(f"Classes with 6+ samples: {len(class_counts[class_counts >= 6])}")
            
            # Show classes with only 1 sample
            single_sample_classes = class_counts[class_counts == 1].index.tolist()
            if single_sample_classes:
                print(f"\nClasses with only 1 sample (will be excluded):")
                for cls in single_sample_classes[:10]:  # Show first 10
                    print(f"  - {cls}")
                if len(single_sample_classes) > 10:
                    print(f"  ... and {len(single_sample_classes) - 10} more")
            
            # Check bounding box validity
            invalid_bbox = train_df[
                (train_df['xmin'] >= train_df['xmax']) | 
                (train_df['ymin'] >= train_df['ymax']) |
                (train_df['xmin'] < 0) | (train_df['ymin'] < 0)
            ]
            
            if len(invalid_bbox) > 0:
                print(f"\nFound {len(invalid_bbox)} samples with invalid bounding boxes:")
                print(invalid_bbox[['filename', 'xmin', 'ymin', 'xmax', 'ymax']].head())
            
            # Check file existence
            missing_files = []
            for filename in train_df['filename'].unique()[:20]:  # Check first 20 files
                file_path = os.path.join(self.data_dir, 'train', 'train', filename)
                if not os.path.exists(file_path):
                    missing_files.append(filename)
            
            if missing_files:
                print(f"\nFound {len(missing_files)} missing image files (showing first few):")
                for filename in missing_files[:5]:
                    print(f"  - {filename}")
            
            return True
            
        except Exception as e:
            print(f"Error during data diagnosis: {e}")
            return False

def main():
    """Main function to run the CNN card classifier"""
    print("CNN Card Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = CNNCardClassifier()
    
    # First, diagnose any data issues
    print("\nStep 1: Data Diagnosis")
    print("-" * 30)
    if not classifier.diagnose_data_issues():
        print("Data diagnosis failed. Please check your data directory and files.")
        return
    
    # Check if model already exists
    print("\nStep 2: Model Loading")
    print("-" * 30)
    if classifier.load_model():
        print("Loaded existing model!")
    else:
        print("Training new model...")
        
        try:
            # Load and preprocess data
            print("\nStep 3: Data Loading and Preprocessing")
            print("-" * 40)
            X_train, X_val, y_train, y_val = classifier.load_and_preprocess_data()
            
            # Build model
            print("\nStep 4: Model Building")
            print("-" * 25)
            classifier.model = classifier.build_model(len(classifier.classes))
            
            # Train model
            print("\nStep 5: Model Training")
            print("-" * 25)
            history = classifier.train_model(X_train, X_val, y_train, y_val, epochs=15)
            
            # Evaluate model
            print("\nStep 6: Model Evaluation")
            print("-" * 30)
            accuracy, predictions = classifier.evaluate_model(X_val, y_val)
            
            # Save model
            print("\nStep 7: Model Saving")
            print("-" * 25)
            classifier.save_model()
            
            # Plot training history
            try:
                print("\nStep 8: Generating Training Plots")
                print("-" * 35)
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Model Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('training_history.png')
                print("Training plots saved to 'training_history.png'")
                plt.show()
            except Exception as e:
                print(f"Warning: Could not generate training plots: {e}")
        
        except Exception as e:
            print(f"\nError during training: {e}")
            print("Please check your data and try again.")
            return
    
    # Start camera
    print("\nStep 9: Starting Camera")
    print("-" * 25)
    print("Starting camera for real-time prediction...")
    try:
        classifier.start_camera()
    except Exception as e:
        print(f"Error starting camera: {e}")
        print("Camera functionality may not be available on your system.")

if __name__ == "__main__":
    main()
