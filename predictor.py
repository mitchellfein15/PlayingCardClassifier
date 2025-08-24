from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import os

# Load the data
print("Loading data...")
train = pd.read_csv('card-data/train_labels.csv')
test = pd.read_csv('card-data/test_labels.csv')

print(f"Training data shape: {train.shape}")
print(f"Test data shape: {test.shape}")

# Data preprocessing
def preprocess_data(df):
    """Preprocess the data and engineer features"""
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Ensure class column is string type and handle any missing values
    df_processed['class'] = df_processed['class'].astype(str)
    
    # Extract card rank and suit from class using newer pandas syntax
    import re
    
    # Extract rank (first word before "of")
    df_processed['rank'] = df_processed['class'].apply(lambda x: re.search(r'(\w+)', x).group(1).lower() if re.search(r'(\w+)', x) else 'unknown')
    
    # Extract suit (word after "of")
    df_processed['suit'] = df_processed['class'].apply(lambda x: re.search(r'of (\w+)', x).group(1).lower() if re.search(r'of (\w+)', x) else 'unknown')
    
    # Handle special cases for ranks
    rank_mapping = {
        'ace': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
        'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'jack': 11, 'queen': 12, 'king': 13
    }
    df_processed['rank_numeric'] = df_processed['rank'].map(rank_mapping)
    
    # Encode suits numerically
    suit_mapping = {'hearts': 1, 'diamonds': 2, 'clubs': 3, 'spades': 4}
    df_processed['suit_numeric'] = df_processed['suit'].map(suit_mapping)
    
    # Calculate bounding box features
    df_processed['bbox_width'] = df_processed['xmax'] - df_processed['xmin']
    df_processed['bbox_height'] = df_processed['ymax'] - df_processed['ymin']
    df_processed['bbox_area'] = df_processed['bbox_width'] * df_processed['bbox_height']
    
    # Handle division by zero for aspect ratio
    df_processed['bbox_aspect_ratio'] = np.where(
        df_processed['bbox_height'] != 0,
        df_processed['bbox_width'] / df_processed['bbox_height'],
        1.0  # Default aspect ratio when height is 0
    )
    
    # Calculate relative positions (normalized by image dimensions)
    df_processed['x_center_rel'] = (df_processed['xmin'] + df_processed['xmax']) / (2 * df_processed['width'])
    df_processed['y_center_rel'] = (df_processed['ymin'] + df_processed['ymax']) / (2 * df_processed['height'])
    
    # Calculate relative bounding box dimensions
    df_processed['bbox_width_rel'] = df_processed['bbox_width'] / df_processed['width']
    df_processed['bbox_height_rel'] = df_processed['bbox_height'] / df_processed['height']
    df_processed['bbox_area_rel'] = df_processed['bbox_area'] / (df_processed['width'] * df_processed['height'])
    
    # Image aspect ratio - handle division by zero
    df_processed['image_aspect_ratio'] = np.where(
        df_processed['height'] != 0,
        df_processed['width'] / df_processed['height'],
        1.0  # Default aspect ratio when height is 0
    )
    
    # Bounding box position features
    df_processed['xmin_rel'] = df_processed['xmin'] / df_processed['width']
    df_processed['ymin_rel'] = df_processed['ymin'] / df_processed['height']
    df_processed['xmax_rel'] = df_processed['xmax'] / df_processed['width']
    df_processed['ymax_rel'] = df_processed['ymax'] / df_processed['height']
    
    # Distance from center
    df_processed['distance_from_center'] = np.sqrt(
        (df_processed['x_center_rel'] - 0.5)**2 + (df_processed['y_center_rel'] - 0.5)**2
    )
    
    # Card type features (face cards vs number cards)
    df_processed['is_face_card'] = df_processed['rank_numeric'].isin([1, 11, 12, 13]).astype(int)
    df_processed['is_ace'] = (df_processed['rank_numeric'] == 1).astype(int)
    df_processed['is_royal'] = df_processed['rank_numeric'].isin([11, 12, 13]).astype(int)
    
    # Suit color features (red vs black)
    df_processed['is_red_suit'] = df_processed['suit'].isin(['hearts', 'diamonds']).astype(int)
    
    return df_processed

# Preprocess both datasets
print("Preprocessing training data...")
train_processed = preprocess_data(train)
print("Preprocessing test data...")
test_processed = preprocess_data(test)

# Define feature columns (predictors)
predictors = [
    'width', 'height', 'rank_numeric', 'suit_numeric',
    'bbox_width', 'bbox_height', 'bbox_area', 'bbox_aspect_ratio',
    'x_center_rel', 'y_center_rel', 'bbox_width_rel', 'bbox_height_rel',
    'bbox_area_rel', 'image_aspect_ratio', 'xmin_rel', 'ymin_rel',
    'xmax_rel', 'ymax_rel', 'distance_from_center', 'is_face_card',
    'is_ace', 'is_royal', 'is_red_suit'
]

print(f"Number of features: {len(predictors)}")
print("Features:", predictors)

# Prepare the data for training
X_train = train_processed[predictors].fillna(0)  # Fill any NaN values with 0
X_test = test_processed[predictors].fillna(0)

# Replace any remaining infinite values with large finite numbers
X_train = X_train.replace([np.inf, -np.inf], 1e6)
X_test = X_test.replace([np.inf, -np.inf], 1e6)

# Encode the target variable (class) - handle unseen classes
le = LabelEncoder()
y_train = le.fit_transform(train_processed['class'])

# For test set, only keep samples with classes that were seen during training
test_mask = test_processed['class'].isin(le.classes_)
X_test_filtered = X_test[test_mask]
y_test_filtered = test_processed['class'][test_mask]

# Encode the filtered test labels
y_test = le.transform(y_test_filtered)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test_filtered.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Number of test samples after filtering: {len(y_test)}")
print(f"Classes in training set: {len(le.classes_)}")
print(f"Classes in filtered test set: {len(set(y_test))}")

# Display some sample data
print("\nSample training features:")
print(X_train.head())
print("\nSample training labels:")
print(pd.DataFrame({'original_class': train_processed['class'], 'encoded_class': y_train}).head())

# Now you have clean X_train, X_test, y_train, y_test variables
# and a comprehensive set of engineered features in the predictors array

# Initialize and train the Random Forest
print("\nTraining Random Forest classifier...")
rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the filtered test set
y_pred = rf.predict(X_test_filtered)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Show feature importance
feature_importance = pd.DataFrame({
    'feature': predictors,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))

# Show detailed classification report
print("\nClassification Report:")
# Get unique classes that are actually present in the test set
test_classes = np.unique(y_test)
test_class_names = [le.classes_[i] for i in test_classes]
print(classification_report(y_test, y_pred, target_names=test_class_names))





